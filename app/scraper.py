"""Scrape onderwijsconcept data from scholenopdekaart.nl using Playwright."""

import asyncio
import io
import re
import sys
from pathlib import Path

import polars as pl
from playwright.async_api import async_playwright

# Force unbuffered output and UTF-8 encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stdout.reconfigure(line_buffering=True)

# Data directory for caching
DATA_DIR = Path(__file__).parent.parent / "data"

# All Dutch provinces for comprehensive scraping
PROVINCES = [
    "Utrecht",
    "Noord-Holland",
    "Zuid-Holland",
    "Noord-Brabant",
    "Gelderland",
    "Overijssel",
    "Limburg",
    "Friesland",
    "Groningen",
    "Drenthe",
    "Zeeland",
    "Flevoland",
]


def extract_school_id(href: str) -> str | None:
    """Extract school ID from URL path like /basisscholen/utrecht/6166/obs-het-zand/."""
    match = re.search(r"/basisscholen/[^/]+/(\d+)/", href)
    return match.group(1) if match else None


def extract_postal_code(address: str) -> str | None:
    """Extract postal code from address string like '3544 DB Utrecht'."""
    match = re.search(r"(\d{4}\s?[A-Z]{2})", address)
    return match.group(1).replace(" ", "") if match else None


async def dismiss_cookie_dialog(page):
    """Try to dismiss the cookie consent dialog if present."""
    try:
        # Look for cookie consent button/dialog and click accept
        cookie_btn = page.locator('button:has-text("Accepteren"), button:has-text("Accept"), button:has-text("akkoord")')
        if await cookie_btn.count() > 0:
            await cookie_btn.first.click()
            await page.wait_for_timeout(500)
            return True

        # Also try clicking a close button on cookie dialog
        close_btn = page.locator('dialog button, [aria-label="close"], [aria-label="sluiten"]')
        if await close_btn.count() > 0 and await close_btn.first.is_visible():
            await close_btn.first.click()
            await page.wait_for_timeout(500)
            return True
    except Exception:
        pass
    return False


async def scrape_province(page, province: str) -> list[dict]:
    """Scrape all schools from a single province."""
    schools = []
    url = f"https://scholenopdekaart.nl/zoeken/basisscholen?zoektermen={province}&weergave=Lijst"

    print(f"  Navigating to {province}...")
    await page.goto(url)

    # Wait for content to load
    try:
        await page.wait_for_selector("article", timeout=10000)
    except Exception:
        print(f"  No schools found for {province}")
        return schools

    # Try to dismiss any cookie dialog that might block interactions
    await dismiss_cookie_dialog(page)

    # Click "Bekijk meer resultaten" until all results are loaded
    load_more_clicks = 0
    max_clicks = 500  # Safety limit to prevent infinite loops

    while load_more_clicks < max_clicks:
        try:
            # Get current article count before clicking
            initial_count = await page.locator("article").count()

            # Try to find the "load more" button using multiple strategies
            # The button text may be "Bekijk meer resultaten" (mixed case)
            btn = page.locator('button:text-is("Bekijk meer resultaten")')

            # If not found, try case-insensitive search
            if await btn.count() == 0:
                btn = page.locator('button:has-text("meer resultaten")')

            # Check if button exists
            if await btn.count() == 0:
                # No more button found, we're done
                break

            # Scroll to bottom of page first to make button visible
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(500)

            # Try to click the button using JavaScript to bypass any overlays
            try:
                # Use JavaScript click to avoid "element obscured" errors
                await page.evaluate("""
                    () => {
                        const btn = document.querySelector('button');
                        const buttons = Array.from(document.querySelectorAll('button'));
                        const loadMoreBtn = buttons.find(b =>
                            b.textContent.toLowerCase().includes('meer resultaten')
                        );
                        if (loadMoreBtn) {
                            loadMoreBtn.click();
                            return true;
                        }
                        return false;
                    }
                """)
                load_more_clicks += 1

                # Wait for new content to load (wait for article count to increase)
                for _ in range(20):  # Wait up to 10 seconds
                    await page.wait_for_timeout(500)
                    new_count = await page.locator("article").count()
                    if new_count > initial_count:
                        break

                # Check if we actually loaded more
                final_count = await page.locator("article").count()
                if final_count == initial_count:
                    # No new content loaded, we're done
                    break

                if load_more_clicks % 5 == 0:
                    print(f"    Loaded more results ({load_more_clicks} clicks, {final_count} schools)...")

            except Exception:
                # Button click failed
                break

        except Exception as e:
            # Log the error and break
            if load_more_clicks > 0:
                print(f"    Stopped loading after {load_more_clicks} clicks")
            break

    # Extract all school cards
    articles = await page.query_selector_all("article")
    print(f"  Found {len(articles)} schools in {province}")

    for article in articles:
        try:
            # Get school name and URL
            name_el = await article.query_selector("h3 a")
            if not name_el:
                continue

            href = await name_el.get_attribute("href")
            school_name = await name_el.inner_text()
            school_id = extract_school_id(href) if href else None

            # Get address - scan all text content in article for postal code pattern
            article_text = await article.inner_text()
            postal_code = None
            city = None

            # Find postal code pattern (e.g., "3544 DB Utrecht" or "3544DB Utrecht")
            postal_match = re.search(r"(\d{4}\s?[A-Z]{2})\s+(\w+)", article_text)
            if postal_match:
                postal_code = postal_match.group(1).replace(" ", "")
                city = postal_match.group(2).strip()

            # Get tags (denomination, education type, student count, onderwijsconcept)
            tags = await article.query_selector_all("li")
            tag_texts = []
            for tag in tags:
                text = await tag.inner_text()
                tag_texts.append(text.strip())

            # Parse tags - onderwijsconcept is typically the 4th tag
            denomination = tag_texts[0] if len(tag_texts) > 0 else None
            education_type = tag_texts[1] if len(tag_texts) > 1 else None
            student_count_text = tag_texts[2] if len(tag_texts) > 2 else None
            onderwijsconcept = tag_texts[3] if len(tag_texts) > 3 else "Overige"

            # Parse student count
            student_count = None
            if student_count_text:
                match = re.search(r"(\d+)", student_count_text)
                if match:
                    student_count = int(match.group(1))

            schools.append(
                {
                    "scholenopdekaart_id": school_id,
                    "name": school_name.strip(),
                    "postal_code": postal_code,
                    "city": city,
                    "denomination_scraped": denomination,
                    "education_type_scraped": education_type,
                    "student_count_scraped": student_count,
                    "onderwijsconcept": onderwijsconcept,
                    "province_scraped": province,
                    "url": f"https://scholenopdekaart.nl{href}" if href else None,
                }
            )
        except Exception as e:
            print(f"    Error parsing school: {e}")
            continue

    return schools


async def scrape_all_schools(headless: bool = True) -> pl.DataFrame:
    """Scrape all schools from scholenopdekaart.nl with onderwijsconcept data."""
    all_schools = []

    print("Starting scrape of scholenopdekaart.nl...")
    print(f"Provinces to scrape: {len(PROVINCES)}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        )
        page = await context.new_page()

        for i, province in enumerate(PROVINCES, 1):
            print(f"\n[{i}/{len(PROVINCES)}] Scraping {province}...")
            try:
                schools = await scrape_province(page, province)
                all_schools.extend(schools)
                print(f"  Total schools so far: {len(all_schools)}")
            except Exception as e:
                print(f"  Error scraping {province}: {e}")
                continue

            # Small delay between provinces
            await page.wait_for_timeout(2000)

        await browser.close()

    print(f"\nScraping complete! Total schools: {len(all_schools)}")

    # Convert to DataFrame and deduplicate
    df = pl.DataFrame(all_schools)

    # Remove duplicates based on name + postal_code
    df = df.unique(subset=["name", "postal_code"], keep="first")
    print(f"After deduplication: {len(df)} schools")

    return df


def save_onderwijsconcept_data(df: pl.DataFrame) -> Path:
    """Save scraped data to parquet file."""
    DATA_DIR.mkdir(exist_ok=True)
    cache_path = DATA_DIR / "onderwijsconcept.parquet"
    df.write_parquet(cache_path)
    print(f"Saved to {cache_path}")
    return cache_path


def load_onderwijsconcept_data() -> pl.DataFrame | None:
    """Load cached onderwijsconcept data if available."""
    cache_path = DATA_DIR / "onderwijsconcept.parquet"
    if cache_path.exists():
        return pl.read_parquet(cache_path)
    return None


async def run_scraper(headless: bool = True) -> pl.DataFrame:
    """Run the full scraping pipeline."""
    df = await scrape_all_schools(headless=headless)
    save_onderwijsconcept_data(df)
    return df


# CLI entry point
if __name__ == "__main__":
    import sys

    headless = "--visible" not in sys.argv
    df = asyncio.run(run_scraper(headless=headless))

    # Print summary
    print("\n=== Summary ===")
    print(f"Total schools: {len(df)}")
    print("\nOnderwijsconcept distribution:")
    concept_counts = df.group_by("onderwijsconcept").len().sort("len", descending=True)
    print(concept_counts)
