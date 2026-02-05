"""Data loading and caching for school data from DUO."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import polars as pl

# Data source URLs - Primary schools (Basisonderwijs)
SCHOOLS_URL = (
    "https://duo.nl/open_onderwijsdata/images/02.-alle-schoolvestigingen-basisonderwijs.csv"
)
# 3 years of reference level data for averaging
SCORES_URLS = {
    "2024-2025": "https://duo.nl/open_onderwijsdata/images/10.-leerlingen-bo-referentieniveaus-2024-2025.csv",
    "2023-2024": "https://duo.nl/open_onderwijsdata/images/10.-leerlingen-bo-referentieniveaus-2023-2024.csv",
    "2022-2023": "https://duo.nl/open_onderwijsdata/images/10.-leerlingen-bo-referentieniveaus-2022-2023.csv",
}
ENROLLMENT_URL = "https://duo.nl/open_onderwijsdata/images/01.-leerlingen-po-soort-po-cluster-leeftijd-2024-2025.csv"

# Data source URLs - Secondary schools (Voortgezet onderwijs)
VO_SCHOOLS_URL = "https://duo.nl/open_onderwijsdata/images/02.-alle-vestigingen-vo.csv"
VO_EXAM_RESULTS_URL = (
    "https://duo.nl/open_onderwijsdata/images/examenkandidaten-en-geslaagden-2020-2025.csv"
)

# Exam grades with profile breakdown (geslaagden, gezakten, cijfers per profiel)
VO_EXAM_GRADES_URLS = {
    "2023-2024": "https://duo.nl/open_onderwijsdata/images/geslaagden-gezakten-en-cijfers-2023-2024.csv",
    "2022-2023": "https://duo.nl/open_onderwijsdata/images/geslaagden-gezakten-en-cijfers-2022-2023.csv",
    "2021-2022": "https://duo.nl/open_onderwijsdata/images/geslaagden,-gezakten-en-cijfers-2021-2022.csv",
}

# Doorstroom data (zittenblijvers, op-stroom, af-stroom)
VO_DOORSTROOM_URL = "https://duo.nl/open_onderwijsdata/images/05.-zittenblijvers-niveau-vestiging-2023-2024-tov-2024-2025.csv"

# Cache configuration
DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_MAX_AGE_DAYS = 30


# =============================================================================
# Geocoding functions using PDOK Locatieserver
# =============================================================================


async def geocode_address(
    client: httpx.AsyncClient,
    street: str | None,
    house_number: str | None,
    postal_code: str | None,
    city: str | None,
) -> tuple[float, float] | None:
    """Geocode a Dutch address using PDOK Locatieserver (government API)."""
    # Build full address query for maximum precision
    parts = []
    if street:
        parts.append(str(street))
    if house_number:
        parts.append(str(house_number))
    if postal_code:
        parts.append(str(postal_code))
    if city:
        parts.append(str(city))

    query = " ".join(parts)
    if not query.strip():
        return None

    try:
        response = await client.get(
            "https://api.pdok.nl/bzk/locatieserver/search/v3_1/free",
            params={"q": query, "rows": 1, "fq": "type:adres"},
        )
        data = response.json()
        if data.get("response", {}).get("docs"):
            doc = data["response"]["docs"][0]
            # centroide_ll format: "POINT(lon lat)"
            point = doc.get("centroide_ll", "")
            if point:
                coords = point.replace("POINT(", "").replace(")", "").split()
                return (float(coords[1]), float(coords[0]))  # lat, lon
    except Exception:
        pass
    return None


async def geocode_schools_async(df: pl.DataFrame) -> dict[str, tuple[float, float]]:
    """Geocode all schools using their full addresses (async version)."""
    cache_path = DATA_DIR / "coordinates.parquet"

    # Load cached coordinates
    cached: dict[str, tuple[float, float]] = {}
    if cache_path.exists():
        cached_df = pl.read_parquet(cache_path)
        for row in cached_df.iter_rows(named=True):
            cached[row["school_id"]] = (row["latitude"], row["longitude"])

    # Find schools needing geocoding
    to_geocode = []
    for row in df.iter_rows(named=True):
        if row["school_id"] not in cached:
            to_geocode.append(row)

    if to_geocode:
        print(f"Geocoding {len(to_geocode)} schools...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Process in batches to avoid overwhelming the API
            batch_size = 50
            for i in range(0, len(to_geocode), batch_size):
                batch = to_geocode[i : i + batch_size]
                tasks = [
                    geocode_address(
                        client,
                        row.get("street"),
                        row.get("house_number"),
                        row.get("postal_code"),
                        row.get("city"),
                    )
                    for row in batch
                ]
                results = await asyncio.gather(*tasks)

                for row, coords in zip(batch, results):
                    if coords:
                        cached[row["school_id"]] = coords

                # Progress indicator
                done = min(i + batch_size, len(to_geocode))
                print(f"  Geocoded {done}/{len(to_geocode)} schools...")

                # Small delay between batches
                await asyncio.sleep(0.1)

        # Save updated cache
        coords_list = [
            {"school_id": k, "latitude": v[0], "longitude": v[1]} for k, v in cached.items()
        ]
        pl.DataFrame(coords_list).write_parquet(cache_path)
        print(f"Geocoding complete. Cached {len(cached)} coordinates.")

    return cached


def geocode_schools(df: pl.DataFrame) -> pl.DataFrame:
    """Geocode all schools and add latitude/longitude columns (sync wrapper)."""
    # Run async geocoding
    cached = asyncio.run(geocode_schools_async(df))

    # Create mapping dictionaries
    lat_map = {k: v[0] for k, v in cached.items()}
    lon_map = {k: v[1] for k, v in cached.items()}

    # Add coordinates to dataframe using map_dict
    return df.with_columns(
        [
            pl.col("school_id").replace_strict(lat_map, default=None).alias("latitude"),
            pl.col("school_id").replace_strict(lon_map, default=None).alias("longitude"),
        ]
    )


def get_cache_path(name: str) -> Path:
    """Get path for cached parquet file."""
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR / f"{name}.parquet"


def is_cache_valid(cache_path: Path, max_age_days: int = CACHE_MAX_AGE_DAYS) -> bool:
    """Check if cache exists and is not too old."""
    if not cache_path.exists():
        return False

    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    age = datetime.now() - mtime
    return age < timedelta(days=max_age_days)


def download_csv(url: str) -> pl.DataFrame:
    """Download CSV from DUO and return as Polars DataFrame."""
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

        # DUO uses semicolon separator and ISO-8859-1 encoding
        content = response.content.decode("iso-8859-1")

        return pl.read_csv(
            content.encode("utf-8"),
            separator=";",
            infer_schema_length=10000,
            ignore_errors=True,
        )


def load_schools_raw(force_refresh: bool = False) -> pl.DataFrame:
    """Load school location data with caching."""
    cache_path = get_cache_path("schools_raw")

    if not force_refresh and is_cache_valid(cache_path):
        return pl.read_parquet(cache_path)

    df = download_csv(SCHOOLS_URL)
    df.write_parquet(cache_path)
    return df


def load_scores_raw(force_refresh: bool = False) -> pl.DataFrame:
    """Load reference level scores for most recent year with caching."""
    cache_path = get_cache_path("scores_raw")

    if not force_refresh and is_cache_valid(cache_path):
        return pl.read_parquet(cache_path)

    # Load most recent year for school info
    df = download_csv(SCORES_URLS["2024-2025"])
    df.write_parquet(cache_path)
    return df


def load_scores_multiyear(force_refresh: bool = False) -> pl.DataFrame:
    """Load 3 years of reference level scores and combine them.

    This matches standard methodology of using a 3-year average
    ("driejaarsgemiddelde") for more stable results, especially for small schools.
    """
    cache_path = get_cache_path("scores_3year")

    if not force_refresh and is_cache_valid(cache_path):
        return pl.read_parquet(cache_path)

    all_years = []
    for year, url in SCORES_URLS.items():
        try:
            df = download_csv(url)
            df = df.with_columns(pl.lit(year).alias("school_year"))
            all_years.append(df)
        except Exception as e:
            print(f"Warning: Could not load {year} data: {e}")

    if not all_years:
        raise ValueError("Could not load any score data")

    # Combine all years
    combined = pl.concat(all_years, how="diagonal")
    combined.write_parquet(cache_path)
    return combined


def load_enrollment_raw(force_refresh: bool = False) -> pl.DataFrame:
    """Load student enrollment data with caching."""
    cache_path = get_cache_path("enrollment_raw")

    if not force_refresh and is_cache_valid(cache_path):
        return pl.read_parquet(cache_path)

    df = download_csv(ENROLLMENT_URL)
    df.write_parquet(cache_path)
    return df


def calculate_enrollment(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate total student enrollment from age columns."""
    # Clean column names
    clean_names = {}
    for col in df.columns:
        clean = col.strip().strip('"').strip("\ufeff")
        if clean != col:
            clean_names[col] = clean
    if clean_names:
        df = df.rename(clean_names)

    # Age columns to sum (LEEFTIJD_4 through LEEFTIJD_OUDER)
    age_cols = [c for c in df.columns if c.startswith("LEEFTIJD_")]

    # Convert age columns to numeric and sum them
    for col in age_cols:
        df = df.with_columns(
            pl.col(col)
            .cast(pl.Utf8)
            .str.replace(",", ".")
            .str.strip_chars()
            .cast(pl.Int64, strict=False)
            .fill_null(0)
            .alias(col)
        )

    # Calculate total students per row
    if age_cols:
        df = df.with_columns(pl.sum_horizontal(age_cols).alias("student_count"))
    else:
        df = df.with_columns(pl.lit(0).alias("student_count"))

    # Aggregate by school (INSTELLINGSCODE) - sum all locations
    # Use VESTIGINGSCODE for location-level data
    school_id_col = "INSTELLINGSCODE" if "INSTELLINGSCODE" in df.columns else None
    postal_col = "POSTCODE" if "POSTCODE" in df.columns else None

    if school_id_col:
        # Group by school ID and postal code to get per-location enrollment
        group_cols = [school_id_col]
        if postal_col:
            group_cols.append(postal_col)

        enrollment = df.group_by(group_cols).agg(
            pl.col("student_count").sum().alias("student_count")
        )

        # Rename for consistency
        rename_dict = {}
        if school_id_col:
            rename_dict[school_id_col] = "school_id"
        if postal_col and postal_col in enrollment.columns:
            rename_dict[postal_col] = "postal_code_enroll"

        enrollment = enrollment.rename(rename_dict)
        return enrollment

    return df.select(["student_count"])


def _clean_score_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Clean column names and convert score columns to numeric."""
    # Clean column names (remove BOM and quotes if present)
    clean_names = {}
    for col in df.columns:
        clean = col.strip().strip('"').strip("\ufeff")
        if clean != col:
            clean_names[col] = clean
    if clean_names:
        df = df.rename(clean_names)

    # Map actual column names to expected names
    col_mapping = {
        "REKENEN_LAGER1F": "rekenen_lager_1f",
        "REKENEN_1F": "rekenen_1f",
        "REKENEN_1S": "rekenen_1s",
        "REKENEN_2F": "rekenen_2f",
        "LV_LAGER1F": "lv_lager_1f",
        "LV_1F": "lv_1f",
        "LV_2F": "lv_2f",
        "TV_LAGER1F": "tv_lager_1f",
        "TV_1F": "tv_1f",
        "TV_2F": "tv_2f",
    }

    rename_dict = {k: v for k, v in col_mapping.items() if k in df.columns}
    if rename_dict:
        df = df.rename(rename_dict)

    # Score columns to convert to numeric
    score_cols = [
        "rekenen_lager_1f",
        "rekenen_1f",
        "rekenen_1s",
        "rekenen_2f",
        "lv_lager_1f",
        "lv_1f",
        "lv_2f",
        "tv_lager_1f",
        "tv_1f",
        "tv_2f",
    ]

    for col in score_cols:
        if col in df.columns:
            # Handle '<5' privacy masking: replace with 2.5 (midpoint estimate)
            # This prevents inflated percentages when students are dropped from both
            # numerator and denominator
            df = df.with_columns(
                pl.when(pl.col(col).cast(pl.Utf8).str.contains("<"))
                .then(2.5)  # Midpoint estimate for masked values
                .otherwise(
                    pl.col(col)
                    .cast(pl.Utf8)
                    .str.replace(",", ".")
                    .str.strip_chars()
                    .cast(pl.Float64, strict=False)
                )
                .fill_null(0)
                .alias(col)
            )

    return df


def calculate_metrics_3year(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate 3-year average performance metrics.

    This matches standard methodology: sum raw student counts
    across 3 years, then calculate percentages from the totals.
    """
    df = _clean_score_columns(df)

    # Group by school ID and sum all score columns across years
    score_cols = [
        "rekenen_lager_1f",
        "rekenen_1f",
        "rekenen_1s",
        "rekenen_2f",
        "lv_lager_1f",
        "lv_1f",
        "lv_2f",
        "tv_lager_1f",
        "tv_1f",
        "tv_2f",
    ]

    # Keep school info from most recent year
    info_cols = [
        "INSTELLINGSCODE",
        "INSTELLINGSNAAM_VESTIGING",
        "STRAATNAAM",
        "HUISNUMMER_TOEVOEGING",
        "POSTCODE_VESTIGING",
        "PLAATSNAAM",
        "GEMEENTENAAM",
        "PROVINCIE",
        "DENOMINATIE_VESTIGING",
        "SOORT_PO",
    ]

    # Filter to available columns
    available_info = [c for c in info_cols if c in df.columns]
    available_scores = [c for c in score_cols if c in df.columns]

    # Aggregate: sum scores across years, keep latest school info
    aggregated = df.group_by("INSTELLINGSCODE").agg(
        [pl.col(c).sum().alias(c) for c in available_scores]
        + [pl.col(c).last().alias(c) for c in available_info if c != "INSTELLINGSCODE"]
        + [pl.col("school_year").n_unique().alias("years_with_data")]
    )

    # Calculate totals from 3-year sums
    aggregated = aggregated.with_columns(
        [
            (
                pl.col("rekenen_lager_1f")
                + pl.col("rekenen_1f")
                + pl.col("rekenen_1s")
                + pl.col("rekenen_2f")
            ).alias("math_total"),
            (pl.col("lv_lager_1f") + pl.col("lv_1f") + pl.col("lv_2f")).alias("reading_total"),
            (pl.col("tv_lager_1f") + pl.col("tv_1f") + pl.col("tv_2f")).alias("tv_total"),
        ]
    )

    # Calculate percentages from 3-year totals
    aggregated = aggregated.with_columns(
        [
            # Math F-niveau: 1F + 1S + 2F (at or above fundamental)
            pl.when(pl.col("math_total") > 0)
            .then(
                (pl.col("rekenen_1f") + pl.col("rekenen_1s") + pl.col("rekenen_2f"))
                / pl.col("math_total")
                * 100
            )
            .otherwise(None)
            .round(1)
            .alias("math_f_pct"),
            # Math S-niveau: 1S + 2F (streef level or higher)
            pl.when(pl.col("math_total") > 0)
            .then((pl.col("rekenen_1s") + pl.col("rekenen_2f")) / pl.col("math_total") * 100)
            .otherwise(None)
            .round(1)
            .alias("math_s_pct"),
            # Reading F-niveau: 1F + 2F
            pl.when(pl.col("reading_total") > 0)
            .then((pl.col("lv_1f") + pl.col("lv_2f")) / pl.col("reading_total") * 100)
            .otherwise(None)
            .round(1)
            .alias("reading_f_pct"),
            # Reading S-niveau: 2F only
            pl.when(pl.col("reading_total") > 0)
            .then(pl.col("lv_2f") / pl.col("reading_total") * 100)
            .otherwise(None)
            .round(1)
            .alias("reading_s_pct"),
            # Taalverzorging F-niveau: 1F + 2F
            pl.when(pl.col("tv_total") > 0)
            .then((pl.col("tv_1f") + pl.col("tv_2f")) / pl.col("tv_total") * 100)
            .otherwise(None)
            .round(1)
            .alias("tv_f_pct"),
            # Taalverzorging S-niveau: 2F only
            pl.when(pl.col("tv_total") > 0)
            .then(pl.col("tv_2f") / pl.col("tv_total") * 100)
            .otherwise(None)
            .round(1)
            .alias("tv_s_pct"),
        ]
    )

    # Combined percentages (pooled across all 3 subjects - matches standard methodology)
    # This pools all students across subjects to get ONE combined percentage
    aggregated = aggregated.with_columns(
        [
            # Combined F-niveau: (all at F or above) / (all students across 3 subjects)
            pl.when((pl.col("math_total") + pl.col("reading_total") + pl.col("tv_total")) > 0)
            .then(
                (
                    (pl.col("rekenen_1f") + pl.col("rekenen_1s") + pl.col("rekenen_2f"))
                    + (pl.col("lv_1f") + pl.col("lv_2f"))
                    + (pl.col("tv_1f") + pl.col("tv_2f"))
                )
                / (pl.col("math_total") + pl.col("reading_total") + pl.col("tv_total"))
                * 100
            )
            .otherwise(None)
            .round(1)
            .alias("combined_f_pct"),
            # Combined S-niveau: (all at S or above) / (all students across 3 subjects)
            pl.when((pl.col("math_total") + pl.col("reading_total") + pl.col("tv_total")) > 0)
            .then(
                (
                    (pl.col("rekenen_1s") + pl.col("rekenen_2f"))  # Math S = 1S + 2F
                    + pl.col("lv_2f")  # Reading S = 2F
                    + pl.col("tv_2f")  # TV S = 2F
                )
                / (pl.col("math_total") + pl.col("reading_total") + pl.col("tv_total"))
                * 100
            )
            .otherwise(None)
            .round(1)
            .alias("combined_s_pct"),
        ]
    )

    # Combined score (average of math, reading, taalverzorging S-niveau)
    aggregated = aggregated.with_columns(
        [
            (
                (
                    pl.col("math_s_pct").fill_null(0)
                    + pl.col("reading_s_pct").fill_null(0)
                    + pl.col("tv_s_pct").fill_null(0)
                )
                / 3
            )
            .round(1)
            .alias("combined_score"),
            # Data quality flag - based on 3-year total (more stable)
            pl.when(pl.col("math_total") >= 60)  # ~20 per year * 3 years
            .then(pl.lit("reliable"))
            .when(pl.col("math_total") >= 30)  # ~10 per year * 3 years
            .then(pl.lit("limited"))
            .when(pl.col("math_total") > 0)
            .then(pl.lit("insufficient"))
            .otherwise(pl.lit("no_data"))
            .alias("data_quality"),
        ]
    )

    return aggregated


def calculate_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate performance metrics from raw reference level data (single year)."""
    df = _clean_score_columns(df)

    # Calculate totals
    df = df.with_columns(
        [
            # Math total
            (
                pl.col("rekenen_lager_1f")
                + pl.col("rekenen_1f")
                + pl.col("rekenen_1s")
                + pl.col("rekenen_2f")
            ).alias("math_total"),
            # Reading total
            (pl.col("lv_lager_1f") + pl.col("lv_1f") + pl.col("lv_2f")).alias("reading_total"),
        ]
    )

    # Calculate percentages
    df = df.with_columns(
        [
            # Math F-niveau: 1F + 1S + 2F (at or above fundamental)
            pl.when(pl.col("math_total") > 0)
            .then(
                (pl.col("rekenen_1f") + pl.col("rekenen_1s") + pl.col("rekenen_2f"))
                / pl.col("math_total")
                * 100
            )
            .otherwise(None)
            .round(1)
            .alias("math_f_pct"),
            # Math S-niveau: 1S + 2F (streef level or higher)
            pl.when(pl.col("math_total") > 0)
            .then((pl.col("rekenen_1s") + pl.col("rekenen_2f")) / pl.col("math_total") * 100)
            .otherwise(None)
            .round(1)
            .alias("math_s_pct"),
            # Reading F-niveau: 1F + 2F
            pl.when(pl.col("reading_total") > 0)
            .then((pl.col("lv_1f") + pl.col("lv_2f")) / pl.col("reading_total") * 100)
            .otherwise(None)
            .round(1)
            .alias("reading_f_pct"),
            # Reading S-niveau: 2F only
            pl.when(pl.col("reading_total") > 0)
            .then(pl.col("lv_2f") / pl.col("reading_total") * 100)
            .otherwise(None)
            .round(1)
            .alias("reading_s_pct"),
        ]
    )

    # Combined score (average of math and reading S-niveau)
    df = df.with_columns(
        [
            ((pl.col("math_s_pct").fill_null(0) + pl.col("reading_s_pct").fill_null(0)) / 2)
            .round(1)
            .alias("combined_score"),
            # Data quality flag
            pl.when(pl.col("math_total") >= 20)
            .then(pl.lit("reliable"))
            .when(pl.col("math_total") >= 10)
            .then(pl.lit("limited"))
            .when(pl.col("math_total") > 0)
            .then(pl.lit("insufficient"))
            .otherwise(pl.lit("no_data"))
            .alias("data_quality"),
        ]
    )

    return df


def load_onderwijsconcept_data() -> pl.DataFrame | None:
    """Load scraped onderwijsconcept data if available."""
    cache_path = DATA_DIR / "onderwijsconcept.parquet"
    if cache_path.exists():
        return pl.read_parquet(cache_path)
    return None


def load_signaleringswaarden() -> pl.DataFrame | None:
    """Load signaleringswaarden (benchmark values) from Onderwijsinspectie data.

    The leerresultaten.ods file contains per-school signaleringswaarden which
    vary based on the schoolweging (school weighting calculated by CBS).

    - Signaleringswaarde 1F: Always 85% (fixed baseline)
    - Signaleringswaarde 1S/2F: Varies per school based on schoolweging
    - Vergelijkbare scholen: Average results of schools in same schoolweging category
    """
    ods_path = DATA_DIR / "leerresultaten.ods"
    cache_path = DATA_DIR / "signaleringswaarden.parquet"

    # Use cached parquet if available and newer than ODS
    if cache_path.exists():
        if not ods_path.exists():
            return pl.read_parquet(cache_path)
        if cache_path.stat().st_mtime > ods_path.stat().st_mtime:
            return pl.read_parquet(cache_path)

    if not ods_path.exists():
        return None

    try:
        # Read the Data sheet from leerresultaten.ods
        df = pl.read_ods(ods_path, sheet_name="Data")

        # Extract school_id from OVT (format: "BRIN|suffix" -> take BRIN part)
        df = df.with_columns(pl.col("ovt").str.split("|").list.first().alias("school_id"))

        # Parse numeric columns (they use comma as decimal separator)
        numeric_cols = [
            "Schoolweging",
            "Signaleringswaarde 1F",
            "Signaleringswaarde 1S/2F",
            "Gewogen driejaarsgemiddelde %1F",
            "Gewogen driejaarsgemiddelde %1S/2F",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col)
                    .cast(pl.Utf8)
                    .str.replace(",", ".")
                    .cast(pl.Float64, strict=False)
                    .alias(col)
                )

        # Calculate "vergelijkbare scholen" averages per schoolwegingscategorie
        # This is the average F% and S% for schools in the same category
        category_averages = df.group_by("Schoolwegingscategorie").agg(
            [
                pl.col("Gewogen driejaarsgemiddelde %1F")
                .mean()
                .round(1)
                .alias("vergelijkbaar_f_pct"),
                pl.col("Gewogen driejaarsgemiddelde %1S/2F")
                .mean()
                .round(1)
                .alias("vergelijkbaar_s_pct"),
            ]
        )

        # Join category averages back to the main dataframe
        df = df.join(category_averages, on="Schoolwegingscategorie", how="left")

        # Select relevant columns and rename for clarity
        result = df.select(
            [
                "school_id",
                pl.col("Schoolweging").alias("schoolweging"),
                pl.col("Schoolwegingscategorie").alias("schoolweging_categorie"),
                pl.col("Signaleringswaarde 1F").alias("signaleringswaarde_f"),
                pl.col("Signaleringswaarde 1S/2F").alias("signaleringswaarde_s"),
                pl.col("vergelijkbaar_f_pct"),
                pl.col("vergelijkbaar_s_pct"),
                pl.col("Uitkomst indicator 1F").alias("uitkomst_f"),
                pl.col("Uitkomst indicator 1S/2F").alias("uitkomst_s"),
            ]
        )

        # Cache as parquet for faster subsequent loads
        result.write_parquet(cache_path)
        return result

    except Exception as e:
        print(f"Warning: Could not load signaleringswaarden: {e}")
        return None


def merge_onderwijsconcept(df: pl.DataFrame) -> pl.DataFrame:
    """Merge scraped onderwijsconcept data with DUO school data."""
    concept_df = load_onderwijsconcept_data()

    if concept_df is None:
        # No scraped data available, use placeholder
        return df.with_columns(pl.lit("Overige").alias("concept"))

    # Normalize postal codes for matching (handle nulls)
    df = df.with_columns(
        pl.col("postal_code")
        .fill_null("")
        .cast(pl.Utf8)
        .str.to_uppercase()
        .str.replace_all(" ", "")
        .alias("postal_code_norm")
    )

    concept_df = concept_df.with_columns(
        pl.col("postal_code")
        .fill_null("")
        .cast(pl.Utf8)
        .str.to_uppercase()
        .str.replace_all(" ", "")
        .alias("postal_code_norm")
    )

    # Match primarily on postal code (schools at same location)
    # Use postal code only matching since school names may differ slightly
    concept_subset = (
        concept_df.select(["postal_code_norm", "onderwijsconcept"])
        .filter(pl.col("postal_code_norm") != "")
        .unique(subset=["postal_code_norm"])
    )

    # Join on postal code only
    df = df.join(
        concept_subset,
        on=["postal_code_norm"],
        how="left",
    )

    # Rename onderwijsconcept to concept and normalize case
    if "onderwijsconcept" in df.columns:
        df = df.with_columns(
            pl.col("onderwijsconcept")
            .fill_null("Overige")
            .str.to_titlecase()  # Normalize to title case
            .str.replace("Overige", "Overige")  # Ensure consistent spelling
            .alias("concept")
        ).drop("onderwijsconcept")
    else:
        df = df.with_columns(pl.lit("Overige").alias("concept"))

    # Clean up temporary columns
    cols_to_drop = ["postal_code_norm"]
    if "name_norm" in df.columns:
        cols_to_drop.append("name_norm")
    df = df.drop(cols_to_drop)

    return df


def load_combined_data(force_refresh: bool = False) -> pl.DataFrame:
    """Load and process combined school data with 3-year average metrics.

    Uses 3-year averages to match standard methodology.
    """
    cache_path = get_cache_path("combined")

    if not force_refresh and is_cache_valid(cache_path):
        return pl.read_parquet(cache_path)

    # Load raw data
    _ = load_schools_raw(force_refresh)  # Ensure schools cache is populated
    scores_3year = load_scores_multiyear(force_refresh)
    enrollment_raw = load_enrollment_raw(force_refresh)

    # Calculate 3-year average metrics
    scores = calculate_metrics_3year(scores_3year)

    # Calculate enrollment totals
    enrollment = calculate_enrollment(enrollment_raw)

    # Rename columns for clarity (DUO uses uppercase column names)
    # Note: INSTELLINGSCODE is the unique school identifier, VESTIGINGSCODE is location-specific
    rename_map = {
        "INSTELLINGSCODE": "school_id",
        "INSTELLINGSNAAM_VESTIGING": "name",
        "STRAATNAAM": "street",
        "HUISNUMMER_TOEVOEGING": "house_number",
        "POSTCODE_VESTIGING": "postal_code",
        "PLAATSNAAM": "city",
        "GEMEENTENAAM": "municipality",
        "PROVINCIE": "province",
        "DENOMINATIE_VESTIGING": "denomination",
        "SOORT_PO": "education_type",
    }

    # Select and rename columns from scores (which has location info too)
    available_cols = [c for c in rename_map.keys() if c in scores.columns]
    df = scores.select(
        available_cols
        + [
            "math_total",
            "reading_total",
            "tv_total",
            "math_f_pct",
            "math_s_pct",
            "reading_f_pct",
            "reading_s_pct",
            "tv_f_pct",
            "tv_s_pct",
            "combined_f_pct",
            "combined_s_pct",
            "combined_score",
            "data_quality",
        ]
    )

    # Apply renames for available columns
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename({old: new})

    # Try to merge with scraped onderwijsconcept data
    df = merge_onderwijsconcept(df)

    # Merge signaleringswaarden (benchmark values per school)
    signaleringswaarden = load_signaleringswaarden()
    if signaleringswaarden is not None:
        df = df.join(
            signaleringswaarden,
            on="school_id",
            how="left",
        )
        # Fill defaults for schools without signaleringswaarden data
        df = df.with_columns(
            [
                pl.col("signaleringswaarde_f").fill_null(85.0),  # Fixed 85% baseline
                pl.col("signaleringswaarde_s").fill_null(50.0),  # Conservative default
                pl.col("vergelijkbaar_f_pct").fill_null(97.0),  # National average ~97%
                pl.col("vergelijkbaar_s_pct").fill_null(60.0),  # National average ~60%
            ]
        )
    else:
        # No signaleringswaarden data available, use defaults
        df = df.with_columns(
            [
                pl.lit(85.0).alias("signaleringswaarde_f"),
                pl.lit(50.0).alias("signaleringswaarde_s"),
                pl.lit(97.0).alias("vergelijkbaar_f_pct"),
                pl.lit(60.0).alias("vergelijkbaar_s_pct"),
                pl.lit(None).cast(pl.Float64).alias("schoolweging"),
                pl.lit(None).cast(pl.Utf8).alias("schoolweging_categorie"),
                pl.lit(None).cast(pl.Utf8).alias("uitkomst_f"),
                pl.lit(None).cast(pl.Utf8).alias("uitkomst_s"),
            ]
        )

    # Merge enrollment data
    if "school_id" in enrollment.columns:
        df = df.join(
            enrollment.select(["school_id", "student_count"]),
            on="school_id",
            how="left",
        )
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Int64).alias("student_count"))

    # Fill missing values
    df = df.with_columns(
        [
            pl.col("concept").fill_null("Overige"),
            pl.col("denomination").fill_null("Onbekend"),
            pl.col("student_count").fill_null(0),
        ]
    )

    # Geocode schools to add latitude/longitude for map view
    df = geocode_schools(df)

    df.write_parquet(cache_path)
    return df


def get_filter_options(df: pl.DataFrame) -> dict:
    """Extract unique filter options with counts."""

    def get_counts(col: str) -> list[dict]:
        if col not in df.columns:
            return []
        counts = (
            df.group_by(col)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .filter(pl.col(col).is_not_null())
        )
        return [{"value": row[col], "count": row["count"]} for row in counts.iter_rows(named=True)]

    return {
        "denominations": get_counts("denomination"),
        "concepts": get_counts("concept"),
        "cities": get_counts("city"),
        "provinces": get_counts("province"),
        "school_types": get_counts("school_type"),  # For secondary: VMBO, HAVO, VWO
    }


# =============================================================================
# Secondary school (Voortgezet onderwijs) data loading
# =============================================================================


def load_vo_schools_raw(force_refresh: bool = False) -> pl.DataFrame:
    """Load secondary school location data with caching."""
    cache_path = get_cache_path("vo_schools_raw")

    if not force_refresh and is_cache_valid(cache_path):
        return pl.read_parquet(cache_path)

    df = download_csv(VO_SCHOOLS_URL)
    df.write_parquet(cache_path)
    return df


def load_vo_exam_results_raw(force_refresh: bool = False) -> pl.DataFrame:
    """Load secondary school exam results with caching."""
    cache_path = get_cache_path("vo_exam_results_raw")

    if not force_refresh and is_cache_valid(cache_path):
        return pl.read_parquet(cache_path)

    df = download_csv(VO_EXAM_RESULTS_URL)
    df.write_parquet(cache_path)
    return df


def calculate_vo_metrics(exam_df: pl.DataFrame) -> pl.DataFrame:
    """Calculate aggregated pass rates per school for secondary education.

    Aggregates exam results per school across all education types and profiles,
    calculating overall pass rate for the most recent year.
    """
    # Clean column names
    clean_names = {}
    for col in exam_df.columns:
        clean = col.strip().strip('"').strip("\ufeff")
        if clean != col:
            clean_names[col] = clean
    if clean_names:
        exam_df = exam_df.rename(clean_names)

    # We'll use the most recent year (2024-2025) for the main pass rate
    # Also calculate 3-year average for more stable results
    years = ["2024-2025", "2023-2024", "2022-2023"]

    # Convert candidate and passed columns to numeric
    for year in years:
        cand_col = f"EXAMENKANDIDATEN SCHOOLJAAR {year} - TOTAAL"
        pass_col = f"GESLAAGDEN SCHOOLJAAR {year} - TOTAAL"

        if cand_col in exam_df.columns:
            # Handle '<5' privacy masking
            exam_df = exam_df.with_columns(
                [
                    pl.when(pl.col(cand_col).cast(pl.Utf8).str.contains("<"))
                    .then(2.5)
                    .otherwise(
                        pl.col(cand_col)
                        .cast(pl.Utf8)
                        .str.replace(",", ".")
                        .cast(pl.Float64, strict=False)
                    )
                    .fill_null(0)
                    .alias(f"candidates_{year}"),
                    pl.when(pl.col(pass_col).cast(pl.Utf8).str.contains("<"))
                    .then(2.5)
                    .otherwise(
                        pl.col(pass_col)
                        .cast(pl.Utf8)
                        .str.replace(",", ".")
                        .cast(pl.Float64, strict=False)
                    )
                    .fill_null(0)
                    .alias(f"passed_{year}"),
                ]
            )

    # Aggregate per school and education type (VMBO/HAVO/VWO)
    # First, aggregate per school across all profiles/subjects
    aggregated = exam_df.group_by(["INSTELLINGSCODE", "ONDERWIJSTYPE VO"]).agg(
        [
            # Sum candidates and passed for each year
            pl.col("candidates_2024-2025").sum().alias("candidates_2024"),
            pl.col("passed_2024-2025").sum().alias("passed_2024"),
            pl.col("candidates_2023-2024").sum().alias("candidates_2023"),
            pl.col("passed_2023-2024").sum().alias("passed_2023"),
            pl.col("candidates_2022-2023").sum().alias("candidates_2022"),
            pl.col("passed_2022-2023").sum().alias("passed_2022"),
            # Keep school info
            pl.col("VESTIGINGSCODE").first().alias("VESTIGINGSCODE"),
            pl.col("INSTELLINGSNAAM VESTIGING").first().alias("school_name"),
            pl.col("GEMEENTENAAM").first().alias("municipality"),
        ]
    )

    # Calculate pass rates
    aggregated = aggregated.with_columns(
        [
            # Current year pass rate
            pl.when(pl.col("candidates_2024") > 0)
            .then(pl.col("passed_2024") / pl.col("candidates_2024") * 100)
            .otherwise(None)
            .round(1)
            .alias("pass_rate_current"),
            # 3-year totals
            (
                pl.col("candidates_2024") + pl.col("candidates_2023") + pl.col("candidates_2022")
            ).alias("candidates_3year"),
            (pl.col("passed_2024") + pl.col("passed_2023") + pl.col("passed_2022")).alias(
                "passed_3year"
            ),
        ]
    )

    # 3-year average pass rate
    aggregated = aggregated.with_columns(
        [
            pl.when(pl.col("candidates_3year") > 0)
            .then(pl.col("passed_3year") / pl.col("candidates_3year") * 100)
            .otherwise(None)
            .round(1)
            .alias("pass_rate_3year"),
            # Data quality flag
            pl.when(pl.col("candidates_3year") >= 60)
            .then(pl.lit("reliable"))
            .when(pl.col("candidates_3year") >= 30)
            .then(pl.lit("limited"))
            .when(pl.col("candidates_3year") > 0)
            .then(pl.lit("insufficient"))
            .otherwise(pl.lit("no_data"))
            .alias("data_quality"),
        ]
    )

    return aggregated


def load_vo_combined_data(force_refresh: bool = False) -> pl.DataFrame:
    """Load and process combined secondary school data with exam results.

    Returns a DataFrame with one row per school-type combination
    (e.g., school X has separate rows for VMBO and HAVO if they offer both).
    """
    cache_path = get_cache_path("vo_combined")

    if not force_refresh and is_cache_valid(cache_path):
        return pl.read_parquet(cache_path)

    # Load raw data
    schools_raw = load_vo_schools_raw(force_refresh)
    exam_raw = load_vo_exam_results_raw(force_refresh)

    # Calculate exam metrics per school and type
    exam_metrics = calculate_vo_metrics(exam_raw)

    # Clean school data column names
    clean_names = {}
    for col in schools_raw.columns:
        clean = col.strip().strip('"').strip("\ufeff")
        if clean != col:
            clean_names[col] = clean
    if clean_names:
        schools_raw = schools_raw.rename(clean_names)

    # Select relevant columns from schools data
    schools = schools_raw.select(
        [
            pl.col("INSTELLINGSCODE").alias("school_id"),
            pl.col("VESTIGINGSCODE").alias("vestiging_id"),
            pl.col("VESTIGINGSNAAM").alias("name"),
            pl.col("STRAATNAAM").alias("street"),
            pl.col("HUISNUMMER-TOEVOEGING").alias("house_number"),
            pl.col("POSTCODE").alias("postal_code"),
            pl.col("PLAATSNAAM").alias("city"),
            pl.col("GEMEENTENAAM").alias("municipality"),
            pl.col("PROVINCIE").alias("province"),
            pl.col("DENOMINATIE").alias("denomination"),
            pl.col("ONDERWIJSSTRUCTUUR").alias("education_structure"),
        ]
    )

    # Rename exam metrics columns for join
    exam_metrics = exam_metrics.rename(
        {
            "INSTELLINGSCODE": "school_id",
            "ONDERWIJSTYPE VO": "school_type",
        }
    )

    # Join schools with exam metrics
    # This creates multiple rows per school if they offer multiple types (VMBO, HAVO, VWO)
    df = schools.join(
        exam_metrics.select(
            [
                "school_id",
                "school_type",
                "pass_rate_current",
                "pass_rate_3year",
                "candidates_3year",
                "data_quality",
            ]
        ),
        on="school_id",
        how="left",
    )

    # Load and merge exam grades data (examencijfers)
    exam_grades = calculate_vo_exam_grades(force_refresh)
    if len(exam_grades) > 0:
        df = df.join(
            exam_grades.select(
                [
                    "school_id",
                    "school_type",
                    "avg_schoolexamen",
                    "avg_centraal_examen",
                    "avg_cijferlijst",
                    "national_avg_schoolexamen",
                    "national_avg_centraal_examen",
                    "national_avg_cijferlijst",
                ]
            ),
            on=["school_id", "school_type"],
            how="left",
        )

    # Fill missing values
    df = df.with_columns(
        [
            pl.col("school_type").fill_null("Onbekend"),
            pl.col("denomination").fill_null("Onbekend"),
            pl.col("data_quality").fill_null("no_data"),
        ]
    )

    # Create unique ID for each school-type combination
    df = df.with_columns(
        (pl.col("school_id") + "_" + pl.col("school_type")).alias("school_type_id")
    )

    # Geocode schools
    # Create a temp df with unique locations for geocoding
    unique_locations = df.select(
        [
            pl.col("vestiging_id").alias("school_id"),
            "street",
            "house_number",
            "postal_code",
            "city",
        ]
    ).unique(subset=["school_id"])

    # Geocode unique locations
    geocoded = geocode_schools(unique_locations)

    # Merge coordinates back
    df = df.join(
        geocoded.select(["school_id", "latitude", "longitude"]),
        left_on="vestiging_id",
        right_on="school_id",
        how="left",
        suffix="_geo",
    )

    # Drop the duplicate school_id column from join
    if "school_id_geo" in df.columns:
        df = df.drop("school_id_geo")

    df.write_parquet(cache_path)
    return df


def get_vo_filter_options(df: pl.DataFrame) -> dict:
    """Extract unique filter options for secondary schools with counts."""

    def get_counts(col: str) -> list[dict]:
        if col not in df.columns:
            return []
        counts = (
            df.group_by(col)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .filter(pl.col(col).is_not_null())
        )
        return [{"value": row[col], "count": row["count"]} for row in counts.iter_rows(named=True)]

    return {
        "denominations": get_counts("denomination"),
        "school_types": get_counts("school_type"),  # VMBO, HAVO, VWO
        "cities": get_counts("city"),
        "provinces": get_counts("province"),
        "education_structures": get_counts("education_structure"),
    }


def get_vo_available_types(df: pl.DataFrame, school_id: str) -> list[dict]:
    """Get all available school types (VMBO, HAVO, VWO) for a given school.

    Returns list of dicts with type info and school_type_id for linking.
    """
    # Filter to this school's records
    school_rows = df.filter(pl.col("school_id") == school_id)

    if len(school_rows) == 0:
        return []

    # Get unique school types with their IDs
    types = []
    type_order = {"VMBO": 1, "HAVO": 2, "VWO": 3}

    for row in school_rows.iter_rows(named=True):
        school_type = row.get("school_type")
        if school_type and school_type not in [t["type"] for t in types]:
            types.append(
                {
                    "type": school_type,
                    "school_type_id": row.get("school_type_id"),
                    "pass_rate": row.get("pass_rate_3year"),
                    "avg_grade": row.get("avg_cijferlijst"),
                }
            )

    # Sort by type order (VMBO, HAVO, VWO)
    types.sort(key=lambda x: type_order.get(x["type"], 99))
    return types


def get_vo_historical_data(school_id: str, school_type: str) -> list[dict]:
    """Get historical exam grades data for a school across multiple years.

    Returns list of dicts with year, schoolexamen, centraal_examen, cijferlijst, pass_rate.
    """
    historical = []

    for year in VO_EXAM_GRADES_URLS.keys():
        df = load_vo_exam_grades_raw(year)
        if df is None:
            continue

        # Clean column names
        clean_names = {}
        for col in df.columns:
            clean = col.strip().strip('"').strip("\ufeff")
            if clean != col:
                clean_names[col] = clean
        if clean_names:
            df = df.rename(clean_names)

        # Filter to this school and type
        # Column may be "INSTELLINGSCODE" or "INSTELLING" depending on year
        id_col = "INSTELLINGSCODE" if "INSTELLINGSCODE" in df.columns else "INSTELLING"
        df = df.filter((pl.col(id_col) == school_id) & (pl.col("ONDERWIJSTYPE VO") == school_type))

        if len(df) == 0:
            continue

        # Convert numeric columns
        for col in ["EXAMENKANDIDATEN", "GESLAAGDEN"]:
            if col in df.columns:
                df = df.with_columns(
                    pl.when(pl.col(col).cast(pl.Utf8).str.contains("<"))
                    .then(2.5)
                    .otherwise(
                        pl.col(col)
                        .cast(pl.Utf8)
                        .str.replace(",", ".")
                        .cast(pl.Float64, strict=False)
                    )
                    .fill_null(0)
                    .alias(col)
                )

        for col in [
            "GEMIDDELD CIJFER SCHOOLEXAMEN",
            "GEMIDDELD CIJFER CENTRAAL EXAMEN",
            "GEMIDDELD CIJFER CIJFERLIJST",
        ]:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col)
                    .cast(pl.Utf8)
                    .str.replace(",", ".")
                    .cast(pl.Float64, strict=False)
                    .alias(col)
                )

        # Aggregate across all profiles/subjects for this year
        agg = df.select(
            [
                (
                    (pl.col("GEMIDDELD CIJFER SCHOOLEXAMEN") * pl.col("EXAMENKANDIDATEN")).sum()
                    / pl.col("EXAMENKANDIDATEN").sum()
                )
                .round(2)
                .alias("avg_schoolexamen"),
                (
                    (pl.col("GEMIDDELD CIJFER CENTRAAL EXAMEN") * pl.col("EXAMENKANDIDATEN")).sum()
                    / pl.col("EXAMENKANDIDATEN").sum()
                )
                .round(2)
                .alias("avg_centraal_examen"),
                (
                    (pl.col("GEMIDDELD CIJFER CIJFERLIJST") * pl.col("EXAMENKANDIDATEN")).sum()
                    / pl.col("EXAMENKANDIDATEN").sum()
                )
                .round(2)
                .alias("avg_cijferlijst"),
                pl.col("EXAMENKANDIDATEN").sum().alias("total_candidates"),
                pl.col("GESLAAGDEN").sum().alias("total_passed"),
            ]
        )

        row = agg.to_dicts()[0]

        # Calculate pass rate
        pass_rate = None
        if row["total_candidates"] and row["total_candidates"] > 0:
            pass_rate = round(row["total_passed"] / row["total_candidates"] * 100, 1)

        historical.append(
            {
                "year": year,
                "avg_schoolexamen": row.get("avg_schoolexamen"),
                "avg_centraal_examen": row.get("avg_centraal_examen"),
                "avg_cijferlijst": row.get("avg_cijferlijst"),
                "total_candidates": int(row.get("total_candidates", 0)),
                "pass_rate": pass_rate,
            }
        )

    # Sort by year (most recent first)
    historical.sort(key=lambda x: x["year"], reverse=True)
    return historical


# =============================================================================
# Secondary school exam grades (examencijfers) data
# =============================================================================


def load_vo_exam_grades_raw(year: str, force_refresh: bool = False) -> pl.DataFrame | None:
    """Load secondary school exam grades data for a specific year with caching."""
    if year not in VO_EXAM_GRADES_URLS:
        return None

    cache_path = get_cache_path(f"vo_exam_grades_{year}")

    if not force_refresh and is_cache_valid(cache_path):
        return pl.read_parquet(cache_path)

    try:
        df = download_csv(VO_EXAM_GRADES_URLS[year])
        df.write_parquet(cache_path)
        return df
    except Exception as e:
        print(f"Warning: Could not load exam grades for {year}: {e}")
        return None


def calculate_vo_exam_grades(force_refresh: bool = False) -> pl.DataFrame:
    """Calculate exam grades metrics per school and education type.

    Loads data from multiple years and calculates:
    - Average grades (schoolexamen, centraal examen, cijferlijst)
    - Profile breakdown (N&T, N&G, E&M, C&M)
    - National averages for comparison ("vergelijkbare scholen")
    """
    cache_path = get_cache_path("vo_exam_grades_calculated")

    if not force_refresh and is_cache_valid(cache_path):
        return pl.read_parquet(cache_path)

    all_grades = []

    for year in VO_EXAM_GRADES_URLS.keys():
        df = load_vo_exam_grades_raw(year, force_refresh)
        if df is None:
            continue

        # Clean column names
        clean_names = {}
        for col in df.columns:
            clean = col.strip().strip('"').strip("\ufeff")
            if clean != col:
                clean_names[col] = clean
        if clean_names:
            df = df.rename(clean_names)

        # Add year column
        df = df.with_columns(pl.lit(year).alias("school_year"))
        all_grades.append(df)

    if not all_grades:
        return pl.DataFrame()

    # Combine all years
    grades_df = pl.concat(all_grades, how="diagonal")

    # Clean and convert numeric columns
    grade_cols = [
        "GEMIDDELD CIJFER SCHOOLEXAMEN",
        "GEMIDDELD CIJFER CENTRAAL EXAMEN",
        "GEMIDDELD CIJFER CIJFERLIJST",
    ]

    for col in grade_cols:
        if col in grades_df.columns:
            grades_df = grades_df.with_columns(
                pl.col(col)
                .cast(pl.Utf8)
                .str.replace(",", ".")
                .cast(pl.Float64, strict=False)
                .alias(col)
            )

    # Convert candidate counts
    for col in ["EXAMENKANDIDATEN", "GESLAAGDEN", "GEZAKTEN"]:
        if col in grades_df.columns:
            grades_df = grades_df.with_columns(
                pl.when(pl.col(col).cast(pl.Utf8).str.contains("<"))
                .then(2.5)
                .otherwise(
                    pl.col(col).cast(pl.Utf8).str.replace(",", ".").cast(pl.Float64, strict=False)
                )
                .fill_null(0)
                .alias(col)
            )

    # Map AFDELING to profile names
    profile_mapping = {
        "Cultuur en Maatschappij": "C&M",
        "Economie en Maatschappij": "E&M",
        "Natuur en Gezondheid": "N&G",
        "Natuur en Techniek": "N&T",
    }

    if "AFDELING" in grades_df.columns:
        grades_df = grades_df.with_columns(
            pl.col("AFDELING").replace(profile_mapping).alias("profile")
        )
    else:
        grades_df = grades_df.with_columns(pl.lit(None).cast(pl.Utf8).alias("profile"))

    # Calculate weighted averages per school and education type (most recent year)
    # Filter to most recent year for main metrics
    recent_year = "2023-2024"
    recent_df = grades_df.filter(pl.col("school_year") == recent_year)

    # Aggregate per school and education type
    aggregated = recent_df.group_by(["INSTELLINGSCODE", "ONDERWIJSTYPE VO"]).agg(
        [
            # Weighted average grades (weighted by number of candidates)
            (
                (pl.col("GEMIDDELD CIJFER SCHOOLEXAMEN") * pl.col("EXAMENKANDIDATEN")).sum()
                / pl.col("EXAMENKANDIDATEN").sum()
            )
            .round(2)
            .alias("avg_schoolexamen"),
            (
                (pl.col("GEMIDDELD CIJFER CENTRAAL EXAMEN") * pl.col("EXAMENKANDIDATEN")).sum()
                / pl.col("EXAMENKANDIDATEN").sum()
            )
            .round(2)
            .alias("avg_centraal_examen"),
            (
                (pl.col("GEMIDDELD CIJFER CIJFERLIJST") * pl.col("EXAMENKANDIDATEN")).sum()
                / pl.col("EXAMENKANDIDATEN").sum()
            )
            .round(2)
            .alias("avg_cijferlijst"),
            # Total counts
            pl.col("EXAMENKANDIDATEN").sum().alias("total_candidates"),
            pl.col("GESLAAGDEN").sum().alias("total_passed"),
            pl.col("GEZAKTEN").sum().alias("total_failed"),
        ]
    )

    # Calculate national averages per education type for comparison
    national_avgs = recent_df.group_by("ONDERWIJSTYPE VO").agg(
        [
            (
                (pl.col("GEMIDDELD CIJFER SCHOOLEXAMEN") * pl.col("EXAMENKANDIDATEN")).sum()
                / pl.col("EXAMENKANDIDATEN").sum()
            )
            .round(2)
            .alias("national_avg_schoolexamen"),
            (
                (pl.col("GEMIDDELD CIJFER CENTRAAL EXAMEN") * pl.col("EXAMENKANDIDATEN")).sum()
                / pl.col("EXAMENKANDIDATEN").sum()
            )
            .round(2)
            .alias("national_avg_centraal_examen"),
            (
                (pl.col("GEMIDDELD CIJFER CIJFERLIJST") * pl.col("EXAMENKANDIDATEN")).sum()
                / pl.col("EXAMENKANDIDATEN").sum()
            )
            .round(2)
            .alias("national_avg_cijferlijst"),
        ]
    )

    # Join national averages
    aggregated = aggregated.join(national_avgs, on="ONDERWIJSTYPE VO", how="left")

    # Rename columns for consistency
    aggregated = aggregated.rename(
        {
            "INSTELLINGSCODE": "school_id",
            "ONDERWIJSTYPE VO": "school_type",
        }
    )

    aggregated.write_parquet(cache_path)
    return aggregated


def get_vo_profile_breakdown(
    school_id: str, school_type: str, force_refresh: bool = False
) -> list[dict]:
    """Get detailed profile/sector breakdown for a specific school and education type.

    Aggregates data by profile (HAVO/VWO) or sector (VMBO) and returns list of dicts
    with candidates, passed, and grades.
    """
    # Load most recent year data
    df = load_vo_exam_grades_raw("2023-2024", force_refresh)
    if df is None:
        return []

    # Clean column names
    clean_names = {}
    for col in df.columns:
        clean = col.strip().strip('"').strip("\ufeff")
        if clean != col:
            clean_names[col] = clean
    if clean_names:
        df = df.rename(clean_names)

    # Filter to specific school and education type
    df = df.filter(
        (pl.col("INSTELLINGSCODE") == school_id) & (pl.col("ONDERWIJSTYPE VO") == school_type)
    )

    if len(df) == 0:
        return []

    # Define mappings based on school type
    # HAVO/VWO use profiles (profielen)
    profile_mapping = {
        "Cultuur en Maatschappij": "C&M",
        "Economie en Maatschappij": "E&M",
        "Natuur en Gezondheid": "N&G",
        "Natuur en Techniek": "N&T",
    }

    # VMBO uses sectors (sectoren) - abbreviated codes to full names
    vmbo_sector_mapping = {
        "EO": "Economie",
        "ZWE": "Zorg & Welzijn",
        "HBR": "Handel en Administratie",
        "MTE": "Metaaltechniek",
        "MTR": "Motorvoertuigen",
        "BWI": "Bouwtechniek",
        "PIE": "Elektrotechniek",
        "MVI": "Media",
        "DP": "Dienstverlening & Producten",
        "G": "Groen",
    }

    # Determine which mapping to use
    if school_type == "VMBO":
        # For VMBO, filter to known sectors (excluding empty strings)
        valid_afdelingen = list(vmbo_sector_mapping.keys())
        label_mapping = vmbo_sector_mapping
    else:
        # For HAVO/VWO, filter to main profiles only
        valid_afdelingen = list(profile_mapping.keys())
        label_mapping = profile_mapping

    # Filter to rows with valid afdelingen
    df = df.filter(pl.col("AFDELING").is_in(valid_afdelingen) & (pl.col("AFDELING") != ""))

    if len(df) == 0:
        return []

    # Convert numeric columns
    for col in ["EXAMENKANDIDATEN", "GESLAAGDEN", "GEZAKTEN"]:
        if col in df.columns:
            df = df.with_columns(
                pl.when(pl.col(col).cast(pl.Utf8).str.contains("<"))
                .then(2.5)
                .otherwise(
                    pl.col(col).cast(pl.Utf8).str.replace(",", ".").cast(pl.Float64, strict=False)
                )
                .fill_null(0)
                .alias(col)
            )

    for col in [
        "GEMIDDELD CIJFER SCHOOLEXAMEN",
        "GEMIDDELD CIJFER CENTRAAL EXAMEN",
        "GEMIDDELD CIJFER CIJFERLIJST",
    ]:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col)
                .cast(pl.Utf8)
                .str.replace(",", ".")
                .cast(pl.Float64, strict=False)
                .alias(col)
            )

    # Aggregate by afdeling
    aggregated = df.group_by("AFDELING").agg(
        [
            pl.col("EXAMENKANDIDATEN").sum().alias("candidates"),
            pl.col("GESLAAGDEN").sum().alias("passed"),
            pl.col("GEZAKTEN").sum().alias("failed"),
            # Weighted average grades
            (
                (pl.col("GEMIDDELD CIJFER SCHOOLEXAMEN") * pl.col("EXAMENKANDIDATEN")).sum()
                / pl.col("EXAMENKANDIDATEN").sum()
            )
            .round(2)
            .alias("avg_schoolexamen"),
            (
                (pl.col("GEMIDDELD CIJFER CENTRAAL EXAMEN") * pl.col("EXAMENKANDIDATEN")).sum()
                / pl.col("EXAMENKANDIDATEN").sum()
            )
            .round(2)
            .alias("avg_centraal_examen"),
            (
                (pl.col("GEMIDDELD CIJFER CIJFERLIJST") * pl.col("EXAMENKANDIDATEN")).sum()
                / pl.col("EXAMENKANDIDATEN").sum()
            )
            .round(2)
            .alias("avg_cijferlijst"),
        ]
    )

    # Calculate pass rate
    aggregated = aggregated.with_columns(
        pl.when(pl.col("candidates") > 0)
        .then((pl.col("passed") / pl.col("candidates") * 100).round(1))
        .otherwise(None)
        .alias("pass_rate")
    )

    # Build result list
    profiles = []
    for row in aggregated.iter_rows(named=True):
        afdeling = row.get("AFDELING", "")
        profiles.append(
            {
                "profile": label_mapping.get(afdeling, afdeling),
                "profile_full": afdeling if school_type != "VMBO" else afdeling,
                "candidates": int(row.get("candidates", 0)),
                "passed": int(row.get("passed", 0)),
                "failed": int(row.get("failed", 0)),
                "pass_rate": row.get("pass_rate"),
                "avg_schoolexamen": row.get("avg_schoolexamen"),
                "avg_centraal_examen": row.get("avg_centraal_examen"),
                "avg_cijferlijst": row.get("avg_cijferlijst"),
            }
        )

    # Sort by profile name
    profiles.sort(key=lambda x: x["profile"])
    return profiles


# =============================================================================
# Secondary school doorstroom (progression) data
# =============================================================================


def load_vo_doorstroom_raw(force_refresh: bool = False) -> pl.DataFrame | None:
    """Load secondary school doorstroom data (zittenblijvers, op-stroom, af-stroom)."""
    cache_path = get_cache_path("vo_doorstroom_raw")

    if not force_refresh and is_cache_valid(cache_path):
        return pl.read_parquet(cache_path)

    try:
        df = download_csv(VO_DOORSTROOM_URL)
        df.write_parquet(cache_path)
        return df
    except Exception as e:
        print(f"Warning: Could not load doorstroom data: {e}")
        return None


def calculate_vo_doorstroom(force_refresh: bool = False) -> pl.DataFrame:
    """Calculate doorstroom metrics per vestiging.

    Calculates:
    - pct_zittenblijvers: % students repeating year
    - pct_opstroom: % students moving to higher level
    - pct_afstroom: % students moving to lower level
    - pct_doorstroom: % students progressing normally
    """
    cache_path = get_cache_path("vo_doorstroom_calculated")

    if not force_refresh and is_cache_valid(cache_path):
        return pl.read_parquet(cache_path)

    df = load_vo_doorstroom_raw(force_refresh)
    if df is None:
        return pl.DataFrame()

    # Clean column names
    clean_names = {}
    for col in df.columns:
        clean = col.strip().strip('"').strip("\ufeff")
        if clean != col:
            clean_names[col] = clean
    if clean_names:
        df = df.rename(clean_names)

    # Convert string columns to numeric (handle "<5" privacy masking)
    numeric_cols = [
        "AANTAL_LEERLINGEN",
        "AANTAL_OPSTROMERS",
        "AANTAL_AFSTROMERS",
        "AANTAL_ZITTENBLIJVERS",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df = df.with_columns(
                pl.when(pl.col(col).cast(pl.Utf8).str.contains("<"))
                .then(2.5)  # Midpoint estimate for masked values
                .otherwise(
                    pl.col(col).cast(pl.Utf8).str.replace(",", ".").cast(pl.Float64, strict=False)
                )
                .fill_null(0)
                .alias(col)
            )

    # Calculate percentages
    df = df.with_columns(
        [
            pl.when(pl.col("AANTAL_LEERLINGEN") > 0)
            .then((pl.col("AANTAL_OPSTROMERS") / pl.col("AANTAL_LEERLINGEN") * 100).round(1))
            .otherwise(None)
            .alias("pct_opstroom"),
            pl.when(pl.col("AANTAL_LEERLINGEN") > 0)
            .then((pl.col("AANTAL_AFSTROMERS") / pl.col("AANTAL_LEERLINGEN") * 100).round(1))
            .otherwise(None)
            .alias("pct_afstroom"),
            pl.when(pl.col("AANTAL_LEERLINGEN") > 0)
            .then((pl.col("AANTAL_ZITTENBLIJVERS") / pl.col("AANTAL_LEERLINGEN") * 100).round(1))
            .otherwise(None)
            .alias("pct_zittenblijvers"),
        ]
    )

    # Calculate doorstroom (normal progression = remainder)
    df = df.with_columns(
        (
            100
            - pl.col("pct_opstroom").fill_null(0)
            - pl.col("pct_afstroom").fill_null(0)
            - pl.col("pct_zittenblijvers").fill_null(0)
        )
        .clip(0, 100)
        .round(1)
        .alias("pct_doorstroom")
    )

    # Rename for consistency
    df = df.rename(
        {
            "INSTELLINGSCODE": "school_id",
            "VESTIGINGSCODE": "vestiging_code",
            "AANTAL_LEERLINGEN": "total_students",
        }
    )

    df.write_parquet(cache_path)
    return df


def get_vo_doorstroom_for_school(school_id: str) -> dict | None:
    """Get aggregated doorstroom metrics for a school (all vestigingen combined)."""
    df = calculate_vo_doorstroom()
    if len(df) == 0:
        return None

    # Filter to this school and aggregate across all vestigingen
    school_df = df.filter(pl.col("school_id") == school_id)
    if len(school_df) == 0:
        return None

    # Sum totals across vestigingen, then recalculate percentages
    totals = school_df.select(
        [
            pl.col("total_students").sum().alias("total_students"),
            (pl.col("total_students") * pl.col("pct_opstroom") / 100).sum().alias("opstromers"),
            (pl.col("total_students") * pl.col("pct_afstroom") / 100).sum().alias("afstromers"),
            (pl.col("total_students") * pl.col("pct_zittenblijvers") / 100)
            .sum()
            .alias("zittenblijvers"),
        ]
    ).to_dicts()[0]

    total = totals["total_students"]
    if total == 0:
        return None

    doorstroom_count = (
        total - totals["opstromers"] - totals["afstromers"] - totals["zittenblijvers"]
    )

    return {
        "total_students": int(total),
        "pct_opstroom": round(totals["opstromers"] / total * 100, 1),
        "pct_afstroom": round(totals["afstromers"] / total * 100, 1),
        "pct_zittenblijvers": round(totals["zittenblijvers"] / total * 100, 1),
        "pct_doorstroom": round(max(0, doorstroom_count) / total * 100, 1),
    }
