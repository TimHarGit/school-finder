"""FastAPI application for School Finder."""

from pathlib import Path
from typing import Annotated

import polars as pl
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.data import (
    get_filter_options,
    get_vo_available_types,
    get_vo_doorstroom_for_school,
    get_vo_filter_options,
    get_vo_historical_data,
    get_vo_profile_breakdown,
    load_combined_data,
    load_vo_combined_data,
)

app = FastAPI(title="School Finder", description="Dutch school comparison tool")

# Setup templates and static files
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Load data on startup
_school_data: pl.DataFrame | None = None
_vo_school_data: pl.DataFrame | None = None


def get_school_data() -> pl.DataFrame:
    """Get cached primary school data, loading if necessary."""
    global _school_data
    if _school_data is None:
        _school_data = load_combined_data()
    return _school_data


def get_vo_school_data() -> pl.DataFrame:
    """Get cached secondary school data, loading if necessary."""
    global _vo_school_data
    if _vo_school_data is None:
        _vo_school_data = load_vo_combined_data()
    return _vo_school_data


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main search page."""
    df = get_school_data()
    filter_options = get_filter_options(df)

    # Get first 50 schools sorted by combined score
    schools = df.sort("combined_score", descending=True, nulls_last=True).head(50)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "schools": schools.to_dicts(),
            "filter_options": filter_options,
            "total_count": len(df),
            "search_query": "",
            "sort_by": "combined_s",
            "sort_dir": "desc",
            "data_quality": "all",
            "selected_denoms": [],
            "selected_concepts": [],
            "selected_provinces": [],
        },
    )


@app.get("/api/schools", response_class=HTMLResponse)
async def filter_schools(
    request: Request,
    q: str = "",
    denom: Annotated[list[str], Query()] = [],
    concept: Annotated[list[str], Query()] = [],
    province: str = "",
    sort_by: str = "combined_s",
    sort_dir: str = "desc",
    quality: str = "all",
):
    """Filter and return school cards (HTMX endpoint)."""
    df = get_school_data()

    # Apply search filter (handle null values in columns)
    # Prioritize exact city matches, then partial matches
    if q:
        q_lower = q.lower().strip()
        q_upper = q.upper().strip()

        # First try exact city match (case-insensitive)
        exact_city_match = df.filter(pl.col("city").fill_null("").str.to_uppercase() == q_upper)

        # If we have exact city matches, use those
        if len(exact_city_match) > 0:
            df = exact_city_match
        else:
            # Otherwise, do partial matching across all fields
            df = df.filter(
                pl.col("name").fill_null("").str.to_lowercase().str.contains(q_lower)
                | pl.col("city").fill_null("").str.to_lowercase().str.contains(q_lower)
                | pl.col("postal_code").fill_null("").str.to_lowercase().str.contains(q_lower)
                | pl.col("municipality").fill_null("").str.to_lowercase().str.contains(q_lower)
                | pl.col("province").fill_null("").str.to_lowercase().str.contains(q_lower)
            )

    # Apply denomination filter
    if denom:
        df = df.filter(pl.col("denomination").is_in(denom))

    # Apply concept filter
    if concept:
        df = df.filter(pl.col("concept").is_in(concept))

    # Apply province filter
    if province:
        df = df.filter(pl.col("province") == province)

    # Apply quality filter
    if quality == "reliable":
        df = df.filter(pl.col("data_quality") == "reliable")
    elif quality == "limited":
        df = df.filter(pl.col("data_quality").is_in(["reliable", "limited"]))

    # Apply sorting
    sort_col_map = {
        "combined_s": "combined_s_pct",
        "combined_f": "combined_f_pct",
        "combined": "combined_score",
        "math_s": "math_s_pct",
        "math_f": "math_f_pct",
        "reading_s": "reading_s_pct",
        "reading_f": "reading_f_pct",
    }
    sort_col = sort_col_map.get(sort_by, "combined_s_pct")
    descending = sort_dir == "desc"

    df = df.sort(sort_col, descending=descending, nulls_last=True)

    # Limit results
    schools = df.head(50)

    return templates.TemplateResponse(
        "partials/school_cards.html",
        {
            "request": request,
            "schools": schools.to_dicts(),
        },
    )


@app.get("/api/schools/geojson")
async def schools_geojson(
    q: str = "",
    denom: Annotated[list[str], Query()] = [],
    concept: Annotated[list[str], Query()] = [],
    province: str = "",
    quality: str = "all",
):
    """Return filtered schools as GeoJSON for map markers."""
    df = get_school_data()

    # Apply search filter (same logic as /api/schools)
    if q:
        q_lower = q.lower().strip()
        q_upper = q.upper().strip()

        exact_city_match = df.filter(pl.col("city").fill_null("").str.to_uppercase() == q_upper)

        if len(exact_city_match) > 0:
            df = exact_city_match
        else:
            df = df.filter(
                pl.col("name").fill_null("").str.to_lowercase().str.contains(q_lower)
                | pl.col("city").fill_null("").str.to_lowercase().str.contains(q_lower)
                | pl.col("postal_code").fill_null("").str.to_lowercase().str.contains(q_lower)
                | pl.col("municipality").fill_null("").str.to_lowercase().str.contains(q_lower)
                | pl.col("province").fill_null("").str.to_lowercase().str.contains(q_lower)
            )

    # Apply denomination filter
    if denom:
        df = df.filter(pl.col("denomination").is_in(denom))

    # Apply concept filter
    if concept:
        df = df.filter(pl.col("concept").is_in(concept))

    # Apply province filter
    if province:
        df = df.filter(pl.col("province") == province)

    # Apply quality filter
    if quality == "reliable":
        df = df.filter(pl.col("data_quality") == "reliable")
    elif quality == "limited":
        df = df.filter(pl.col("data_quality").is_in(["reliable", "limited"]))

    # Convert to GeoJSON features
    features = []
    for row in df.iter_rows(named=True):
        lat = row.get("latitude")
        lon = row.get("longitude")
        if lat is not None and lon is not None:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {
                        "id": row.get("school_id"),
                        "name": row.get("name"),
                        "city": row.get("city"),
                        "street": row.get("street"),
                        "house_number": row.get("house_number"),
                        "postal_code": row.get("postal_code"),
                        "denomination": row.get("denomination"),
                        "concept": row.get("concept"),
                        "combined_f_pct": row.get("combined_f_pct"),
                        "combined_s_pct": row.get("combined_s_pct"),
                        "student_count": row.get("student_count"),
                    },
                }
            )

    return {"type": "FeatureCollection", "features": features}


@app.get("/school/{school_id}", response_class=HTMLResponse)
async def school_detail(request: Request, school_id: str):
    """School detail page."""
    df = get_school_data()

    school_df = df.filter(pl.col("school_id") == school_id)

    if len(school_df) == 0:
        return templates.TemplateResponse(
            "base.html",
            {
                "request": request,
                "error": "School niet gevonden",
            },
            status_code=404,
        )

    school = school_df.to_dicts()[0]

    return templates.TemplateResponse(
        "school_detail.html",
        {
            "request": request,
            "school": school,
        },
    )


# =============================================================================
# Secondary School (Voortgezet Onderwijs) Routes
# =============================================================================


@app.get("/middelbare-scholen", response_class=HTMLResponse)
async def vo_index(request: Request):
    """Secondary school search page."""
    df = get_vo_school_data()
    filter_options = get_vo_filter_options(df)

    # Get first 50 schools sorted by pass rate
    schools = df.sort("pass_rate_3year", descending=True, nulls_last=True).head(50)

    return templates.TemplateResponse(
        "vo_index.html",
        {
            "request": request,
            "schools": schools.to_dicts(),
            "filter_options": filter_options,
            "total_count": len(df),
            "search_query": "",
            "sort_by": "pass_rate",
            "sort_dir": "desc",
            "data_quality": "all",
            "min_grade": "",
            "selected_denoms": [],
            "selected_types": [],
            "selected_provinces": [],
        },
    )


@app.get("/api/vo/schools", response_class=HTMLResponse)
async def filter_vo_schools(
    request: Request,
    q: str = "",
    denom: Annotated[list[str], Query()] = [],
    school_type: Annotated[list[str], Query()] = [],
    province: str = "",
    sort_by: str = "pass_rate",
    sort_dir: str = "desc",
    quality: str = "all",
    min_grade: str = "",
):
    """Filter and return secondary school cards (HTMX endpoint)."""
    df = get_vo_school_data()

    # Apply search filter
    if q:
        q_lower = q.lower().strip()
        q_upper = q.upper().strip()

        exact_city_match = df.filter(pl.col("city").fill_null("").str.to_uppercase() == q_upper)

        if len(exact_city_match) > 0:
            df = exact_city_match
        else:
            df = df.filter(
                pl.col("name").fill_null("").str.to_lowercase().str.contains(q_lower)
                | pl.col("city").fill_null("").str.to_lowercase().str.contains(q_lower)
                | pl.col("postal_code").fill_null("").str.to_lowercase().str.contains(q_lower)
                | pl.col("municipality").fill_null("").str.to_lowercase().str.contains(q_lower)
                | pl.col("province").fill_null("").str.to_lowercase().str.contains(q_lower)
            )

    # Apply denomination filter
    if denom:
        df = df.filter(pl.col("denomination").is_in(denom))

    # Apply school type filter (VMBO, HAVO, VWO)
    if school_type:
        df = df.filter(pl.col("school_type").is_in(school_type))

    # Apply province filter
    if province:
        df = df.filter(pl.col("province") == province)

    # Apply quality filter
    if quality == "reliable":
        df = df.filter(pl.col("data_quality") == "reliable")
    elif quality == "limited":
        df = df.filter(pl.col("data_quality").is_in(["reliable", "limited"]))

    # Apply minimum grade filter
    if min_grade:
        try:
            min_grade_val = float(min_grade)
            if "avg_cijferlijst" in df.columns:
                df = df.filter(pl.col("avg_cijferlijst") >= min_grade_val)
        except ValueError:
            pass

    # Apply sorting
    sort_col_map = {
        "pass_rate": "pass_rate_3year",
        "pass_rate_current": "pass_rate_current",
        "avg_grade": "avg_cijferlijst",
        "name": "name",
    }
    sort_col = sort_col_map.get(sort_by, "pass_rate_3year")
    descending = sort_dir == "desc"

    df = df.sort(sort_col, descending=descending, nulls_last=True)

    # Limit results
    schools = df.head(50)

    return templates.TemplateResponse(
        "partials/vo_school_cards.html",
        {
            "request": request,
            "schools": schools.to_dicts(),
        },
    )


@app.get("/api/vo/schools/geojson")
async def vo_schools_geojson(
    q: str = "",
    denom: Annotated[list[str], Query()] = [],
    school_type: Annotated[list[str], Query()] = [],
    province: str = "",
    quality: str = "all",
):
    """Return filtered secondary schools as GeoJSON for map markers."""
    df = get_vo_school_data()

    # Apply search filter
    if q:
        q_lower = q.lower().strip()
        q_upper = q.upper().strip()

        exact_city_match = df.filter(pl.col("city").fill_null("").str.to_uppercase() == q_upper)

        if len(exact_city_match) > 0:
            df = exact_city_match
        else:
            df = df.filter(
                pl.col("name").fill_null("").str.to_lowercase().str.contains(q_lower)
                | pl.col("city").fill_null("").str.to_lowercase().str.contains(q_lower)
                | pl.col("postal_code").fill_null("").str.to_lowercase().str.contains(q_lower)
                | pl.col("municipality").fill_null("").str.to_lowercase().str.contains(q_lower)
                | pl.col("province").fill_null("").str.to_lowercase().str.contains(q_lower)
            )

    # Apply denomination filter
    if denom:
        df = df.filter(pl.col("denomination").is_in(denom))

    # Apply school type filter
    if school_type:
        df = df.filter(pl.col("school_type").is_in(school_type))

    # Apply province filter
    if province:
        df = df.filter(pl.col("province") == province)

    # Apply quality filter
    if quality == "reliable":
        df = df.filter(pl.col("data_quality") == "reliable")
    elif quality == "limited":
        df = df.filter(pl.col("data_quality").is_in(["reliable", "limited"]))

    # Convert to GeoJSON features
    features = []
    for row in df.iter_rows(named=True):
        lat = row.get("latitude")
        lon = row.get("longitude")
        if lat is not None and lon is not None:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {
                        "id": row.get("school_type_id"),
                        "name": row.get("name"),
                        "city": row.get("city"),
                        "street": row.get("street"),
                        "house_number": row.get("house_number"),
                        "postal_code": row.get("postal_code"),
                        "denomination": row.get("denomination"),
                        "school_type": row.get("school_type"),
                        "pass_rate_3year": row.get("pass_rate_3year"),
                        "pass_rate_current": row.get("pass_rate_current"),
                    },
                }
            )

    return {"type": "FeatureCollection", "features": features}


@app.get("/middelbare-school/{school_type_id}", response_class=HTMLResponse)
async def vo_school_detail(request: Request, school_type_id: str):
    """Secondary school detail page."""
    df = get_vo_school_data()

    school_df = df.filter(pl.col("school_type_id") == school_type_id)

    if len(school_df) == 0:
        return templates.TemplateResponse(
            "base.html",
            {
                "request": request,
                "error": "School niet gevonden",
            },
            status_code=404,
        )

    school = school_df.to_dicts()[0]

    # Get profile/sector breakdown for all school types (VMBO, HAVO, VWO)
    profile_breakdown = []
    if school.get("school_type") in ["VMBO", "HAVO", "VWO"]:
        profile_breakdown = get_vo_profile_breakdown(
            school.get("school_id"), school.get("school_type")
        )

    # Get all available school types for this school (for tabs)
    available_types = get_vo_available_types(df, school.get("school_id"))

    # Get historical data for trend section
    historical_data = get_vo_historical_data(school.get("school_id"), school.get("school_type"))

    # Get doorstroom data (opstroom, afstroom, zittenblijvers)
    doorstroom_data = get_vo_doorstroom_for_school(school.get("school_id"))

    return templates.TemplateResponse(
        "vo_school_detail.html",
        {
            "request": request,
            "school": school,
            "profile_breakdown": profile_breakdown,
            "available_types": available_types,
            "historical_data": historical_data,
            "doorstroom_data": doorstroom_data,
        },
    )


# =============================================================================
# Admin/Utility Routes
# =============================================================================


@app.get("/api/refresh")
async def refresh_data():
    """Force refresh of school data (re-downloads and re-geocodes)."""
    global _school_data
    try:
        _school_data = load_combined_data(force_refresh=True)
        has_coords = _school_data.filter(pl.col("latitude").is_not_null()).height
        return {
            "status": "ok",
            "count": len(_school_data),
            "with_coordinates": has_coords,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/reload")
async def reload_data():
    """Reload cached data from parquet files (no re-download)."""
    global _school_data
    try:
        from pathlib import Path

        import polars as pl

        cache_path = Path(__file__).parent.parent / "data" / "combined.parquet"
        if cache_path.exists():
            _school_data = pl.read_parquet(cache_path)
            has_coords = _school_data.filter(pl.col("latitude").is_not_null()).height
            return {
                "status": "ok",
                "count": len(_school_data),
                "with_coordinates": has_coords,
            }
        else:
            return {"status": "error", "message": "No cached data found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
