import argparse
import csv
import os
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests


GRAPH = "https://graph.mapillary.com"
POPULAR_COUNTRY_BBOXES: Dict[str, Tuple[float, float, float, float]] = {
    "ar": (-73.7, -55.1, -53.6, -21.8),   # Argentina
    "au": (112.9, -43.8, 153.7, -10.7),   # Australia
    "at": (9.5, 46.3, 17.2, 49.1),        # Austria
    "be": (2.5, 49.5, 6.4, 51.6),         # Belgium
    "br": (-73.9, -33.8, -34.7, 5.3),     # Brazil
    "ca": (-141.0, 41.7, -52.6, 83.1),    # Canada
    "hr": (13.5, 42.3, 19.5, 46.6),       # Croatia
    "cz": (12.0, 48.5, 18.9, 51.1),       # Czechia
    "dk": (8.1, 54.5, 15.3, 57.8),        # Denmark
    "fi": (20.6, 59.8, 31.6, 70.1),       # Finland
    "fr": (-5.2, 41.2, 9.8, 51.3),        # France (metro)
    "de": (5.9, 47.2, 15.1, 55.1),        # Germany
    "gr": (19.2, 34.8, 28.3, 41.8),       # Greece
    "hu": (16.1, 45.7, 22.9, 48.7),       # Hungary
    "is": (-24.7, 63.2, -13.0, 66.6),     # Iceland
    "id": (95.0, -10.9, 141.1, 5.9),      # Indonesia
    "ie": (-10.7, 51.2, -5.3, 55.5),      # Ireland
    "it": (6.6, 36.5, 18.6, 47.1),        # Italy
    "jp": (129.4, 31.0, 145.8, 45.7),     # Japan
    "mx": (-117.2, 14.3, -86.7, 32.7),    # Mexico
    "nl": (3.2, 50.7, 7.3, 53.7),         # Netherlands
    "nz": (166.4, -47.6, 178.6, -34.0),   # New Zealand
    "no": (4.5, 57.9, 31.4, 71.3),        # Norway
    "pl": (14.1, 49.0, 24.2, 54.9),       # Poland
    "pt": (-9.6, 36.8, -6.0, 42.2),       # Portugal
    "ro": (20.2, 43.5, 29.7, 48.3),       # Romania
    "si": (13.3, 45.4, 16.7, 46.9),       # Slovenia
    "es": (-9.4, 36.0, 3.4, 43.9),        # Spain
    "se": (11.0, 55.3, 24.2, 69.1),       # Sweden
    "ch": (5.9, 45.8, 10.6, 47.9),        # Switzerland
    "th": (97.3, 5.6, 105.8, 20.5),       # Thailand
    "tr": (26.0, 36.0, 45.0, 42.2),       # Turkey
    "gb": (-8.7, 49.8, 1.9, 58.7),        # United Kingdom
    "us": (-124.8, 24.3, -66.9, 49.4),    # United States (contiguous)
    # Additional global coverage.
    "in": (68.1, 6.5, 97.4, 35.7),        # India
    "cn": (73.5, 18.1, 134.8, 53.6),      # China
    "kr": (126.1, 33.1, 129.6, 38.7),     # South Korea
    "tw": (119.3, 21.8, 122.1, 25.4),     # Taiwan
    "my": (99.6, 0.9, 119.3, 7.4),        # Malaysia
    "sg": (103.6, 1.2, 104.1, 1.5),       # Singapore
    "vn": (102.1, 8.2, 109.5, 23.4),      # Vietnam
    "ph": (116.9, 4.6, 126.6, 19.6),      # Philippines
    "za": (16.4, -34.9, 32.9, -22.1),     # South Africa
    "ke": (33.9, -4.8, 41.9, 5.1),        # Kenya
    "eg": (24.7, 22.0, 36.9, 31.7),       # Egypt
    "ma": (-13.2, 27.6, -1.0, 35.9),      # Morocco
    "tn": (7.4, 30.2, 11.7, 37.6),        # Tunisia
    "gh": (-3.3, 4.5, 1.4, 11.2),         # Ghana
    "ng": (2.7, 4.3, 14.7, 13.9),         # Nigeria
    "cl": (-75.7, -55.9, -66.4, -17.5),   # Chile
    "co": (-79.1, -4.3, -66.8, 13.5),     # Colombia
    "pe": (-81.4, -18.4, -68.7, -0.0),    # Peru
    "uy": (-58.5, -35.2, -53.1, -30.0),   # Uruguay
    "ec": (-81.1, -5.1, -75.2, 1.7),      # Ecuador
}
PRESET_COUNTRY_CODES: Dict[str, List[str]] = {
    "popular": sorted(
        [
            "ar",
            "at",
            "au",
            "be",
            "br",
            "ca",
            "ch",
            "cz",
            "de",
            "dk",
            "es",
            "fi",
            "fr",
            "gb",
            "gr",
            "hr",
            "hu",
            "id",
            "ie",
            "is",
            "it",
            "jp",
            "mx",
            "nl",
            "no",
            "nz",
            "pl",
            "pt",
            "ro",
            "se",
            "si",
            "th",
            "tr",
            "us",
        ]
    ),
    "global_balanced": sorted(
        [
            # North America
            "us",
            "ca",
            "mx",
            # South America
            "ar",
            "br",
            "cl",
            "co",
            "ec",
            "pe",
            "uy",
            # Europe
            "at",
            "be",
            "ch",
            "cz",
            "de",
            "dk",
            "es",
            "fi",
            "fr",
            "gb",
            "gr",
            "hr",
            "hu",
            "ie",
            "it",
            "nl",
            "no",
            "pl",
            "pt",
            "ro",
            "se",
            "si",
            # Asia
            "cn",
            "id",
            "in",
            "jp",
            "kr",
            "my",
            "ph",
            "sg",
            "th",
            "tr",
            "tw",
            "vn",
            # Africa
            "eg",
            "gh",
            "ke",
            "ma",
            "ng",
            "tn",
            "za",
            # Oceania
            "au",
            "nz",
        ]
    ),
}
EXPECTED_METADATA_FIELDS = [
    "image_id",
    "thumb_url",
    "path",
    "lat",
    "lon",
    "captured_at",
    "compass_angle",
    "sequence",
    "source",
]
KNOWN_METADATA_FIELDS = {
    "image_id",
    "thumb_url",
    "path",
    "lat",
    "lon",
    "captured_at",
    "compass_angle",
    "sequence",
    "sequence_id",
    "source",
}
_thread_local = threading.local()


def find_env_path() -> Optional[Path]:
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        return cwd_env
    repo_env = Path(__file__).resolve().parents[1] / ".env"
    if repo_env.exists():
        return repo_env
    script_env = Path(__file__).resolve().parent / ".env"
    if script_env.exists():
        return script_env
    return None


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key:
                os.environ[key] = value


def read_seen_ids(meta_csv: Path) -> Set[str]:
    if not meta_csv.exists():
        return set()
    seen = set()
    with open(meta_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            # Handle legacy headers with accidental leading/trailing spaces.
            image_id = row.get("image_id")
            if not image_id:
                image_id = row.get(" image_id")
            if image_id:
                seen.add(image_id.strip())
    return seen


def read_sequence_counts(meta_csv: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not meta_csv.exists():
        return counts
    with open(meta_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sequence = (row.get("sequence") or row.get("sequence_id") or "").strip()
            if not sequence:
                continue
            counts[sequence] = counts.get(sequence, 0) + 1
    return counts


def normalize_col_name(name: str) -> str:
    return name.strip().lower()


def ensure_csv_header(meta_csv: Path, fieldnames: List[str]) -> None:
    if meta_csv.exists():
        return
    meta_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()


def read_csv_header(meta_csv: Path) -> List[str]:
    with open(meta_csv, "r", encoding="utf-8", newline="") as f:
        row = next(csv.reader(f), [])
    return [c for c in row if c is not None]


def can_migrate_header(header: List[str]) -> bool:
    normalized = [normalize_col_name(h) for h in header if h.strip()]
    if not normalized:
        return True
    if not set(normalized).issubset(KNOWN_METADATA_FIELDS):
        return False
    required = {"image_id", "thumb_url", "path", "lat", "lon", "captured_at"}
    return required.issubset(set(normalized))


def migrate_metadata_schema(meta_csv: Path, fieldnames: List[str]) -> None:
    backup_path = meta_csv.with_name(f"{meta_csv.name}.bak.{int(time.time())}")
    tmp_path = meta_csv.with_name(f"{meta_csv.name}.tmp")

    shutil.copy2(meta_csv, backup_path)

    migrated_rows = 0
    with open(meta_csv, "r", encoding="utf-8", newline="") as in_f, open(
        tmp_path, "w", encoding="utf-8", newline=""
    ) as out_f:
        reader = csv.DictReader(in_f)
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            normalized_row = {
                normalize_col_name(k): (v or "")
                for k, v in row.items()
                if isinstance(k, str)
            }
            writer.writerow(
                {
                    "image_id": normalized_row.get("image_id", ""),
                    "thumb_url": normalized_row.get("thumb_url", ""),
                    "path": normalized_row.get("path", ""),
                    "lat": normalized_row.get("lat", ""),
                    "lon": normalized_row.get("lon", ""),
                    "captured_at": normalized_row.get("captured_at", ""),
                    "compass_angle": normalized_row.get("compass_angle", ""),
                    "sequence": normalized_row.get("sequence", normalized_row.get("sequence_id", "")),
                    "source": normalized_row.get("source", "mapillary"),
                }
            )
            migrated_rows += 1

    tmp_path.replace(meta_csv)
    print(
        f"Migrated metadata schema for {meta_csv} "
        f"({migrated_rows} rows). Backup: {backup_path}"
    )


def ensure_metadata_schema(meta_csv: Path, fieldnames: List[str]) -> None:
    ensure_csv_header(meta_csv, fieldnames)
    header = read_csv_header(meta_csv)
    if header == fieldnames:
        return
    if can_migrate_header(header):
        migrate_metadata_schema(meta_csv, fieldnames)
        return
    raise RuntimeError(
        "Metadata schema mismatch. "
        f"Expected header: {fieldnames}. Found: {header}. "
        "Refusing to append to avoid corrupting metadata."
    )


def append_error_row(error_csv: Path, row: Dict[str, str], fieldnames: List[str]) -> None:
    error_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not error_csv.exists()
    with open(error_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def request_with_retries(
    session: requests.Session,
    url: str,
    params: Dict[str, str],
    timeout: float,
    max_retries: int,
    backoff: float,
    stream: bool = False,
) -> requests.Response:
    retry_status = {429, 500, 502, 503, 504}
    attempt = 0
    while True:
        try:
            response = session.get(url, params=params, timeout=timeout, stream=stream)
        except requests.RequestException as e:
            if attempt >= max_retries:
                raise RuntimeError(f"request failed after {attempt + 1} attempts: {e}") from e
            sleep_s = backoff * (2**attempt)
            time.sleep(sleep_s)
            attempt += 1
            continue

        if response.status_code in retry_status:
            snippet = response.text[:240]
            response.close()
            if attempt >= max_retries:
                raise RuntimeError(
                    f"request failed with status {response.status_code} "
                    f"after {attempt + 1} attempts: {snippet}"
                )
            sleep_s = backoff * (2**attempt)
            time.sleep(sleep_s)
            attempt += 1
            continue

        return response


def build_session(pool_size: int) -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
        max_retries=0,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def thread_session(pool_size: int) -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = build_session(pool_size=pool_size)
        _thread_local.session = session
    return session


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def rand_point_in_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> Tuple[float, float]:
    lon = random.uniform(min_lon, max_lon)
    lat = random.uniform(min_lat, max_lat)
    return lon, lat


def small_bbox_around(lon: float, lat: float, half_size_deg: float) -> Tuple[float, float, float, float]:
    # Mapillary bbox is: min_lon,min_lat,max_lon,max_lat :contentReference[oaicite:3]{index=3}
    return (
        lon - half_size_deg,
        lat - half_size_deg,
        lon + half_size_deg,
        lat + half_size_deg,
    )


def mapillary_image_search(
    session: requests.Session,
    token: str,
    bbox: Tuple[float, float, float, float],
    thumb_field: str,
    timeout: float,
    max_retries: int,
    backoff: float,
    limit: int = 200,
) -> List[Dict[str, object]]:
    params = {
        "fields": ",".join(
            [
                "id",
                thumb_field,
                "captured_at",
                "geometry",
                "compass_angle",
                "sequence",
            ]
        ),
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "limit": str(limit),
        "access_token": token,
    }
    url = f"{GRAPH}/images"
    r = request_with_retries(
        session,
        url,
        params=params,
        timeout=timeout,
        max_retries=max_retries,
        backoff=backoff,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code} {r.text[:400]}")
    data = r.json()
    out: List[Dict[str, object]] = []
    for item in data.get("data", []):
        image_id = str(item.get("id", "")).strip()
        thumb_url = item.get(thumb_field)
        geom = item.get("geometry") or {}
        coords = geom.get("coordinates")
        if not image_id or not thumb_url or not coords or len(coords) != 2:
            continue
        out.append(
            {
                "image_id": image_id,
                "thumb_url": str(thumb_url),
                "lon": float(coords[0]),
                "lat": float(coords[1]),
                "captured_at": item.get("captured_at", ""),
                "compass_angle": item.get("compass_angle", ""),
                "sequence": item.get("sequence", ""),
            }
        )
    return out


def download_file(
    session: requests.Session,
    url: str,
    out_path: Path,
    timeout: float,
    max_retries: int,
    backoff: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with request_with_retries(
        session,
        url,
        params={},
        timeout=timeout,
        max_retries=max_retries,
        backoff=backoff,
        stream=True,
    ) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def download_file_parallel(
    url: str,
    out_path: Path,
    timeout: float,
    max_retries: int,
    backoff: float,
    pool_size: int,
) -> None:
    session = thread_session(pool_size=pool_size)
    download_file(
        session,
        url,
        out_path,
        timeout=timeout,
        max_retries=max_retries,
        backoff=backoff,
    )


def guess_ext_from_url(url: str) -> str:
    base = url.split("?")[0]
    ext = os.path.splitext(base)[1].lower()
    if ext in [".jpg", ".jpeg", ".png", ".webp"]:
        return ext
    return ".jpg"


def parse_bbox(s: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'min_lon,min_lat,max_lon,max_lat'")
    min_lon, min_lat, max_lon, max_lat = map(float, parts)
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("bbox invalid: min must be < max")
    return min_lon, min_lat, max_lon, max_lat


def parse_country_codes(raw: str) -> List[str]:
    codes = [c.strip().lower() for c in raw.split(",") if c.strip()]
    if not codes:
        raise ValueError("countries list is empty")
    unknown = [c for c in codes if c not in POPULAR_COUNTRY_BBOXES]
    if unknown:
        known = ",".join(sorted(POPULAR_COUNTRY_BBOXES))
        raise ValueError(
            f"unknown country codes: {','.join(unknown)}. "
            f"Known codes: {known}"
        )
    return codes


def choose_regions(
    bbox_raw: Optional[str],
    countries_raw: Optional[str],
    preset: Optional[str],
) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    selectors = int(bool(bbox_raw)) + int(bool(countries_raw)) + int(bool(preset))
    if selectors > 1:
        raise ValueError("Use only one of --bbox, --countries, or --country-preset")
    if bbox_raw:
        return [("bbox", parse_bbox(bbox_raw))]
    if countries_raw:
        codes = parse_country_codes(countries_raw)
        return [(code, POPULAR_COUNTRY_BBOXES[code]) for code in codes]
    if preset and preset in PRESET_COUNTRY_CODES:
        codes = PRESET_COUNTRY_CODES[preset]
        return [(code, POPULAR_COUNTRY_BBOXES[code]) for code in codes]
    raise ValueError(
        "Provide one of: --bbox, --countries, or --country-preset "
        f"({'|'.join(sorted(PRESET_COUNTRY_CODES))})"
    )


def resolve_path(raw: str, base_dir: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download random Mapillary street-level thumbnails via bbox or multi-country sampling."
    )
    ap.add_argument(
        "--bbox",
        default=None,
        help="min_lon,min_lat,max_lon,max_lat (e.g. '13.5,42.0,19.5,46.6')",
    )
    ap.add_argument(
        "--countries",
        default=None,
        help="Comma-separated country codes (e.g. 'us,ca,gb,jp'). Use --list-countries to see options.",
    )
    ap.add_argument(
        "--country-preset",
        choices=sorted(PRESET_COUNTRY_CODES.keys()),
        default=None,
        help="Named country preset. 'global_balanced' adds more Africa/Asia/South-America coverage.",
    )
    ap.add_argument(
        "--list-countries",
        action="store_true",
        help="Print known country codes and exit",
    )
    ap.add_argument("--out-dir", default="data/raw/mapillary/images")
    ap.add_argument("--meta-csv", default="data/raw/mapillary/metadata.csv")
    ap.add_argument("--target", type=int, default=5000, help="How many images to download total")
    ap.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="How many random points to sample (more = broader coverage)",
    )
    ap.add_argument(
        "--half-size-deg",
        type=float,
        default=0.01,
        help="Half-size of the small bbox around each random point (~0.01 deg ~ 1km-ish lat)",
    )
    ap.add_argument(
        "--thumb",
        choices=["1024", "2048"],
        default="1024",
        help="Thumbnail size to download",
    )
    ap.add_argument(
        "--search-limit",
        type=int,
        default=200,
        help="Max images per bbox search (lower = faster/less repetitive)",
    )
    ap.add_argument(
        "--per-search-max-new",
        type=int,
        default=60,
        help="Cap successful new downloads attempted from each search result (0 disables)",
    )
    ap.add_argument(
        "--download-workers",
        type=int,
        default=16,
        help="Parallel image download workers",
    )
    ap.add_argument(
        "--max-per-sequence",
        type=int,
        default=2,
        help="Cap images per sequence (0 disables)",
    )
    ap.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="HTTP timeout for metadata calls (seconds)",
    )
    ap.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Retries for transient HTTP/network failures",
    )
    ap.add_argument(
        "--retry-backoff",
        type=float,
        default=0.5,
        help="Base exponential backoff (seconds)",
    )
    ap.add_argument(
        "--error-csv",
        default="data/raw/mapillary/download_errors.csv",
        help="Where request/download failures are logged",
    )
    ap.add_argument(
        "--flush-every",
        type=int,
        default=100,
        help="Flush metadata CSV every N successful rows",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for bbox sampling/shuffle")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep between API calls (seconds)")
    args = ap.parse_args()

    if args.half_size_deg <= 0:
        ap.error("--half-size-deg must be > 0")
    if args.half_size_deg > 0.05:
        ap.error("--half-size-deg too large for Mapillary search area limit; use <= 0.05")
    if args.download_workers < 1:
        ap.error("--download-workers must be >= 1")
    if args.per_search_max_new < 0:
        ap.error("--per-search-max-new must be >= 0")
    if args.flush_every < 1:
        ap.error("--flush-every must be >= 1")

    random.seed(args.seed)

    if args.list_countries:
        print("Available presets:")
        for preset_name in sorted(PRESET_COUNTRY_CODES):
            codes = PRESET_COUNTRY_CODES[preset_name]
            print(f"- {preset_name}: {len(codes)} countries")
            print(f"  {','.join(codes)}")
        print("")
        print("Available country codes and bbox:")
        for code in sorted(POPULAR_COUNTRY_BBOXES):
            bbox = POPULAR_COUNTRY_BBOXES[code]
            print(f"{code}: {bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")
        return

    env_path = find_env_path()
    if env_path:
        load_env_file(env_path)

    token = os.getenv("MAPILLARY_TOKEN") or os.getenv("MAPILLARY_ACCESS_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing MAPILLARY_TOKEN env var. Put it in .env or export it."
        )
    print("Loaded .env:", env_path)
    print("Token starts with:", token[:4], "len:", len(token))
    if not token.startswith("MLY|") or len(token) < 40:
        raise RuntimeError(f"MAPILLARY_TOKEN looks wrong (len={len(token)})")

    try:
        regions = choose_regions(args.bbox, args.countries, args.country_preset)
    except ValueError as e:
        ap.error(str(e))
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = resolve_path(args.out_dir, repo_root)
    meta_csv = resolve_path(args.meta_csv, repo_root)
    error_csv = resolve_path(args.error_csv, repo_root)

    fieldnames = EXPECTED_METADATA_FIELDS
    ensure_metadata_schema(meta_csv, fieldnames)
    seen = read_seen_ids(meta_csv)
    seq_counts = read_sequence_counts(meta_csv)

    thumb_field = f"thumb_{args.thumb}_url"

    downloaded = len(seen)

    print(f"Existing metadata rows: {len(seen)}")
    print(f"Existing sequences tracked: {len(seq_counts)}")
    print(f"Existing rows counted toward target: {downloaded}")
    print(f"Output image directory: {out_dir}")
    print(f"Sampling regions: {len(regions)}")
    print(f"Parallel download workers: {args.download_workers}")
    print(f"Per-search max new downloads: {args.per_search_max_new}")
    if len(regions) <= 10:
        print("Region codes:", ",".join([name for name, _ in regions]))
    else:
        preview = ",".join([name for name, _ in regions[:10]])
        print(f"Region codes (first 10): {preview} ...")
    print(f"Target total files: {args.target}")

    # open metadata for append
    error_fields = ["ts", "stage", "sample_i", "image_id", "status", "message"]
    session_pool_size = max(16, args.download_workers * 2)
    session = build_session(pool_size=session_pool_size)
    searches = 0
    skipped_seen = 0
    skipped_sequence_cap = 0
    search_by_region: Dict[str, int] = {}
    downloaded_by_region: Dict[str, int] = {}
    rows_since_flush = 0
    with open(meta_csv, "a", encoding="utf-8", newline="") as f, ThreadPoolExecutor(
        max_workers=args.download_workers
    ) as pool:
        w = csv.DictWriter(f, fieldnames=fieldnames)

        for i in range(args.samples):
            if downloaded >= args.target:
                break

            region_name, region_bbox = random.choice(regions)
            lon, lat = rand_point_in_bbox(*region_bbox)
            small = small_bbox_around(lon, lat, args.half_size_deg)

            # clamp small bbox to region bbox (so we don't drift outside)
            min_lon = clamp(small[0], region_bbox[0], region_bbox[2])
            min_lat = clamp(small[1], region_bbox[1], region_bbox[3])
            max_lon = clamp(small[2], region_bbox[0], region_bbox[2])
            max_lat = clamp(small[3], region_bbox[1], region_bbox[3])
            if min_lon >= max_lon or min_lat >= max_lat:
                continue
            small = (min_lon, min_lat, max_lon, max_lat)

            try:
                items = mapillary_image_search(
                    session,
                    token,
                    small,
                    thumb_field=thumb_field,
                    timeout=args.request_timeout,
                    max_retries=args.max_retries,
                    backoff=args.retry_backoff,
                    limit=args.search_limit,
                )
                searches += 1
                search_by_region[region_name] = search_by_region.get(region_name, 0) + 1
            except Exception as e:
                print(f"[{i+1}/{args.samples}] search error ({region_name}): {e}")
                append_error_row(
                    error_csv,
                    {
                        "ts": str(int(time.time())),
                        "stage": "search",
                        "sample_i": str(i + 1),
                        "image_id": "",
                        "status": "error",
                        "message": str(e)[:400],
                    },
                    error_fields,
                )
                time.sleep(args.sleep)
                continue

            time.sleep(args.sleep)

            # shuffle so we don't always pull the same order
            random.shuffle(items)

            pending_seq_counts: Dict[str, int] = {}
            candidates: List[Dict[str, object]] = []
            for info in items:
                if downloaded + len(candidates) >= args.target:
                    break
                if args.per_search_max_new > 0 and len(candidates) >= args.per_search_max_new:
                    break
                image_id = str(info["image_id"])
                if image_id in seen:
                    skipped_seen += 1
                    continue

                sequence = str(info.get("sequence", "") or "")
                if args.max_per_sequence > 0 and sequence:
                    existing = seq_counts.get(sequence, 0)
                    pending = pending_seq_counts.get(sequence, 0)
                    if existing + pending >= args.max_per_sequence:
                        skipped_sequence_cap += 1
                        continue
                    pending_seq_counts[sequence] = pending + 1

                seen.add(image_id)
                candidates.append(info)

            if not candidates:
                continue

            futures = {}
            for info in candidates:
                ext = guess_ext_from_url(str(info["thumb_url"]))
                local_path = out_dir / f"{info['image_id']}{ext}"
                future = pool.submit(
                    download_file_parallel,
                    url=str(info["thumb_url"]),
                    out_path=local_path,
                    timeout=max(args.request_timeout, 60.0),
                    max_retries=args.max_retries,
                    backoff=args.retry_backoff,
                    pool_size=session_pool_size,
                )
                futures[future] = (info, local_path, region_name)

            for future in as_completed(futures):
                info, local_path, item_region = futures[future]
                image_id = str(info["image_id"])
                sequence = str(info.get("sequence", "") or "")
                try:
                    future.result()
                except Exception as e:
                    seen.discard(image_id)
                    print(f"download error for {image_id}: {e}")
                    append_error_row(
                        error_csv,
                        {
                            "ts": str(int(time.time())),
                            "stage": "download",
                            "sample_i": str(i + 1),
                            "image_id": image_id,
                            "status": "error",
                            "message": str(e)[:400],
                        },
                        error_fields,
                    )
                    continue

                rel_path = local_path.relative_to(repo_root)
                w.writerow(
                    {
                        "image_id": info["image_id"],
                        "thumb_url": info["thumb_url"],
                        "path": str(rel_path).replace("\\", "/"),
                        "lat": info["lat"],
                        "lon": info["lon"],
                        "captured_at": info["captured_at"],
                        "compass_angle": info["compass_angle"],
                        "sequence": info["sequence"],
                        "source": "mapillary",
                    }
                )
                rows_since_flush += 1
                if rows_since_flush >= args.flush_every:
                    f.flush()
                    rows_since_flush = 0
                if args.max_per_sequence > 0 and sequence:
                    seq_counts[sequence] = seq_counts.get(sequence, 0) + 1
                seen.add(info["image_id"])
                downloaded += 1
                downloaded_by_region[item_region] = downloaded_by_region.get(item_region, 0) + 1

                if downloaded % 50 == 0:
                    print(f"Downloaded: {downloaded}/{args.target}")

        if rows_since_flush > 0:
            f.flush()

    print(f"Search calls: {searches}")
    print(f"Skipped because already seen: {skipped_seen}")
    print(f"Skipped because sequence cap reached: {skipped_sequence_cap}")
    if downloaded_by_region:
        parts = [
            f"{name}:{count}"
            for name, count in sorted(
                downloaded_by_region.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ]
        print("Downloaded by region:", " | ".join(parts))
    print(f"DONE. Total downloaded files: {downloaded}")


if __name__ == "__main__":
    main()
