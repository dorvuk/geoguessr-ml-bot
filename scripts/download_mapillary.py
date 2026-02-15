import argparse
import csv
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests


GRAPH = "https://graph.mapillary.com"
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
    timeout: float,
    max_retries: int,
    backoff: float,
    limit: int = 200,
) -> List[str]:
    params = {
        "fields": "id",
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
    return [str(item["id"]) for item in data.get("data", []) if "id" in item]


def mapillary_image_detail(
    session: requests.Session,
    token: str,
    image_id: str,
    thumb_field: str,
    timeout: float,
    max_retries: int,
    backoff: float,
) -> Optional[Dict]:
    fields = [
        "id",
        thumb_field,
        "captured_at",
        "geometry",
        "compass_angle",
        "sequence",
    ]
    params = {
        "fields": ",".join(fields),
        "access_token": token,
    }
    url = f"{GRAPH}/{image_id}"
    r = request_with_retries(
        session,
        url,
        params=params,
        timeout=timeout,
        max_retries=max_retries,
        backoff=backoff,
    )
    if r.status_code == 404:
        return None
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code} {r.text[:400]}")
    j = r.json()

    thumb_url = j.get(thumb_field)
    geom = j.get("geometry") or {}
    coords = geom.get("coordinates")  # [lon, lat]
    if not thumb_url or not coords or len(coords) != 2:
        return None

    lon, lat = float(coords[0]), float(coords[1])
    return {
        "image_id": str(j.get("id", image_id)),
        "thumb_url": thumb_url,
        "lon": lon,
        "lat": lat,
        "captured_at": j.get("captured_at", ""),
        "compass_angle": j.get("compass_angle", ""),
        "sequence": j.get("sequence", ""),
    }


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


def resolve_path(raw: str, base_dir: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download random Mapillary street-level thumbnails via bbox sampling."
    )
    ap.add_argument(
        "--bbox",
        required=True,
        help="min_lon,min_lat,max_lon,max_lat (e.g. '13.5,42.0,19.5,46.6')",
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
        default=0.002,
        help="Half-size of the small bbox around each random point (~0.002 deg ~ 200m-ish lat)",
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
    ap.add_argument("--seed", type=int, default=42, help="Random seed for bbox sampling/shuffle")
    ap.add_argument("--sleep", type=float, default=0.05, help="Sleep between API calls (seconds)")
    args = ap.parse_args()

    random.seed(args.seed)

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

    big_bbox = parse_bbox(args.bbox)
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
    print(f"Target total files: {args.target}")

    # open metadata for append
    error_fields = ["ts", "stage", "sample_i", "image_id", "status", "message"]
    session = requests.Session()
    with open(meta_csv, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)

        for i in range(args.samples):
            if downloaded >= args.target:
                break

            lon, lat = rand_point_in_bbox(*big_bbox)
            small = small_bbox_around(lon, lat, args.half_size_deg)

            # clamp small bbox to big bbox (so we don't drift outside)
            min_lon = clamp(small[0], big_bbox[0], big_bbox[2])
            min_lat = clamp(small[1], big_bbox[1], big_bbox[3])
            max_lon = clamp(small[2], big_bbox[0], big_bbox[2])
            max_lat = clamp(small[3], big_bbox[1], big_bbox[3])
            if min_lon >= max_lon or min_lat >= max_lat:
                continue
            small = (min_lon, min_lat, max_lon, max_lat)

            try:
                ids = mapillary_image_search(
                    session,
                    token,
                    small,
                    timeout=args.request_timeout,
                    max_retries=args.max_retries,
                    backoff=args.retry_backoff,
                    limit=args.search_limit,
                )
            except Exception as e:
                print(f"[{i+1}/{args.samples}] search error: {e}")
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
            random.shuffle(ids)

            for image_id in ids:
                if downloaded >= args.target:
                    break
                if image_id in seen:
                    continue

                try:
                    info = mapillary_image_detail(
                        session,
                        token,
                        image_id,
                        thumb_field=thumb_field,
                        timeout=args.request_timeout,
                        max_retries=args.max_retries,
                        backoff=args.retry_backoff,
                    )
                except Exception as e:
                    print(f"detail error for {image_id}: {e}")
                    append_error_row(
                        error_csv,
                        {
                            "ts": str(int(time.time())),
                            "stage": "detail",
                            "sample_i": str(i + 1),
                            "image_id": image_id,
                            "status": "error",
                            "message": str(e)[:400],
                        },
                        error_fields,
                    )
                    time.sleep(args.sleep)
                    continue

                time.sleep(args.sleep)

                if not info:
                    continue

                sequence = str(info.get("sequence", "") or "")
                if args.max_per_sequence > 0 and sequence:
                    if seq_counts.get(sequence, 0) >= args.max_per_sequence:
                        continue

                ext = guess_ext_from_url(info["thumb_url"])
                local_path = out_dir / f"{info['image_id']}{ext}"

                try:
                    download_file(
                        session,
                        info["thumb_url"],
                        local_path,
                        timeout=max(args.request_timeout, 60.0),
                        max_retries=args.max_retries,
                        backoff=args.retry_backoff,
                    )
                except Exception as e:
                    print(f"download error for {info['image_id']}: {e}")
                    append_error_row(
                        error_csv,
                        {
                            "ts": str(int(time.time())),
                            "stage": "download",
                            "sample_i": str(i + 1),
                            "image_id": info["image_id"],
                            "status": "error",
                            "message": str(e)[:400],
                        },
                        error_fields,
                    )
                    time.sleep(args.sleep)
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
                f.flush()
                if args.max_per_sequence > 0 and sequence:
                    seq_counts[sequence] = seq_counts.get(sequence, 0) + 1
                seen.add(info["image_id"])
                downloaded += 1

                if downloaded % 50 == 0:
                    print(f"Downloaded: {downloaded}/{args.target}")

    print(f"DONE. Total downloaded files: {downloaded}")


if __name__ == "__main__":
    main()
