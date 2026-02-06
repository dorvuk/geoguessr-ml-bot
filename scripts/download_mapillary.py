import argparse
import csv
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests


GRAPH = "https://graph.mapillary.com"


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
            if row.get("image_id"):
                seen.add(row["image_id"])
    return seen


def ensure_csv_header(meta_csv: Path, fieldnames: List[str]) -> None:
    if meta_csv.exists():
        return
    meta_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def rand_point_in_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> Tuple[float, float]:
    lon = random.uniform(min_lon, max_lon)
    lat = random.uniform(min_lat, max_lat)
    return lon, lat


def small_bbox_around(lon: float, lat: float, half_size_deg: float) -> Tuple[float, float, float, float]:
    # bbox order required by Mapillary is: min_lon,min_lat,max_lon,max_lat :contentReference[oaicite:3]{index=3}
    return (
        lon - half_size_deg,
        lat - half_size_deg,
        lon + half_size_deg,
        lat + half_size_deg,
    )


def mapillary_image_search(
    token: str,
    bbox: Tuple[float, float, float, float],
    limit: int = 200,
) -> List[str]:
    params = {
        "fields": "id",
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "limit": str(limit),
        "access_token": token,
    }
    url = f"{GRAPH}/images"
    r = requests.get(url, params=params, timeout=30)
    if r.status_code >= 400:
        # Include a short body snippet to explain the failure.
        raise RuntimeError(f"{r.status_code} {r.text[:400]}")
    data = r.json()
    return [str(item["id"]) for item in data.get("data", []) if "id" in item]


def mapillary_image_detail(
    token: str,
    image_id: str,
    thumb_field: str,
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
    r = requests.get(url, params=params, timeout=30)
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


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def guess_ext_from_url(url: str) -> str:
    # Usually .jpg. If there's a querystring, strip it.
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
    ap.add_argument("--sleep", type=float, default=0.05, help="Sleep between API calls (seconds)")
    args = ap.parse_args()

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

    fieldnames = [
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
    ensure_csv_header(meta_csv, fieldnames)
    seen = read_seen_ids(meta_csv)

    thumb_field = f"thumb_{args.thumb}_url"

    downloaded = len(seen)

    print(f"Existing metadata rows: {len(seen)}")
    print(f"Existing files in {out_dir}: {downloaded}")
    print(f"Target total files: {args.target}")

    # Open metadata for append
    with open(meta_csv, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        seq_counts: Dict[str, int] = {}

        for i in range(args.samples):
            if downloaded >= args.target:
                break

            lon, lat = rand_point_in_bbox(*big_bbox)
            small = small_bbox_around(lon, lat, args.half_size_deg)

            # Clamp small bbox to big bbox (so we don't drift outside)
            min_lon = clamp(small[0], big_bbox[0], big_bbox[2])
            min_lat = clamp(small[1], big_bbox[1], big_bbox[3])
            max_lon = clamp(small[2], big_bbox[0], big_bbox[2])
            max_lat = clamp(small[3], big_bbox[1], big_bbox[3])
            if min_lon >= max_lon or min_lat >= max_lat:
                continue
            small = (min_lon, min_lat, max_lon, max_lat)

            try:
                ids = mapillary_image_search(token, small, limit=args.search_limit)
            except Exception as e:
                print(f"[{i+1}/{args.samples}] search error: {e}")
                time.sleep(args.sleep)
                continue

            time.sleep(args.sleep)

            # Shuffle so we don't always pull the same "top" ordering
            random.shuffle(ids)

            for image_id in ids:
                if downloaded >= args.target:
                    break
                if image_id in seen:
                    continue

                try:
                    info = mapillary_image_detail(token, image_id, thumb_field=thumb_field)
                except Exception as e:
                    print(f"detail error for {image_id}: {e}")
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
                    download_file(info["thumb_url"], local_path)
                except Exception as e:
                    print(f"download error for {info['image_id']}: {e}")
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
