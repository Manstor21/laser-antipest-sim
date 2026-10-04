"""Descarga fotos REALES libres desde GBIF API (sin clave) para diagnóstico sim-to-real.

Uso:
    python sim/dataset/fetch_gbif.py [--base data/real] [--velutina 150] [--abeja 150] [--crabro 80]
    python sim/dataset/fetch_gbif.py --help
    python sim/dataset/fetch_gbif.py --offline   # No network calls, uses cached data only

Variables de entorno:
    LASER_OFFLINE=1  # Deshabilita todas las llamadas de red (equivalente a --offline)

Especies por defecto:
    Vespa velutina  -> 150 imgs -> data/real/gbif_velutina/
    Apis mellifera  -> 150 imgs -> data/real/gbif_abeja/
    Vespa crabro    ->  80 imgs -> data/real/gbif_crabro/

GBIF API sin clave (paginación con offset, limit=300 por llamada):
    https://api.gbif.org/v1/occurrence/search?scientificName=<sp>&mediaType=StillImage&limit=300&offset=<n>

Filtros:
    - Solo registros con media[0..n].identifier (URL directa de imagen).
    - Solo licencia abierta: `license` contiene `creativecommons.org/licenses/by`
      (acepta by, by-nc, by-sa, by-nc-sa) o CC0/publicdomain. Rechaza registros
      sin licencia o con "all rights reserved".

Descarga con 3 reintentos y timeout (httpx, sin subprocess/curl).
Guarda como <taxonKey>_<photoId>.jpg. Omite las ya descargadas (re-ejecutable).
Atribución en data/real/gbif_ATTRIBUTION.csv:
    fichero,especie,fotografo,licencia,url_foto,url_observacion

Límites de seguridad: max 500 ficheros y max 600 MB totales (HEAD antes de
cada descarga; si se supera, para con aviso).
Cortesía: pausa 0.3 s entre llamadas API y 0.2 s entre descargas.

Fallback: si GBIF falla o da <30 imgs por especie, se usa iNaturalist API
abierta (sin clave):
    https://api.inaturalist.org/v1/observations?taxon_name=<sp>&photos=true&quality_grade=research&per_page=200
Solo fotos con license_code CC (cc-by, cc-by-nc, cc-by-sa, cc0...). Se documenta
en consola y en el CSV (columna url_observacion apunta a inaturalist.org).

Llamadas de red realizadas (desactivables con --offline / LASER_OFFLINE=1):
    - GBIF API: occurrence/search (paginado, mediaType=StillImage)
    - iNaturalist API: observations (fallback, quality_grade=research)
    - HEAD requests para Content-Length antes de descargar
    - GET requests para descargar archivos de imagen
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = REPO_ROOT / "data" / "real"

SPECIES_DEFAULTS = [
    {"scientificName": "Vespa velutina", "quota": 150, "dirname": "gbif_velutina"},
    {"scientificName": "Apis mellifera", "quota": 150, "dirname": "gbif_abeja"},
    {"scientificName": "Vespa crabro", "quota": 80, "dirname": "gbif_crabro"},
]

GBIF_SEARCH = "https://api.gbif.org/v1/occurrence/search"
INAT_OBS = "https://api.inaturalist.org/v1/observations"

MAX_FILES_DEFAULT = 500
MAX_MB_DEFAULT = 600
API_PAUSE = 0.3
DL_PAUSE = 0.2
API_TIMEOUT = 30.0
DL_TIMEOUT = 60.0

HEADERS = {"User-Agent": "Velutina-sim/1.0 (diagnostico)"}


def is_offline() -> bool:
    """Check if offline mode is enabled via flag or environment variable."""
    return os.environ.get("LASER_OFFLINE", "0") == "1"


def is_open_license(lic: str | None) -> bool:
    """True si la licencia GBIF es abierta (CC-BY* o CC0/dominio público)."""
    if not lic:
        return False
    low = lic.strip().lower()
    if not low or "all rights reserved" in low or low in ("unknown", "unknown-or-unlisted"):
        return False
    if "creativecommons.org/licenses/by" in low:
        return True
    if "cc0" in low or "publicdomain/zero" in low or "publicdomain/cc0" in low:
        return True
    if "creativecommons.org/publicdomain" in low:
        return True
    return False


def is_inat_open_license(code: str | None) -> bool:
    if not code:
        return False
    c = code.strip().lower()
    return c.startswith("cc-") or c == "cc0"


def http_get_json(client: httpx.Client, url: str, timeout: float = API_TIMEOUT, retries: int = 3) -> dict | None:
    last = None
    for attempt in range(retries):
        try:
            resp = client.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - registra y sigue
            last = exc
            print(f"Aviso: API fallo intento {attempt + 1}/3 {url[:120]}... ({exc})", file=sys.stderr)
            time.sleep(1.0)
    print(f"Aviso: API agotada {url[:120]}... ({last})", file=sys.stderr)
    return None


def head_content_length(client: httpx.Client, url: str, timeout: float = 20.0) -> int | None:
    """Tamaño en bytes vía HEAD (httpx). None si desconocido."""
    try:
        resp = client.head(url, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        cl = resp.headers.get("Content-Length")
        return int(cl) if cl else None
    except Exception:
        return None


def download_url(client: httpx.Client, url: str, dest: Path, timeout: float = DL_TIMEOUT, retries: int = 3) -> tuple[bool, str]:
    """Descarga con 3 reintentos. Devuelve (ok, mensaje)."""
    for attempt in range(retries):
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            with client.stream("GET", url, timeout=timeout, follow_redirects=True) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=8192):
                        fh.write(chunk)
            if tmp.exists() and tmp.stat().st_size > 0:
                tmp.replace(dest)
                return True, "ok httpx"
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            print(f"Aviso: descarga vacía intento {attempt + 1}/3 {url[:100]}...", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"Aviso: descarga intento {attempt + 1}/3 {url[:100]}... ({exc})", file=sys.stderr)
        time.sleep(1.0)
    return False, "fallo tras 3 intentos"


def pick_gbif_media(rec: dict) -> str | None:
    """Primera URL directa de imagen en rec['media'] (jpg/jpeg/png/webp)."""
    media = rec.get("media") or []
    for m in media:
        if not isinstance(m, dict):
            continue
        ident = (m.get("identifier") or "").strip()
        if not ident.startswith("http"):
            continue
        low = ident.lower().split("?")[0]
        if low.endswith((".jpg", ".jpeg", ".png", ".webp")) or "image" in (m.get("format") or "").lower():
            return ident
        # Acepta cualquier URL http de media StillImage aunque no tenga extensión
        if (m.get("type") or "") == "StillImage":
            return ident
    return None


def gbif_candidates(client: httpx.Client, scientific_name: str, quota: int) -> list[dict]:
    """Pagina GBIF (limit=300) hasta reunir `quota` candidatos filtrados."""
    if is_offline():
        print(f"Modo offline: omitiendo GBIF para {scientific_name}", file=sys.stderr)
        return []
    cands: list[dict] = []
    offset = 0
    seen_urls: set[str] = set()
    pages = 0
    while len(cands) < quota and pages < 20:
        q = urllib.parse.urlencode({
            "scientificName": scientific_name,
            "mediaType": "StillImage",
            "limit": 300,
            "offset": offset,
        })
        url = f"{GBIF_SEARCH}?{q}"
        data = http_get_json(client, url)
        time.sleep(API_PAUSE)
        pages += 1
        if not data:
            break
        results = data.get("results") or []
        if not results:
            break
        for rec in results:
            if len(cands) >= quota:
                break
            lic = rec.get("license") or ""
            if not is_open_license(lic):
                continue
            img_url = pick_gbif_media(rec)
            if not img_url or img_url in seen_urls:
                continue
            seen_urls.add(img_url)
            key = rec.get("key")
            taxon_key = rec.get("taxonKey", key)
            rights = rec.get("rightsHolder") or rec.get("creator") or rec.get("recordedBy") or ""
            cands.append({
                "file_id": f"{taxon_key}_{key}",
                "species": scientific_name,
                "photographer": str(rights).strip(),
                "license": lic,
                "url_foto": img_url,
                "url_obs": f"https://www.gbif.org/occurrence/{key}",
            })
        total = data.get("count", 0)
        offset += len(results)
        if offset >= total:
            break
    return cands[:quota]


def inat_candidates(client: httpx.Client, scientific_name: str, quota: int) -> list[dict]:
    """Fallback iNaturalist: observaciones research-grade con fotos CC."""
    if is_offline():
        print(f"Modo offline: omitiendo iNaturalist para {scientific_name}", file=sys.stderr)
        return []
    cands: list[dict] = []
    seen: set[str] = set()
    page = 1
    while len(cands) < quota and page <= 10:
        q = urllib.parse.urlencode({
            "taxon_name": scientific_name,
            "photos": "true",
            "quality_grade": "research",
            "per_page": 200,
            "page": page,
            "order_by": "votes",
        })
        data = http_get_json(client, f"{INAT_OBS}?{q}")
        time.sleep(API_PAUSE)
        page += 1
        if not data:
            break
        for obs in data.get("results") or []:
            if len(cands) >= quota:
                break
            obs_id = obs.get("id")
            for ph in obs.get("photos") or []:
                if len(cands) >= quota:
                    break
                lic = (ph.get("license_code") or "").strip()
                if not is_inat_open_license(lic):
                    continue
                raw = (ph.get("url") or "").strip()
                if not raw.startswith("http"):
                    continue
                # iNat da URLs square (.../square.jpg); pide medium/large.
                img_url = raw.replace("/square.", "/medium.").replace("square.jpg", "medium.jpg")
                if img_url in seen:
                    continue
                seen.add(img_url)
                pid = ph.get("id", len(seen))
                cands.append({
                    "file_id": f"{obs_id}_{pid}",
                    "species": scientific_name,
                    "photographer": str(ph.get("attribution") or obs.get("user", {}).get("login") or "").strip()[:200],
                    "license": lic,
                    "url_foto": img_url,
                    "url_obs": f"https://www.inaturalist.org/observations/{obs_id}",
                })
    return cands[:quota]


def load_existing_attribution(csv_path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if csv_path.exists():
        try:
            with open(csv_path, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r.get("fichero"):
                        rows[r["fichero"]] = r
        except Exception as exc:  # noqa: BLE001
            print(f"Aviso: no se pudo leer {csv_path} ({exc}), se reescribe.", file=sys.stderr)
    return rows


def current_totals(base: Path) -> tuple[int, int]:
    files = [p for p in base.rglob("*.jpg")] if base.exists() else []
    total = 0
    for p in files:
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return len(files), total


def fetch_species(client: httpx.Client, spec: dict, base: Path, attrib: dict[str, dict],
                  max_files: int, max_bytes: int) -> tuple[int, int, int, bool]:
    """Descarga una especie. Devuelve (nuevas, omitidas, fallos, uso_fallback)."""
    outdir = base / spec["dirname"]
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"== {spec['scientificName']} -> {outdir} (objetivo {spec['quota']}) ==")
    cands = gbif_candidates(client, spec["scientificName"], spec["quota"])
    used_fallback = False
    if len(cands) < 30:
        print(f"Aviso: GBIF dio {len(cands)} candidatos (<30) para {spec['scientificName']}; "
              f"probando fallback iNaturalist (documentado).")
        inat = inat_candidates(client, spec["scientificName"], spec["quota"])
        print(f"Fallback iNaturalist: {len(inat)} candidatos para {spec['scientificName']}.")
        # Fusiona sin duplicar URLs
        urls = {c["url_foto"] for c in cands}
        for c in inat:
            if c["url_foto"] not in urls and len(cands) < spec["quota"]:
                cands.append(c)
                urls.add(c["url_foto"])
        used_fallback = True
    nuevas, omitidas, fallos = 0, 0, 0
    for c in cands:
        fname = f"{c['file_id']}.jpg"
        dest = outdir / fname
        if dest.exists() and dest.stat().st_size > 0:
            omitidas += 1
            attrib.setdefault(fname, {
                "fichero": fname, "especie": c["species"], "fotografo": c["photographer"],
                "licencia": c["license"], "url_foto": c["url_foto"], "url_observacion": c["url_obs"],
            })
            continue
        n_files, n_bytes = current_totals(base)
        if n_files >= max_files:
            print(f"Aviso: límite de seguridad {max_files} ficheros alcanzado, se para.", file=sys.stderr)
            break
        size = head_content_length(client, c["url_foto"])
        if size and n_bytes + size > max_bytes:
            print(f"Aviso: límite de seguridad {max_bytes / 1e6:.0f}MB alcanzado "
                  f"({n_bytes / 1e6:.1f}MB + {size / 1e6:.1f}MB), se para.", file=sys.stderr)
            break
        ok, msg = download_url(client, c["url_foto"], dest)
        time.sleep(DL_PAUSE)
        if ok:
            nuevas += 1
            attrib[fname] = {
                "fichero": fname, "especie": c["species"], "fotografo": c["photographer"],
                "licencia": c["license"], "url_foto": c["url_foto"], "url_observacion": c["url_obs"],
            }
        else:
            fallos += 1
            print(f"Aviso: se omite {fname} ({msg}), se sigue.", file=sys.stderr)
    print(f"   {spec['scientificName']}: nuevas={nuevas}, omitidas={omitidas}, fallos={fallos}")
    return nuevas, omitidas, fallos, used_fallback


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Descarga fotos reales libres (GBIF, fallback iNaturalist).")
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--velutina", type=int, default=150)
    ap.add_argument("--abeja", type=int, default=150)
    ap.add_argument("--crabro", type=int, default=80)
    ap.add_argument("--max-files", type=int, default=MAX_FILES_DEFAULT)
    ap.add_argument("--max-mb", type=int, default=MAX_MB_DEFAULT)
    ap.add_argument("--offline", action="store_true", help="Deshabilita todas las llamadas de red (usa solo datos en caché)")
    args = ap.parse_args(argv)

    # Set offline mode from flag
    if args.offline:
        os.environ["LASER_OFFLINE"] = "1"

    if is_offline():
        print("Modo OFFLINE activado (--offline o LASER_OFFLINE=1). No se harán llamadas de red.", file=sys.stderr)

    base = Path(args.base)
    base.mkdir(parents=True, exist_ok=True)
    specs = [
        {"scientificName": "Vespa velutina", "quota": args.velutina, "dirname": "gbif_velutina"},
        {"scientificName": "Apis mellifera", "quota": args.abeja, "dirname": "gbif_abeja"},
        {"scientificName": "Vespa crabro", "quota": args.crabro, "dirname": "gbif_crabro"},
    ]
    csv_path = base / "gbif_ATTRIBUTION.csv"
    attrib = load_existing_attribution(csv_path)
    max_bytes = int(args.max_mb * 1_000_000)
    total_new = 0
    any_fallback = False

    # Create httpx client with connection pooling
    with httpx.Client(headers=HEADERS, timeout=httpx.Timeout(API_TIMEOUT, connect=10.0)) as client:
        for spec in specs:
            nuevas, _omit, _fail, fb = fetch_species(client, spec, base, attrib, args.max_files, max_bytes)
            total_new += nuevas
            any_fallback = any_fallback or fb

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["fichero", "especie", "fotografo", "licencia", "url_foto", "url_observacion"])
        w.writeheader()
        for fname in sorted(attrib):
            w.writerow(attrib[fname])
    n_files, n_bytes = current_totals(base)
    print(f"Total: {n_files} ficheros, {n_bytes / 1e6:.1f} MB en {base}. Atribución: {csv_path}")
    if any_fallback:
        print("Nota: se usó fallback iNaturalist en al menos una especie (ver arriba).")
    for spec in specs:
        n = len(list((base / spec['dirname']).glob("*.jpg"))) if (base / spec["dirname"]).exists() else 0
        print(f"  {spec['scientificName']}: {n} fotos en {spec['dirname']}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())