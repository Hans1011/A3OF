import os
import re
import time
import json
import asyncio
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import pandas as pd
import requests

from browser_use import Agent
from browser_use.llm import ChatOpenAI


# ========= 0. API Key & Config (from .env) =========

API_KEY = os.getenv("OPENAI_API_KEY", "")
BROWSER_USE_API_KEY = os.getenv("BROWSER_USE_API_KEY", "")

CONSENSUS_EMAIL = os.getenv("CONSENSUS_EMAIL", "")
CONSENSUS_PASSWORD = os.getenv("CONSENSUS_PASSWORD", "")

PROJECT_DIR = Path(__file__).resolve().parent
QUERY_RESULTS_PATH = str(PROJECT_DIR / "query_results.json")

def _load_search_query():
    """Lazy-load query2 from query_results.json (created by Query Agent)."""
    if not os.path.exists(QUERY_RESULTS_PATH):
        raise FileNotFoundError(f"query_results.json not found at {QUERY_RESULTS_PATH}. "
                                "Run Query Agent first.")
    with open(QUERY_RESULTS_PATH, "r", encoding="utf-8") as f:
        query_data = json.load(f)
    return query_data.get("query2", "")

DOWNLOAD_DIR = (PROJECT_DIR / "downloads").resolve()
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

DOI_COLUMN = "DOI"
OUTPUT_DIR = str(PROJECT_DIR / "Papers")
YOUR_EMAIL = CONSENSUS_EMAIL
SLEEP_SECONDS = 1.0


def clean_filename(name: str, max_len: int = 180) -> str:
    name = re.sub(r'[\\/*?:"<>|]+', "_", str(name))
    name = re.sub(r"\s+", " ", name).strip()
    return name[:max_len] if len(name) > max_len else name


def normalize_doi(doi: str) -> str | None:
    if pd.isna(doi):
        return None
    doi = str(doi).strip()
    if not doi:
        return None
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    doi = doi.replace("doi:", "").strip()
    return doi or None


def get_unpaywall_info(doi: str, email: str) -> dict:
    url = f"https://api.unpaywall.org/v2/{doi}"
    params = {"email": email}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code == 404:
        return {"found": False, "reason": "not_in_unpaywall"}
    r.raise_for_status()
    data = r.json()
    best_pdf = None
    best_oa = data.get("best_oa_location") or {}
    if best_oa:
        best_pdf = best_oa.get("url_for_pdf") or best_oa.get("url")
    return {
        "found": True,
        "is_oa": data.get("is_oa"),
        "oa_status": data.get("oa_status"),
        "title": data.get("title"),
        "journal_name": data.get("journal_name"),
        "year": data.get("year"),
        "doi_url": data.get("doi_url"),
        "pdf_url": best_pdf,
    }


def get_crossref_info(doi: str, email: str) -> dict:
    url = f"https://api.crossref.org/works/{doi}"
    headers = {"User-Agent": f"DOI-downloader/1.0 (mailto:{email})"}
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code == 404:
        return {"found": False}
    r.raise_for_status()
    data = r.json()["message"]
    title = ""
    if data.get("title"):
        title = data["title"][0]
    journal = ""
    if data.get("container-title"):
        journal = data["container-title"][0]
    year = None
    for key in ["published-print", "published-online", "issued"]:
        if key in data and data[key].get("date-parts"):
            year = data[key]["date-parts"][0][0]
            break
    links = data.get("link", [])
    fulltext_links = [x.get("URL") for x in links if x.get("URL")]
    return {
        "found": True,
        "title": title,
        "journal_name": journal,
        "year": year,
        "doi_url": f"https://doi.org/{doi}",
        "fulltext_links": fulltext_links,
    }


def download_file(url: str, save_path: Path) -> tuple[bool, str]:
    try:
        with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "").lower()
            if "pdf" not in content_type and not str(r.url).lower().endswith(".pdf"):
                return False, f"not_pdf_content_type={content_type}"
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
        if save_path.stat().st_size == 0:
            return False, "empty_file"
        return True, "downloaded"
    except Exception as e:
        return False, str(e)


async def download_csv_from_consensus() -> Path | None:
    if not API_KEY:
        raise ValueError("No OPENAI_API_KEY")

    before_files = {p.name for p in DOWNLOAD_DIR.iterdir() if p.is_file()}

    llm = ChatOpenAI(
        model="gpt-5-mini",
        api_key=API_KEY,
    )

    agent = Agent(
        task=f"""
        Open https://consensus.app.

        Find the "Sign in" button and log into the account.

        Login procedure:
        1. Click the "Sign in" button located at the bottom-left corner.
        2. Choose the login option and enter the email address:
           {CONSENSUS_EMAIL}
        3. Wait for the page to load.
        4. Do NOT enter any email verification code.
        5. Find and click the "Use password" option below the verification input.
        6. Enter the password:
           {CONSENSUS_PASSWORD}
        7. Complete the sign-in process.

        After successful login:
        1. Search for the following query:
           "{_load_search_query()}"

        2. Wait until the search results are fully loaded.
        3. On the right side of the Consensus results page, locate the download button.
        4. Click the download button to download the CSV file containing the article information.
        5. IMPORTANT: The CSV will be saved to the browser's temp download folder automatically.
           Do NOT try to read, open, or copy the CSV file. Just confirm the download completed.
           The filename may contain Chinese characters — this is expected, do not try to rename or read it.

        Important instructions:
        - Your ONLY job is to navigate, log in, search, and click download. Then call done().
        - Do NOT try to read or verify the downloaded file contents.
        - If a popup appears, close it and continue.
        """,
        llm=llm,
    )

    new_csv_path = None

    try:
        result = await agent.run()
        print("\n[INFO] Agent success")
        print(result)
    except Exception as e:
        print(f"\n[ERROR] Agent failed: {e}")
        raise
    finally:
        await asyncio.sleep(5)

        # 同时扫描 DOWNLOAD_DIR 和 browser-use 临时下载目录
        import glob as _glob
        _temp_dirs = list(Path.home().glob(
            "AppData/Local/Temp/browser-use-downloads-*"
        ))
        _all_dirs = [DOWNLOAD_DIR] + _temp_dirs
        _found_csvs = []

        for _dir in _all_dirs:
            if _dir.exists():
                for _f in _dir.iterdir():
                    if _f.is_file() and _f.suffix.lower() == '.csv':
                        _found_csvs.append(_f)

        if _found_csvs:
            new_csv_path = max(_found_csvs, key=lambda p: p.stat().st_mtime)
            print(f"\n[INFO] Found CSV: {new_csv_path} ({new_csv_path.stat().st_size} bytes)")

            if new_csv_path.parent != DOWNLOAD_DIR:
                _dest = DOWNLOAD_DIR / new_csv_path.name
                _dest.write_bytes(new_csv_path.read_bytes())
                print(f"[INFO] Copied to: {_dest}")
                new_csv_path = _dest
        else:
            print("\n[WARNING] No CSV found in download dirs")
            for _dir in _all_dirs:
                if _dir.exists():
                    print(f"  Contents of {_dir}:")
                    for _f in sorted(_dir.iterdir(), key=lambda x: x.name.lower())[-10:]:
                        try:
                            print(f"    - {_f.name} ({_f.stat().st_size} bytes)")
                        except OSError:
                            print(f"    - {_f.name}")

    return new_csv_path


def download_pdfs_from_csv(csv_path: str | Path):
    csv_path = Path(csv_path)

    out_dir = Path(OUTPUT_DIR)
    pdf_dir = out_dir  # download directly into Papers/
    meta_dir = out_dir / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    if DOI_COLUMN not in df.columns:
        raise ValueError(f"CSV cannot find: {DOI_COLUMN}got: {list(df.columns)}")

    results = []

    for idx, raw_doi in enumerate(df[DOI_COLUMN], start=1):
        doi = normalize_doi(raw_doi)
        print(f"[{idx}/{len(df)}] DOI = {doi}")

        row_result = {
            "raw_doi": raw_doi,
            "doi": doi,
            "status": "",
            "title": "",
            "journal": "",
            "year": "",
            "doi_url": "",
            "pdf_url": "",
            "pdf_path": "",
            "note": "",
        }

        if not doi:
            row_result["status"] = "invalid_doi"
            results.append(row_result)
            continue

        try:
            upw = get_unpaywall_info(doi, YOUR_EMAIL)
            title = upw.get("title") if upw.get("found") else ""
            journal = upw.get("journal_name") if upw.get("found") else ""
            year = upw.get("year") if upw.get("found") else ""
            doi_url = upw.get("doi_url") if upw.get("found") else f"https://doi.org/{doi}"
            pdf_url = upw.get("pdf_url") if upw.get("found") else ""

            if not title or not journal or not year:
                cr = get_crossref_info(doi, YOUR_EMAIL)
                if cr.get("found"):
                    title = title or cr.get("title", "")
                    journal = journal or cr.get("journal_name", "")
                    year = year or cr.get("year", "")
                    doi_url = doi_url or cr.get("doi_url", f"https://doi.org/{doi}")
                    if not pdf_url:
                        links = cr.get("fulltext_links", [])
                        if links:
                            pdf_url = links[0]

            row_result["title"] = title
            row_result["journal"] = journal
            row_result["year"] = year
            row_result["doi_url"] = doi_url
            row_result["pdf_url"] = pdf_url

            if pdf_url:
                base_name = clean_filename(doi)
                pdf_path = pdf_dir / f"{base_name}.pdf"
                ok, msg = download_file(pdf_url, pdf_path)
                if ok:
                    row_result["status"] = "downloaded"
                    row_result["pdf_path"] = str(pdf_path)
                    row_result["note"] = msg
                else:
                    row_result["status"] = "pdf_not_downloaded"
                    row_result["note"] = msg
            else:
                row_result["status"] = "no_open_pdf_found"
                row_result["note"] = "No OA PDF found via Unpaywall / Crossref"

            meta_path = meta_dir / f"{clean_filename(doi)}.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(row_result, f, ensure_ascii=False, indent=2)

        except Exception as e:
            row_result["status"] = "error"
            row_result["note"] = str(e)

        results.append(row_result)
        time.sleep(SLEEP_SECONDS)

    result_df = pd.DataFrame(results)
    result_csv = out_dir / "download_results.csv"
    result_df.to_csv(result_csv, index=False, encoding="utf-8-sig")

    print("\nResult saved")
    print(result_csv)


async def main():
    csv_path = await download_csv_from_consensus()
    if not csv_path:
        return
    download_pdfs_from_csv(csv_path)


def run(state: dict) -> dict:
    """
    LangGraph node: Mining Agent.
    Uses Consensus API (browser-use) to search for papers matching query2,
    downloads PDFs directly into Papers/ directory.
    """
    csv_path = asyncio.run(download_csv_from_consensus())
    if csv_path:
        download_pdfs_from_csv(csv_path)
    state["downloaded_papers_dir"] = str(PROJECT_DIR / "Papers")
    return state


if __name__ == "__main__":
    asyncio.run(main())
