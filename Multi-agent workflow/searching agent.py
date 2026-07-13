import os
import re
import sys
import glob
import json
import time
import shutil
import hashlib
import zipfile
import asyncio
from pathlib import Path
from functools import wraps
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import requests
import chromadb
from chromadb.api.types import EmbeddingFunction
from openai import OpenAI
import pypdf

from fast_graphrag import GraphRAG, QueryParam
from fast_graphrag._llm import OpenAILLMService, OpenAIEmbeddingService
from fast_graphrag._utils import logger


# ============================================================
# Config
# ============================================================

OPENAI_API_KEY = os.getenv("AGICTO_API_KEY")
OPENAI_BASE_URL = os.getenv("AGICTO_BASE_URL", "https://api.agicto.cn/v1")

MINERU_TOKEN = os.getenv("MINERU_TOKEN")
MINERU_API_BASE = "https://mineru.net"

MINERU_MODEL_VERSION = "vlm"
MINERU_ENABLE_FORMULA = True
MINERU_ENABLE_TABLE = True
MINERU_IS_OCR = False
MINERU_LANGUAGE = "en"
MINERU_POLL_INTERVAL_SEC = 8
MINERU_POLL_TIMEOUT_SEC = 3600

MODEL_NAME_QA = "gpt-4o-mini"
LLM_MODEL_FOR_GRAPHRAG = os.getenv("LLM_MODEL_FOR_GRAPHRAG", "gpt-5-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

DOMAIN = (
    "Analyze scientific documents. Identify key concepts, methods, "
    "experimental conditions, variables, and relationships between them."
)

EXAMPLE_QUERIES = [
    "What materials and methods are used in the experiment?",
    "What parameters affect the experimental results?",
    "How does the process influence the final performance?",
    "What relationships exist between materials, processes, and results?",
]

ENTITY_TYPES = [
    "Concept",
    "Method",
    "Parameter",
    "Material",
    "Process",
    "Result",
    "Device",
]

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
OVERLAP = int(os.getenv("OVERLAP", "200"))
SHOW_TOPK = 20
SHOW_CHUNK_SNIPPET_CHARS = 1000
ENABLE_GRAPHRAG_STAGE_LOGS = True
ENABLE_PROFILE_CALLTREE = False

PROJECT_DIR = Path(__file__).resolve().parent

PDF_DIR = PROJECT_DIR / "Papers"
MINERU_OUT_DIR = PROJECT_DIR / "mineru_outputs"
TXT_DIR = PROJECT_DIR / "Converted txt"
GRAPH_DB_DIR = PROJECT_DIR / "graph_db-2"
CHROMA_DB_DIR = PROJECT_DIR / "rag_db-2"
CHROMA_COLLECTION_NAME = "papers"

client_openai = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


class _PatchedEmbeddingService(OpenAIEmbeddingService):

    async def _embedding_request(self, input, model):
        async with self.embedding_max_requests_concurrent:
            async with self.embedding_per_minute_limiter:
                async with self.embedding_per_second_limiter:
                    return await self.embedding_async_client.embeddings.create(
                        model=model, input=input, encoding_format="float"
                    )


# ============================================================
# Section A: Question Decomposition
# ============================================================

def to_snake_case(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def parse_llm_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^```", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def llm_call(prompt: str) -> str:
    response = client_openai.chat.completions.create(
        model=MODEL_NAME_QA,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a scientific text understanding assistant. "
                    "You extract structured relationships from compound scientific questions. "
                    "Return only the requested output."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )
    return response.choices[0].message.content


def build_extraction_prompt(question: str) -> str:
    return f"""
You are given a compound scientific question.

Your task is to decompose it into independent factor-response-outcome relationships.

Return ONLY valid JSON.

Required JSON format:
{{
  "relationships": [
    {{
      "factor": "...",
      "response_pattern": "...",
      "outcome": "..."
    }}
  ]
}}

Rules:
- Extract each independent factor separately.
- Extract the response pattern associated with each factor.
- Extract the common outcome or performance metric.
- Keep the original scientific meaning.
- Do not answer the scientific question.
- Do not explain mechanisms.
- Do not invent missing factors.
- Do not invent missing response patterns.
- If multiple factors and multiple response patterns are listed, align them by order unless the sentence clearly implies another alignment.
- response_pattern must contain ONLY the raw pattern name or shape, without any article ("a"/"an") and without the word "response". The words "follow a ... response" will be prepended/appended automatically by the template.
- outcome must contain ONLY the raw performance metric name, without the word "controlling". The word "controlling" will be prepended automatically by the template.
- Return JSON only.

Question:
{question}
"""


def format_mechanistic_question(item: Dict[str, str]) -> str:
    factor = to_snake_case(item["factor"])
    response_pattern = item["response_pattern"].strip()
    outcome = item["outcome"].strip()
    return (
        f"Why does {factor} follow a {response_pattern} response, "
        f"controlling {outcome}?"
    )


def decompose_question_to_mechanistic_questions(question: str) -> List[str]:
    prompt = build_extraction_prompt(question)
    raw_response = llm_call(prompt)
    data = parse_llm_json(raw_response)
    relationships = data.get("relationships", [])
    if not relationships:
        raise ValueError("LLM did not extract any relationships.")
    return [format_mechanistic_question(item) for item in relationships]


# ============================================================
# Section B: PDF Indexing via MinerU + GraphRAG
# ============================================================

def _index_make_safe_name(pdf_name, prefix_len=20):
    clean = re.sub(r'[\\/:*?"<>|]', '_', pdf_name)
    clean = re.sub(r'\s+', '_', clean)
    h = hashlib.md5(pdf_name.encode("utf-8")).hexdigest()[:8]
    prefix = clean[:prefix_len]
    return f"{prefix}_{h}"


_MINERU_HEADERS_JSON = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {MINERU_TOKEN}",
}


def _mineru_apply_upload_urls(file_paths):
    url = f"{MINERU_API_BASE}/api/v4/file-urls/batch"
    files_payload = []
    for p in file_paths:
        name = Path(p).name
        stem = Path(name).stem
        data_id = hashlib.md5(stem.encode("utf-8")).hexdigest()[:16]
        files_payload.append({
            "name": name,
            "data_id": data_id,
            "is_ocr": MINERU_IS_OCR,
        })
    payload = {
        "files": files_payload,
        "model_version": MINERU_MODEL_VERSION,
        "enable_formula": MINERU_ENABLE_FORMULA,
        "enable_table": MINERU_ENABLE_TABLE,
        "language": MINERU_LANGUAGE,
    }
    r = requests.post(url, headers=_MINERU_HEADERS_JSON, json=payload, timeout=60)
    r.raise_for_status()
    j = r.json()
    if j.get("code") != 0:
        raise RuntimeError(f"apply_upload_urls failed: {j}")
    return j["data"]["batch_id"], j["data"]["file_urls"]


def _mineru_upload_files(file_paths, upload_urls):
    if len(file_paths) != len(upload_urls):
        raise ValueError("file_paths and upload_urls mismatch")
    for path, put_url in zip(file_paths, upload_urls):
        print(f"Uploading: {Path(path).name}")
        with open(path, "rb") as f:
            resp = requests.put(put_url, data=f, timeout=600)
        if resp.status_code != 200:
            raise RuntimeError(f"Upload failed: {path}, status={resp.status_code}")
        print("Uploaded")


def _mineru_get_batch_results(batch_id):
    url = f"{MINERU_API_BASE}/api/v4/extract-results/batch/{batch_id}"
    r = requests.get(url, headers=_MINERU_HEADERS_JSON, timeout=60)
    r.raise_for_status()
    j = r.json()
    if j.get("code") != 0:
        raise RuntimeError(f"get_batch_results failed: {j}")
    return j["data"]["extract_result"]


def _mineru_poll_until_done(batch_id, interval_sec=MINERU_POLL_INTERVAL_SEC, timeout_sec=MINERU_POLL_TIMEOUT_SEC):
    start = time.time()
    while True:
        results = _mineru_get_batch_results(batch_id)
        states = [x.get("state") for x in results]
        if all(s in ("done", "failed") for s in states):
            return results
        if time.time() - start > timeout_sec:
            raise TimeoutError(f"Polling timeout after {timeout_sec}s")
        brief = []
        for x in results:
            name = x.get("file_name", "unknown")
            st = x.get("state")
            prog = x.get("extract_progress", {})
            if prog:
                brief.append(f"{name}:{st}({prog.get('extracted_pages')}/{prog.get('total_pages')})")
            else:
                brief.append(f"{name}:{st}")
        print("......", " | ".join(brief))
        time.sleep(interval_sec)


def _mineru_download_zip(zip_url, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(zip_url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def _mineru_unzip(zip_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    return out_dir


def _mineru_find_first_markdown(unzipped_dir):
    unzipped_dir = Path(unzipped_dir)
    md_files = list(unzipped_dir.rglob("*.md"))
    if md_files:
        md_files.sort(key=lambda p: p.stat().st_size, reverse=True)
        return md_files[0]
    return None


def _mineru_md_to_txt_keep_latex(md_text: str) -> str:
    text = md_text
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', text)
    text = re.sub(r'```[\s\S]*?```', lambda m: m.group(0).strip('`'), text)
    text = re.sub(r'^\s{0,3}#{1,6}\s+', '', text, flags=re.M)
    text = re.sub(r'^\s{0,3}>\s?', '', text, flags=re.M)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.M)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.M)
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def mineru_pdf_to_txt(pdf_dir=None, out_dir=None):
    if pdf_dir is None:
        pdf_dir = PDF_DIR
    if out_dir is None:
        out_dir = MINERU_OUT_DIR
    pdf_dir = Path(pdf_dir)
    out_dir = Path(out_dir)
    # Start fresh — remove outputs from previous runs
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted([p for p in pdf_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])
    if not pdf_files:
        print(f"[INFO] No PDFs found in {pdf_dir}, skipping MinerU step")
        return

    BATCH_SIZE = 200
    for batch_start in range(0, len(pdf_files), BATCH_SIZE):
        chunk = pdf_files[batch_start: batch_start + BATCH_SIZE]
        file_paths = [str(p) for p in chunk]
        print(f"\n=== Batch {batch_start//BATCH_SIZE + 1}: {len(chunk)} files ===")
        batch_id, upload_urls = _mineru_apply_upload_urls(file_paths)
        print("batch_id =", batch_id)
        _mineru_upload_files(file_paths, upload_urls)
        print("All files uploaded")
        results = _mineru_poll_until_done(batch_id)
        print("Start converting to txt")
        for item in results:
            name = item.get("file_name", "unknown.pdf")
            state = item.get("state")
            stem = Path(name).stem
            safe_name = _index_make_safe_name(stem)
            file_out = out_dir / safe_name
            file_out.mkdir(parents=True, exist_ok=True)
            if state != "done":
                err = item.get("err_msg", "")
                print(f" {name} failed: {err}")
                continue
            zip_url = item["full_zip_url"]
            zip_path = file_out / "result.zip"
            unzipped_dir = file_out / "unzipped"
            if unzipped_dir.exists():
                shutil.rmtree(unzipped_dir, ignore_errors=True)
            print(f"Downloading zip: {name}")
            _mineru_download_zip(zip_url, zip_path)
            _mineru_unzip(zip_path, unzipped_dir)
            md_path = _mineru_find_first_markdown(unzipped_dir)
            if not md_path:
                continue
            md_text = md_path.read_text(encoding="utf-8", errors="ignore")
            txt_text = _mineru_md_to_txt_keep_latex(md_text)
            safe_txt_name = _index_make_safe_name(stem) + ".txt"
            txt_path = file_out / safe_txt_name
            txt_path.write_text(txt_text, encoding="utf-8")
            print(f"{name} -> {txt_path}")
    print("\nMinerU finished")


def collect_txt_outputs(source_dir=None, target_dir=None):
    if source_dir is None:
        source_dir = MINERU_OUT_DIR
    if target_dir is None:
        target_dir = TXT_DIR
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    # Clear target dir so only this run's files are present
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    txt_files = sorted(source_dir.rglob("*.txt"))
    if not txt_files:
        print(f"[INFO] No TXT files in {source_dir}")
        return
    for txt_path in txt_files:
        target_path = target_dir / txt_path.name
        counter = 1
        while target_path.exists():
            target_path = target_dir / f"{txt_path.stem}_{counter}{txt_path.suffix}"
            counter += 1
        shutil.copy2(txt_path, target_path)


def _read_txt_for_index(path: Path) -> str:
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            return path.read_text(encoding=enc).strip()
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Failed to read {path}")


async def graphrag_index_main(working_dir=None, txt_dir=None):
    if working_dir is None:
        working_dir = GRAPH_DB_DIR
    if txt_dir is None:
        txt_dir = TXT_DIR
    working_dir = Path(working_dir)
    txt_dir = Path(txt_dir)
    working_dir.mkdir(exist_ok=True)

    grag = GraphRAG(
        working_dir=str(working_dir),
        domain=DOMAIN,
        entity_types=ENTITY_TYPES,
        example_queries=EXAMPLE_QUERIES,
        config=GraphRAG.Config(
            llm_service=OpenAILLMService(
                model="gpt-4o-mini",
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
            ),
            embedding_service=_PatchedEmbeddingService(
                model="text-embedding-3-small",
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                embedding_dim=1536,
                max_elements_per_request=2,
                max_requests_concurrent=3,
                max_requests_per_minute=200,
            ),
        ),
    )

    txt_files = sorted(txt_dir.glob("*.txt"))
    if not txt_files:
        print(f"[INFO] No TXT files in {txt_dir}, skipping GraphRAG indexing")
        return

    print(f"Start indexing {len(txt_files)} TXT files")
    for i, txt_path in enumerate(txt_files, 1):
        try:
            text = _read_txt_for_index(txt_path)
            if not text:
                print(f"Skip: {txt_path.name}")
                continue

            # Retry up to 3 times with backoff for embedding API errors
            for attempt in range(1, 4):
                try:
                    await grag.async_insert(text)
                    print(f"[{i}/{len(txt_files)}] Indexed: {txt_path.name}")
                    break
                except Exception as e:
                    if attempt < 3:
                        wait = 30 * attempt
                        print(f"  Retry {attempt}/3 after {wait}s (embedding API error): {e}")
                        await asyncio.sleep(wait)
                    else:
                        raise

            # Pause between files to avoid hammering the embedding API
            if i < len(txt_files):
                await asyncio.sleep(15)

        except Exception:
            logger.exception(f"Failed: {txt_path.name}", exc_info=True)
    print("GraphRAG indexing finished")


# ============================================================
# Section C: Evidence Retrieval (GraphRAG + Chroma)
# ============================================================

_EVIDENCE_DIR = PROJECT_DIR / "evidence_default"
_OUTPUT_SOURCE_DIR = PROJECT_DIR / "sources_default"
_FIXED_QUERY = ""
_RUN_GRAPHRAG = True
_RUN_CHROMA = True
_BUILD_CHROMA_INDEX = False
_GRAPHRAG_WORKING_DIR = GRAPH_DB_DIR
_CHROMA_DB_DIR = CHROMA_DB_DIR
_SOURCE_DIR = TXT_DIR


def _ev_ensure_dirs(evidence_dir, output_source_dir):
    Path(evidence_dir).mkdir(parents=True, exist_ok=True)
    Path(output_source_dir).mkdir(parents=True, exist_ok=True)


def _ev_save_text(filename: str, text: str, evidence_dir):
    _ev_ensure_dirs(evidence_dir, _OUTPUT_SOURCE_DIR)
    path = Path(evidence_dir) / filename
    path.write_text(text, encoding="utf-8")
    print(f"[SAVE] {path.resolve()}")


def _ev_save_json(filename: str, data: Any, evidence_dir):
    _ev_ensure_dirs(evidence_dir, _OUTPUT_SOURCE_DIR)
    path = Path(evidence_dir) / filename
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVE] {path.resolve()}")


def _ev_safe_str(x) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)


def _ev_copy_source_files(file_names, source_dir, output_source_dir):
    _ev_ensure_dirs(_EVIDENCE_DIR, output_source_dir)
    copied = set()
    for name in file_names:
        if not name:
            continue
        name = str(name)
        if name in copied:
            continue
        src = Path(source_dir) / name
        dst = Path(output_source_dir) / name
        if src.exists():
            try:
                shutil.copy2(src, dst)
                print(f"[COPY] {name}")
                copied.add(name)
            except Exception as e:
                print(f"[WARN] copy failed {name}: {e}")
        else:
            print(f"[WARN] source not found: {src}")
    print(f"[INFO] Copied {len(copied)} files -> {Path(output_source_dir).resolve()}")


def _ev_extract_source_file_from_chunk_id(chunk_id: Optional[str]) -> Optional[str]:
    if chunk_id is None:
        return None
    cid = str(chunk_id)
    if "::" in cid:
        return cid.split("::")[0]
    return None


class CallTreeProfiler:
    def __init__(self, module_keyword="fast_graphrag", max_depth=80):
        self.module_keyword = module_keyword
        self.max_depth = max_depth
        self.depth = 0
        self.enabled = False

    def _want(self, frame):
        fn = frame.f_code.co_filename
        return self.module_keyword in fn

    def _fmt(self, frame):
        code = frame.f_code
        func = code.co_name
        file = code.co_filename.replace("\\", "/").split("/")[-1]
        line = frame.f_lineno
        return f"{file}:{line} :: {func}()"

    def tracer(self, frame, event, arg):
        if not self.enabled:
            return self.tracer
        if event == "call":
            if self.depth < self.max_depth and self._want(frame):
                print("  " * self.depth + "-> " + self._fmt(frame))
                self.depth += 1
        elif event == "return":
            if self.depth > 0 and self._want(frame):
                self.depth -= 1
        return self.tracer

    def __enter__(self):
        self.enabled = True
        sys.setprofile(self.tracer)
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.setprofile(None)
        self.enabled = False


async def _ev_init_query_environment(grag: GraphRAG):
    await grag.state_manager.query_start()
    await grag.state_manager.query_done()


def _ev_dump_context(context, topk=SHOW_TOPK):
    print("\n[GraphRAG CONTEXT] Top Entities")
    for i, (e, s) in enumerate(context.entities[:topk], 1):
        name = getattr(e, "name", _ev_safe_str(e))
        desc = getattr(e, "description", "")
        desc = desc.replace("\n", " ")[:300] if desc else ""
        print(f"  {i:02d}. score={float(s):.4f} | {name} | {desc}")
    print("\n[GraphRAG CONTEXT] Top Relations")
    for i, (r, s) in enumerate(context.relations[:topk], 1):
        src = getattr(r, "source", "?")
        tgt = getattr(r, "target", "?")
        desc = getattr(r, "description", "")
        desc = desc.replace("\n", " ")[:400] if desc else ""
        print(f"  {i:02d}. score={float(s):.4f} | {src} -> {tgt} | {desc}")
    print("\n[GraphRAG CONTEXT] Top Chunks")
    for i, (c, s) in enumerate(context.chunks[:topk], 1):
        cid = getattr(c, "id", None)
        content = _ev_safe_str(c).replace("\n", " ")
        snippet = content[:SHOW_CHUNK_SNIPPET_CHARS]
        print(f"  {i:02d}. score={float(s):.4f} | chunk_id={cid}")
        print(f"      {snippet}")


def _ev_context_to_text(context, topk=SHOW_TOPK) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("GraphRAG Evidence Only")
    lines.append("=" * 80)
    lines.append("\n[Top Entities]")
    for i, (e, s) in enumerate(context.entities[:topk], 1):
        name = getattr(e, "name", _ev_safe_str(e))
        desc = getattr(e, "description", "")
        desc = desc.replace("\n", " ") if desc else ""
        lines.append(f"{i:02d}. score={float(s):.6f}")
        lines.append(f"    name: {name}")
        lines.append(f"    description: {desc}")
    lines.append("\n[Top Relations]")
    for i, (r, s) in enumerate(context.relations[:topk], 1):
        src = getattr(r, "source", "?")
        tgt = getattr(r, "target", "?")
        desc = getattr(r, "description", "")
        desc = desc.replace("\n", " ") if desc else ""
        lines.append(f"{i:02d}. score={float(s):.6f}")
        lines.append(f"    source: {src}")
        lines.append(f"    target: {tgt}")
        lines.append(f"    description: {desc}")
    lines.append("\n[Top Chunks]")
    for i, (c, s) in enumerate(context.chunks[:topk], 1):
        cid = getattr(c, "id", None)
        content = _ev_safe_str(c)
        lines.append("\n" + "-" * 80)
        lines.append(f"[GraphRAG Chunk Evidence {i}]")
        lines.append(f"score: {float(s):.6f}")
        lines.append(f"chunk_id: {cid}")
        lines.append(f"source_file: {_ev_extract_source_file_from_chunk_id(cid)}")
        lines.append("-" * 80)
        lines.append(content)
    return "\n".join(lines)


def _ev_context_to_json(context, topk=SHOW_TOPK) -> Dict[str, Any]:
    data = {"entities": [], "relations": [], "chunks": []}
    for e, s in context.entities[:topk]:
        data["entities"].append({
            "score": float(s),
            "name": getattr(e, "name", _ev_safe_str(e)),
            "description": getattr(e, "description", ""),
        })
    for r, s in context.relations[:topk]:
        data["relations"].append({
            "score": float(s),
            "source": getattr(r, "source", "?"),
            "target": getattr(r, "target", "?"),
            "description": getattr(r, "description", ""),
        })
    for c, s in context.chunks[:topk]:
        cid = getattr(c, "id", None)
        data["chunks"].append({
            "score": float(s),
            "chunk_id": str(cid),
            "source_file": _ev_extract_source_file_from_chunk_id(cid),
            "content": _ev_safe_str(c),
        })
    return data


def _ev_patch_graphrag_for_debug(grag: GraphRAG):
    if not ENABLE_GRAPHRAG_STAGE_LOGS:
        return
    try:
        ie = grag.information_extraction_service
        orig_extract_entities = ie.extract_entities_from_query

        @wraps(orig_extract_entities)
        async def wrapped_extract_entities_from_query(*args, **kwargs):
            q = kwargs.get("query", None)
            print("\n" + "=" * 80)
            print("[GraphRAG STEP 1] extract_entities_from_query()")
            print(f"query = {q!r}")
            out = await orig_extract_entities(*args, **kwargs)
            print("extracted_entities =", out)
            print("=" * 80)
            return out

        ie.extract_entities_from_query = wrapped_extract_entities_from_query
    except Exception as e:
        print("[WARN] patch extract_entities_from_query failed:", e)

    try:
        sm = grag.state_manager
        orig_get_context = sm.get_context

        @wraps(orig_get_context)
        async def wrapped_get_context(*args, **kwargs):
            q = kwargs.get("query", None)
            ents = kwargs.get("entities", None)
            print("\n" + "=" * 80)
            print("[GraphRAG STEP 2] state_manager.get_context()")
            print(f"query = {q!r}")
            print(f"entities(seed) = {ents}")
            context = await orig_get_context(*args, **kwargs)
            if context is None:
                print("context = None")
            else:
                print("context fetched")
                _ev_dump_context(context)
            print("=" * 80)
            return context

        sm.get_context = wrapped_get_context
    except Exception as e:
        print("[WARN] patch get_context failed:", e)

    try:
        from fast_graphrag._types import TContext
        orig_truncate = TContext.truncate

        @wraps(orig_truncate)
        def wrapped_truncate(self, max_chars, output_context_str=False):
            print("\n" + "=" * 80)
            print("[GraphRAG STEP 3] context.truncate()")
            print("max_chars =", max_chars)
            print("output_context_str =", output_context_str)
            s = orig_truncate(self, max_chars=max_chars, output_context_str=output_context_str)
            print(f"truncated context string length = {len(s)} chars")
            preview = s[:800].replace("\n", "\\n")
            print("preview =", preview + ("..." if len(s) > 800 else ""))
            print("=" * 80)
            return s

        TContext.truncate = wrapped_truncate
    except Exception as e:
        print("[WARN] patch TContext.truncate failed:", e)


async def retrieve_evidence_graphrag(
    query: str,
    graphrag_working_dir=None,
    evidence_dir=None,
    output_source_dir=None,
    source_dir=None,
):
    if graphrag_working_dir is None:
        graphrag_working_dir = _GRAPHRAG_WORKING_DIR
    if evidence_dir is None:
        evidence_dir = _EVIDENCE_DIR
    if output_source_dir is None:
        output_source_dir = _OUTPUT_SOURCE_DIR
    if source_dir is None:
        source_dir = _SOURCE_DIR

    graphrag_working_dir = Path(graphrag_working_dir)
    evidence_dir = Path(evidence_dir)
    output_source_dir = Path(output_source_dir)
    source_dir = Path(source_dir)

    if not graphrag_working_dir.exists():
        raise RuntimeError(f"GraphRAG DB not found: {graphrag_working_dir.resolve()}")

    logger.setLevel("INFO")

    grag = GraphRAG(
        working_dir=str(graphrag_working_dir),
        domain=DOMAIN,
        example_queries=EXAMPLE_QUERIES,
        entity_types=ENTITY_TYPES,
        config=GraphRAG.Config(
            llm_service=OpenAILLMService(
                model=LLM_MODEL_FOR_GRAPHRAG,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
            ),
            embedding_service=_PatchedEmbeddingService(
                model=EMBED_MODEL,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                embedding_dim=1536,
                max_elements_per_request=2,
                max_requests_concurrent=3,
                max_requests_per_minute=200,
            ),
        ),
    )

    await _ev_init_query_environment(grag)
    _ev_patch_graphrag_for_debug(grag)

    params = QueryParam(
        with_references=False,
        only_context=True,
        entities_max_tokens=4000,
        relations_max_tokens=3000,
        chunks_max_tokens=9000,
    )

    print("\n" + "#" * 80)
    print("GraphRAG Evidence Retrieval")
    print("#" * 80)
    print("Query:")
    print(query)

    # Retry up to 3 times with backoff for embedding API errors
    for attempt in range(1, 4):
        try:
            if ENABLE_PROFILE_CALLTREE:
                with CallTreeProfiler(module_keyword="fast_graphrag", max_depth=80):
                    result = await grag.async_query(query, params=params)
            else:
                result = await grag.async_query(query, params=params)
            break
        except Exception as e:
            if attempt < 3:
                wait = 30 * attempt
                print(f"  GraphRAG query retry {attempt}/3 after {wait}s (API overload): {e}")
                await asyncio.sleep(wait)
            else:
                raise

    if not hasattr(result, "context") or result.context is None:
        print("[WARN] GraphRAG result has no context.")
        return None

    context = result.context
    evidence_text = _ev_context_to_text(context, topk=SHOW_TOPK)
    evidence_json = _ev_context_to_json(context, topk=SHOW_TOPK)
    _ev_save_text("graphrag_evidence.txt", evidence_text, evidence_dir)
    _ev_save_json("graphrag_evidence.json", evidence_json, evidence_dir)

    source_files = set()
    for chunk, score in context.chunks:
        cid = getattr(chunk, "id", None)
        fname = _ev_extract_source_file_from_chunk_id(cid)
        if fname:
            source_files.add(fname)
    _ev_copy_source_files(source_files, source_dir, output_source_dir)

    print("\n[GraphRAG Evidence Finished]")
    print(f"entities: {len(context.entities)}")
    print(f"relations: {len(context.relations)}")
    print(f"chunks: {len(context.chunks)}")
    return context


def _ev_info(msg: str):
    print(f"[INFO] {msg}")


def _ev_warn(msg: str):
    print(f"[WARN] {msg}")


def _ev_ok(msg: str):
    print(f"[ OK ] {msg}")


def _ev_die(msg: str, code: int = 1):
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _ev_read_pdf(path: str) -> str:
    reader = pypdf.PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(f"\n\n[PAGE {i + 1}]\n{text}")
    return "\n".join(pages)


def _ev_read_txt(path: str) -> str:
    p = Path(path)
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return p.read_text(encoding="utf-8", errors="ignore")


def _ev_chunk_text(text: str, chunk_size=1200, overlap=200) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text)
    chunks = []
    start = 0
    n = len(text)
    if n == 0:
        return []
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


class OpenAIChromaEmbedding(EmbeddingFunction):
    def __call__(self, texts: List[str]) -> List[List[float]]:
        cleaned = [t if isinstance(t, str) else str(t) for t in texts]
        cleaned = [t if t.strip() else " " for t in cleaned]

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                resp = client_openai.embeddings.create(model=EMBED_MODEL, input=cleaned)
                return [d.embedding for d in resp.data]
            except Exception as e:
                if attempt < max_retries:
                    wait = 20 * attempt
                    print(f"  [Embedding] retry {attempt}/{max_retries} after {wait}s "
                          f"(API overload): {e}")
                    time.sleep(wait)
                else:
                    raise


def build_chroma_index(source_dir=None, chroma_db_dir=None):
    if source_dir is None:
        source_dir = _SOURCE_DIR
    if chroma_db_dir is None:
        chroma_db_dir = _CHROMA_DB_DIR
    source_dir = Path(source_dir)
    chroma_db_dir = Path(chroma_db_dir)

    if not source_dir.exists():
        _ev_die(f"Source dir not found: {source_dir.resolve()}")

    chroma_db_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(source_dir / "**" / "*.*"), recursive=True))
    files = [f for f in files if Path(f).suffix.lower() in [".pdf", ".txt"]]

    if not files:
        _ev_info(f"No .pdf/.txt files in {source_dir}, skipping Chroma build")
        return

    _ev_info(f"Found {len(files)} input files")
    _ev_info("Connecting to Chroma")

    chroma = chromadb.PersistentClient(path=str(chroma_db_dir))
    col = chroma.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=OpenAIChromaEmbedding(),
    )

    existing_ids = set()
    try:
        existing = col.get(include=[])
        existing_ids = set(existing.get("ids", []))
        _ev_ok(f"Existing chunks: {len(existing_ids)}")
    except Exception as e:
        _ev_warn(f"Reading existing ids failed: {e}")

    _ev_info("Start writing to Chroma")
    total_added = 0

    for f in files:
        ext = Path(f).suffix.lower()
        doc_id_base = Path(f).name
        _ev_info(f"Processing {doc_id_base}")

        try:
            text = _ev_read_pdf(f) if ext == ".pdf" else _ev_read_txt(f)
        except Exception as e:
            _ev_warn(f"Read failed, skip {doc_id_base}: {e}")
            continue

        if not text or not text.strip():
            _ev_warn(f"Empty text, skip {doc_id_base}")
            continue

        chunks = _ev_chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        if not chunks:
            _ev_warn(f"No chunks, skip {doc_id_base}")
            continue

        ids, docs, metas = [], [], []
        for idx, ch in enumerate(chunks):
            cid = f"{doc_id_base}::chunk{idx}"
            if cid in existing_ids:
                continue
            if len(ch.strip()) < 30:
                continue
            ids.append(cid)
            docs.append(ch)
            metas.append({"source_file": doc_id_base, "chunk_index": idx})

        if not ids:
            _ev_warn(f"No new chunks: {doc_id_base}")
            continue

        batch_size = 64
        for s in range(0, len(ids), batch_size):
            batch_ids = ids[s:s + batch_size]
            batch_docs = docs[s:s + batch_size]
            batch_metas = metas[s:s + batch_size]
            try:
                col.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                _ev_ok(f"Written {doc_id_base} batch {s // batch_size} | +{len(batch_ids)}")
                total_added += len(batch_ids)
                # Rate-limit pause to avoid embedding API errors
                if s + batch_size < len(ids):
                    time.sleep(3)
            except Exception as e:
                _ev_warn(f"Write failed: {doc_id_base} batch {s // batch_size} | {e}")
                continue

    _ev_ok(f"Total new chunks: {total_added}")
    _ev_ok("Chroma build finished")


def retrieve_evidence_chroma(
    question: str,
    k=20,
    chroma_db_dir=None,
    evidence_dir=None,
    output_source_dir=None,
    source_dir=None,
) -> Dict[str, Any]:
    if chroma_db_dir is None:
        chroma_db_dir = _CHROMA_DB_DIR
    if evidence_dir is None:
        evidence_dir = _EVIDENCE_DIR
    if output_source_dir is None:
        output_source_dir = _OUTPUT_SOURCE_DIR
    if source_dir is None:
        source_dir = _SOURCE_DIR

    chroma_db_dir = Path(chroma_db_dir)
    evidence_dir = Path(evidence_dir)
    output_source_dir = Path(output_source_dir)
    source_dir = Path(source_dir)

    if not chroma_db_dir.exists():
        raise RuntimeError(f"Chroma DB not found: {chroma_db_dir.resolve()}")

    chroma = chromadb.PersistentClient(path=str(chroma_db_dir))
    col = chroma.get_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=OpenAIChromaEmbedding(),
    )

    res = col.query(
        query_texts=[question],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]

    evidence_list = []
    source_files = set()

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        source_file = meta.get("source_file")
        chunk_index = meta.get("chunk_index")
        if source_file:
            source_files.add(source_file)
        evidence_list.append({
            "rank": i,
            "source_file": source_file,
            "chunk_index": chunk_index,
            "distance": float(dist),
            "content": doc,
        })

    _ev_copy_source_files(source_files, source_dir, output_source_dir)

    data = {"query": question, "top_k": k, "evidence": evidence_list}
    return data


def chroma_evidence_to_text(data: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("Chroma RAG Evidence Only")
    lines.append("=" * 80)
    lines.append(f"Query: {data.get('query')}")
    lines.append(f"Top K: {data.get('top_k')}")
    if data.get("error"):
        lines.append(f"\nERROR: {data['error']}")
        return "\n".join(lines)
    for item in data.get("evidence", []):
        lines.append("\n" + "-" * 80)
        lines.append(f"[Chroma Chunk Evidence {item['rank']}]")
        lines.append(f"source_file: {item['source_file']}")
        lines.append(f"chunk_index: {item['chunk_index']}")
        lines.append(f"distance: {item['distance']:.6f}")
        lines.append("-" * 80)
        lines.append(item["content"])
    return "\n".join(lines)


async def main():
    # Step 0: Read query3 from query_results.json
    query_results_path = PROJECT_DIR / "query_results.json"
    with open(query_results_path, "r", encoding="utf-8") as f:
        query_data = json.load(f)
    query3 = query_data.get("query3", "")
    if not query3:
        raise ValueError(f"query3 not found in {query_results_path}")

    print(f"[Step 0] query3: {query3[:200]}...")

    # Step 1: Decompose query3 into sub-questions
    sub_questions = decompose_question_to_mechanistic_questions(query3)
    print(f"\n[Step 1] Decomposed into {len(sub_questions)} sub-questions:")
    for i, q in enumerate(sub_questions, 1):
        print(f"  Q{i}: {q}")

    # Step 2: Index papers (MinerU -> collect TXT -> GraphRAG index)
    print(f"\n[Step 2] Indexing papers...")
    print(f"  PDF_DIR: {PDF_DIR}")
    print(f"  TXT_DIR: {TXT_DIR}")
    print(f"  GRAPH_DB_DIR: {GRAPH_DB_DIR}")

    if list(PDF_DIR.glob("*.pdf")):
        if (GRAPH_DB_DIR / "entities.parquet").exists():
            print("[Step 2] GraphRAG DB already exists, skipping PDF conversion and indexing")
        else:
            mineru_pdf_to_txt(PDF_DIR, MINERU_OUT_DIR)
            collect_txt_outputs(MINERU_OUT_DIR, TXT_DIR)
            await graphrag_index_main(GRAPH_DB_DIR, TXT_DIR)
    else:
        print("[Step 2] No PDFs in Papers/, checking for existing TXT files...")
        if list(TXT_DIR.glob("*.txt")):
            print(f"[Step 2] Found existing TXT files in {TXT_DIR}")
            if not (GRAPH_DB_DIR / "entities.parquet").exists():
                print("[Step 2] GraphRAG DB not found, building index from existing TXT...")
                await graphrag_index_main(GRAPH_DB_DIR, TXT_DIR)
            else:
                print("[Step 2] GraphRAG DB already exists, skipping indexing")
        else:
            print("[Step 2] No TXT files found either. Please place PDFs in Papers/ or TXTs in Converted txt/")
            print("[Step 2] Skipping indexing for now, proceeding to evidence retrieval...")

    # Step 3: Retrieve evidence for each sub-question
    print(f"\n[Step 3] Retrieving evidence for {len(sub_questions)} sub-questions...")
    build_chroma_flag = True

    for i, question in enumerate(sub_questions, 1):
        evidence_dir = PROJECT_DIR / f"evidence_q{i}"
        sources_dir = PROJECT_DIR / f"sources_q{i}"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        sources_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Q{i}: {question[:120]}...")
        print(f"  Evidence -> {evidence_dir}")
        print(f"  Sources  -> {sources_dir}")
        print(f"{'='*60}")

        # GraphRAG
        if _RUN_GRAPHRAG:
            try:
                await retrieve_evidence_graphrag(
                    query=question,
                    graphrag_working_dir=GRAPH_DB_DIR,
                    evidence_dir=evidence_dir,
                    output_source_dir=sources_dir,
                    source_dir=TXT_DIR,
                )
            except Exception:
                logger.exception(f"GraphRAG retrieval failed for Q{i}", exc_info=True)

        # Chroma build (first time only)
        if build_chroma_flag:
            try:
                build_chroma_index(source_dir=TXT_DIR, chroma_db_dir=CHROMA_DB_DIR)
                build_chroma_flag = False
            except Exception as e:
                print(f"[ERROR] Chroma build failed: {e}")

        # Chroma retrieval
        if _RUN_CHROMA:
            try:
                chroma_data = retrieve_evidence_chroma(
                    question=question,
                    k=SHOW_TOPK,
                    chroma_db_dir=CHROMA_DB_DIR,
                    evidence_dir=evidence_dir,
                    output_source_dir=sources_dir,
                    source_dir=TXT_DIR,
                )
                chroma_text = chroma_evidence_to_text(chroma_data)
                _ev_save_text("chroma_evidence.txt", chroma_text, evidence_dir)
                _ev_save_json("chroma_evidence.json", chroma_data, evidence_dir)
                print(f"\n[Chroma Evidence Finished] chunks: {len(chroma_data.get('evidence', []))}")
            except Exception as e:
                print(f"[ERROR] Chroma retrieval failed: {e}")

    print("\n" + "=" * 80)
    print("All done.")
    print("=" * 80)
    for i in range(1, len(sub_questions) + 1):
        print(f"  Q{i}: {PROJECT_DIR / f'evidence_q{i}'}")


def run(state: dict) -> dict:
    """
    LangGraph node: Searching Agent.
    Performs Hybrid RAG (GraphRAG + Chroma) evidence retrieval.
    Reads query3 from query_results.json, indexes papers from Papers/ via MinerU,
    retrieves evidence for each sub-question into evidence_qN/ directories.
    Always reads from Papers/ — pre-store PDFs there or let Mining agent fill it.
    Updates state with 'evidence_dirs'.
    """
    asyncio.run(main())

    # Collect evidence directories
    evidence_dirs = []
    for d in sorted(PROJECT_DIR.glob("evidence_q*")):
        if d.is_dir():
            evidence_dirs.append(str(d))
    state["evidence_dirs"] = evidence_dirs
    return state


if __name__ == "__main__":
    asyncio.run(main())
