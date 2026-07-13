import os
import re
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from openai import AsyncOpenAI


# ============================================================
# 0. Config
# ============================================================

API_KEY = os.getenv("AGICTO_API_KEY")
BASE_URL = os.getenv("AGICTO_BASE_URL", "https://api.agicto.cn/v1")
MODEL = os.getenv("LLM_MODEL_MAIN", "gpt-5")
MODEL_DECOMPOSE = "gpt-4o-mini"

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "writing_output"


# ============================================================
# 0b. Question decomposition (from final-转换问法.py)
# ============================================================

def _to_snake_case(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def _parse_llm_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^```", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def _decompose_query(query3: str) -> List[str]:
    """Decompose query3 into sub-questions using agicto API."""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    prompt = f"""
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
{query3}
"""

    response = client.chat.completions.create(
        model=MODEL_DECOMPOSE,
        messages=[{"role": "system", "content": "You are a scientific text understanding assistant. Return only the requested output."},
                  {"role": "user", "content": prompt}],
        temperature=0.0,
    )

    data = _parse_llm_json(response.choices[0].message.content)
    relationships = data.get("relationships", [])
    if not relationships:
        raise ValueError("LLM did not extract any relationships.")

    questions = []
    for item in relationships:
        factor = _to_snake_case(item["factor"])
        response_pattern = item["response_pattern"].strip()
        outcome = item["outcome"].strip()
        questions.append(
            f"Why does {factor} follow a {response_pattern} response, "
            f"controlling {outcome}?"
        )
    return questions
# ============================================================
# 1. Prompt: step 1 - extract generalized mechanisms
# ============================================================

PROMPT_EXTRACT_GENERAL_MECHANISMS = """You are a scientific evidence abstraction assistant.

# INPUT EVIDENCE
{context}

# TASK
Extract general, transferable scientific conclusions from the retrieved evidence.

Do not answer the user query yet.
Do not rewrite or modify the evidence.
Do not treat the evidence system as the user's system.

Your task is to infer mechanism-level conclusions that are supported by the evidence but are not tied to evidence-specific materials, names, devices, concentrations, geometries, or experimental settings.

# EXTRACTION RULES

For each useful piece of evidence, extract:

1. Evidence-specific observation:
   A concise statement of what the evidence specifically reports.

2. Generalized principle:
   A more general scientific principle that can be inferred from the evidence.

3. Transferable mechanism:
   A mechanism-level statement that may apply to other related systems.

4. Applicability condition:
   The condition under which this mechanism can be transferred to another system.

5. Uncertainty or limitation:
   What cannot be assumed when transferring this mechanism.

# IMPORTANT RULES

- Do not delete or alter the meaning of the evidence.
- Do not claim that evidence-specific materials exist in another system.
- Do not preserve evidence-specific names in the generalized principle unless they are scientifically necessary.
- Prefer functional and mechanistic language over material-specific language.
- If a conclusion is only weakly supported, mark it as plausible rather than established.
- If the evidence is too specific to generalize, say so.
- Focus on transferable scientific mechanisms, not literature-summary details.
- Avoid using internal retrieval labels, chunk IDs, entity IDs, relation IDs, or file names in the final abstraction.

# OUTPUT FORMAT

Return the result as bullet points using this structure:

- Evidence-specific observation:
- Generalized principle:
- Transferable mechanism:
- Applicability condition:
- Uncertainty / limitation:

Generalized mechanisms:
"""


# ============================================================
# 2. Prompt: step 2 - apply generalized mechanisms to query
# ============================================================

PROMPT_APPLY_GENERAL_MECHANISMS = """You are a scientific reasoning assistant.

# GENERALIZED MECHANISMS EXTRACTED FROM EVIDENCE
{generalized_mechanisms}

# USER QUERY
{query}

# USER'S CURRENT STUDY
The user's current study concerns the behavior and performance of magnetic nanoparticle clusters crossing a water-oil interface.

The controlled variables include:
- aqueous-phase ion (NaCl) concentration
- aqueous-phase nonionic surfactant type/concentration
- oil type
- oil-phase surfactant type/concentration

# TASK
Use the generalized mechanisms extracted from the retrieved evidence to answer the user query in the context of the user's study.

# REASONING RULES

- Do not summarize the original evidence.
- Do not mention evidence-specific systems, materials, chemical names, oil fractions, devices, concentrations, or experimental settings unless they are explicitly part of the user query.
- Use the generalized principles and transferable mechanisms as the basis for reasoning.
- You may make reasonable mechanism-based inferences, but mark them as plausible if not proven.
- Focus on the user's experimental system.
- If multiple mechanisms are possible, organize them clearly.

# OUTPUT REQUIREMENTS

- Answer the query directly.
- Be concise, mechanistic, and hypothesis-oriented.
- Do not include internal retrieval labels such as G-R8, G-Source 3, C-Source 2, chunk IDs, entity IDs, relation IDs, or file names.
- Do not write a literature-summary paragraph.
- Do not present evidence-specific assumptions as facts about the user's study.

Answer:
"""


# ============================================================
# 4. 通用读取函数
# ============================================================

def load_json_if_exists(path: Path) -> Optional[Union[Dict[str, Any], List[Any]]]:
    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def load_txt_if_exists(path: Path) -> Optional[str]:
    if not path.exists():
        return None

    return path.read_text(encoding="utf-8")


# ============================================================
# 5. GraphRAG evidence -> context
# ============================================================

def graphrag_evidence_to_context(
    evidence: Dict[str, Any],
    topk_entities: int = 20,
    topk_relations: int = 20,
    topk_chunks: int = 20,
    max_chunk_chars: Optional[int] = None,
) -> str:
    lines: List[str] = []

    lines.append("===== GraphRAG Entities =====")
    lines.append("id | entity | score | description")

    entities = evidence.get("entities", [])
    if entities:
        for i, e in enumerate(entities[:topk_entities], start=1):
            name = e.get("name", "")
            score = e.get("score", "")
            desc = e.get("description", "")
            lines.append(f"G-E{i} | {name} | {score} | {desc}")
    else:
        lines.append("No GraphRAG entities found.")

    lines.append("\n===== GraphRAG Relationships =====")
    lines.append("id | source | target | score | description")

    relations = evidence.get("relations", [])
    if relations:
        for i, r in enumerate(relations[:topk_relations], start=1):
            source = r.get("source", "")
            target = r.get("target", "")
            score = r.get("score", "")
            desc = r.get("description", "")
            lines.append(f"G-R{i} | {source} | {target} | {score} | {desc}")
    else:
        lines.append("No GraphRAG relationships found.")

    lines.append("\n===== GraphRAG Sources =====")

    chunks = evidence.get("chunks", [])
    if chunks:
        for i, c in enumerate(chunks[:topk_chunks], start=1):
            source_file = c.get("source_file", "")
            chunk_id = c.get("chunk_id", "")
            score = c.get("score", "")
            content = c.get("content", "")

            if max_chunk_chars is not None and len(content) > max_chunk_chars:
                content = content[:max_chunk_chars] + "\n...[TRUNCATED]"

            lines.append(f"\n[G-Source {i}]")
            lines.append(f"source_file: {source_file}")
            lines.append(f"chunk_id: {chunk_id}")
            lines.append(f"score: {score}")
            lines.append("content:")
            lines.append(content)
    else:
        lines.append("No GraphRAG source chunks found.")

    return "\n".join(lines)


# ============================================================
# 6. Chroma evidence -> context
# ============================================================

def extract_text_from_chroma_item(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        content = (
            item.get("content")
            or item.get("text")
            or item.get("document")
            or item.get("page_content")
            or ""
        )

        metadata = item.get("metadata", {}) or {}

        source_file = (
            item.get("source_file")
            or metadata.get("source_file")
            or metadata.get("source")
            or metadata.get("file")
            or ""
        )

        chunk_id = (
            item.get("chunk_id")
            or metadata.get("chunk_id")
            or metadata.get("id")
            or ""
        )

        score = (
            item.get("score")
            or item.get("similarity")
            or item.get("distance")
            or ""
        )

        return {
            "source_file": source_file,
            "chunk_id": chunk_id,
            "score": score,
            "content": content,
            "metadata": metadata,
        }

    return {
        "source_file": "",
        "chunk_id": "",
        "score": "",
        "content": str(item),
        "metadata": {},
    }


def normalize_chroma_evidence(
    evidence: Union[Dict[str, Any], List[Any]]
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []

    if isinstance(evidence, list):
        for item in evidence:
            chunks.append(extract_text_from_chroma_item(item))
        return chunks

    if not isinstance(evidence, dict):
        return chunks

    for key in ["chunks", "sources", "results", "documents"]:
        value = evidence.get(key)

        if isinstance(value, list):
            if key == "documents" and value and isinstance(value[0], list):
                docs = evidence.get("documents", [[]])[0]
                metadatas = evidence.get("metadatas", [[]])[0]
                distances = evidence.get("distances", [[]])[0]

                for i, doc in enumerate(docs):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    distance = distances[i] if i < len(distances) else ""

                    chunks.append({
                        "source_file": metadata.get("source_file", metadata.get("source", "")),
                        "chunk_id": metadata.get("chunk_id", metadata.get("id", "")),
                        "score": distance,
                        "content": doc,
                        "metadata": metadata,
                    })

                return chunks

            for item in value:
                chunks.append(extract_text_from_chroma_item(item))

            return chunks

    if any(k in evidence for k in ["content", "text", "document", "page_content"]):
        chunks.append(extract_text_from_chroma_item(evidence))

    return chunks


def chroma_evidence_to_context(
    evidence: Union[Dict[str, Any], List[Any]],
    topk_chunks: int = 20,
    max_chunk_chars: Optional[int] = None,
) -> str:
    lines: List[str] = []

    lines.append("\n===== Chroma / Vector RAG Sources =====")

    chunks = normalize_chroma_evidence(evidence)

    if not chunks:
        lines.append("No Chroma source chunks found.")
        return "\n".join(lines)

    for i, c in enumerate(chunks[:topk_chunks], start=1):
        source_file = c.get("source_file", "")
        chunk_id = c.get("chunk_id", "")
        score = c.get("score", "")
        content = c.get("content", "")

        if max_chunk_chars is not None and len(content) > max_chunk_chars:
            content = content[:max_chunk_chars] + "\n...[TRUNCATED]"

        lines.append(f"\n[C-Source {i}]")
        lines.append(f"source_file: {source_file}")
        lines.append(f"chunk_id: {chunk_id}")
        lines.append(f"score_or_distance: {score}")
        lines.append("content:")
        lines.append(content)

    return "\n".join(lines)


# ============================================================
# 7. fallback: txt evidence -> context
# ============================================================

def txt_evidence_to_context(
    text: str,
    title: str,
    max_chars: Optional[int] = None,
) -> str:
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars] + "\n...[TRUNCATED]"

    return f"\n===== {title} =====\n{text}"


# ============================================================
# 8. 构造合并 context
# ============================================================

def build_combined_context(
    graphrag_json_path: Path,
    chroma_json_path: Path,
    graphrag_txt_path: Optional[Path] = None,
    chroma_txt_path: Optional[Path] = None,
    topk_entities: int = 20,
    topk_relations: int = 20,
    topk_graphrag_chunks: int = 20,
    topk_chroma_chunks: int = 20,
    max_chunk_chars: Optional[int] = None,
) -> str:
    context_parts: List[str] = []

    graphrag_json = load_json_if_exists(graphrag_json_path)

    if isinstance(graphrag_json, dict):
        graphrag_context = graphrag_evidence_to_context(
            evidence=graphrag_json,
            topk_entities=topk_entities,
            topk_relations=topk_relations,
            topk_chunks=topk_graphrag_chunks,
            max_chunk_chars=max_chunk_chars,
        )
        context_parts.append(graphrag_context)
    else:
        if graphrag_txt_path is not None:
            graphrag_txt = load_txt_if_exists(graphrag_txt_path)
            if graphrag_txt:
                context_parts.append(
                    txt_evidence_to_context(
                        graphrag_txt,
                        title="GraphRAG Evidence TXT",
                        max_chars=None,
                    )
                )
            else:
                context_parts.append("===== GraphRAG Evidence =====\nNo GraphRAG evidence found.")
        else:
            context_parts.append("===== GraphRAG Evidence =====\nNo GraphRAG evidence found.")

    chroma_json = load_json_if_exists(chroma_json_path)

    if chroma_json is not None:
        chroma_context = chroma_evidence_to_context(
            evidence=chroma_json,
            topk_chunks=topk_chroma_chunks,
            max_chunk_chars=max_chunk_chars,
        )
        context_parts.append(chroma_context)
    else:
        if chroma_txt_path is not None:
            chroma_txt = load_txt_if_exists(chroma_txt_path)
            if chroma_txt:
                context_parts.append(
                    txt_evidence_to_context(
                        chroma_txt,
                        title="Chroma / Vector RAG Evidence TXT",
                        max_chars=None,
                    )
                )
            else:
                context_parts.append("===== Chroma Evidence =====\nNo Chroma evidence found.")
        else:
            context_parts.append("===== Chroma Evidence =====\nNo Chroma evidence found.")

    return "\n\n".join(context_parts)


# ============================================================
# 9. 构造 prompt：原始单步回答，保留
# ============================================================

def build_prompt(
    query: str,
    context: str,
    with_references: bool = False,
) -> str:
    return PROMPT_NO_REFERENCES.format(
        query=query,
        context=context,
    )


# ============================================================
# 10. 调用 LLM
# ============================================================

async def call_llm(
    prompt: str,
    model: str = MODEL,
    temperature: float = 0.0,
) -> str:
    if not API_KEY:
        raise RuntimeError(
            "API key not found. Please set AGICTO_API_KEY as an environment variable."
        )

    client = AsyncOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=temperature,
    )

    answer = response.choices[0].message.content

    if answer is None:
        return ""

    return answer.strip()


# ============================================================
# 11. 新增：从原始 evidence 抽取通用机制
# ============================================================

async def extract_generalized_mechanisms(
    context: str,
) -> str:
    prompt = PROMPT_EXTRACT_GENERAL_MECHANISMS.format(
        context=context,
    )

    mechanisms = await call_llm(
        prompt=prompt,
        model=MODEL,
        temperature=0.0,
    )

    return mechanisms.strip()


# ============================================================
# 12. 新增：基于通用机制回答 query
# ============================================================

async def answer_from_generalized_mechanisms(
    query: str,
    generalized_mechanisms: str,
) -> str:
    prompt = PROMPT_APPLY_GENERAL_MECHANISMS.format(
        query=query,
        generalized_mechanisms=generalized_mechanisms,
    )

    answer = await call_llm(
        prompt=prompt,
        model=MODEL,
        temperature=0.0,
    )

    return answer.strip()


# ============================================================
# 13. 原始主流程：单步 evidence -> answer，保留用于对比
# ============================================================

async def generate_answer_from_combined_evidence(
    query: str,
    graphrag_json_path: Path,
    chroma_json_path: Path,
    graphrag_txt_path: Optional[Path] = None,
    chroma_txt_path: Optional[Path] = None,
    with_references: bool = True,
):
    context = build_combined_context(
        graphrag_json_path=graphrag_json_path,
        chroma_json_path=chroma_json_path,
        graphrag_txt_path=graphrag_txt_path,
        chroma_txt_path=chroma_txt_path,
        topk_entities=20,
        topk_relations=20,
        topk_graphrag_chunks=20,
        topk_chroma_chunks=20,
        max_chunk_chars=None,
    )

    prompt = build_prompt(
        query=query,
        context=context,
        with_references=with_references,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context_path = OUTPUT_DIR / "combined_context_used.txt"
    prompt_path = OUTPUT_DIR / "combined_prompt_used.txt"
    answer_path = OUTPUT_DIR / "combined_llm_answer.txt"

    context_path.write_text(context, encoding="utf-8")
    prompt_path.write_text(prompt, encoding="utf-8")

    answer = await call_llm(
        prompt=prompt,
        model=MODEL,
        temperature=0.0,
    )

    answer_path.write_text(answer, encoding="utf-8")

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)

    print("\n[SAVE]")
    print(f"context: {context_path.resolve()}")
    print(f"prompt:  {prompt_path.resolve()}")
    print(f"answer:  {answer_path.resolve()}")

    return answer


# ============================================================
# 14. 新主流程：evidence -> generalized mechanisms -> answer
# ============================================================

async def generate_answer_via_generalized_mechanisms(
    query: str,
    graphrag_json_path: Path,
    chroma_json_path: Path,
    graphrag_txt_path: Optional[Path] = None,
    chroma_txt_path: Optional[Path] = None,
):
    # 1. 构造原始 combined context
    context = build_combined_context(
        graphrag_json_path=graphrag_json_path,
        chroma_json_path=chroma_json_path,
        graphrag_txt_path=graphrag_txt_path,
        chroma_txt_path=chroma_txt_path,
        topk_entities=20,
        topk_relations=20,
        topk_graphrag_chunks=20,
        topk_chroma_chunks=20,
        max_chunk_chars=None,
    )
    # context = build_combined_context(
    #     graphrag_json_path=graphrag_json_path,
    #     chroma_json_path=chroma_json_path,
    #     graphrag_txt_path=graphrag_txt_path,
    #     chroma_txt_path=chroma_txt_path,
    #     topk_entities=10,
    #     topk_relations=10,
    #     topk_graphrag_chunks=8,
    #     topk_chroma_chunks=8,
    #     max_chunk_chars=1200,
    # )
    # 2. 第一阶段：从 evidence 中抽取通用机制
    mechanism_extraction_prompt = PROMPT_EXTRACT_GENERAL_MECHANISMS.format(
        context=context,
    )

    generalized_mechanisms = await call_llm(
        prompt=mechanism_extraction_prompt,
        model=MODEL,
        temperature=0.0,
    )

    generalized_mechanisms = generalized_mechanisms.strip()

    # 3. 第二阶段：把通用机制应用到用户 query
    mechanism_application_prompt = PROMPT_APPLY_GENERAL_MECHANISMS.format(
        query=query,
        generalized_mechanisms=generalized_mechanisms,
    )

    answer = await call_llm(
        prompt=mechanism_application_prompt,
        model=MODEL,
        temperature=0.0,
    )

    answer = answer.strip()

    # 4. 保存所有中间文件
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context_path = OUTPUT_DIR / "combined_context_used.txt"
    mechanism_extraction_prompt_path = OUTPUT_DIR / "mechanism_extraction_prompt.txt"
    generalized_mechanisms_path = OUTPUT_DIR / "generalized_mechanisms.txt"
    mechanism_application_prompt_path = OUTPUT_DIR / "mechanism_application_prompt.txt"
    answer_path = OUTPUT_DIR / "final_answer_from_generalized_mechanisms.txt"

    context_path.write_text(context, encoding="utf-8")
    mechanism_extraction_prompt_path.write_text(mechanism_extraction_prompt, encoding="utf-8")
    generalized_mechanisms_path.write_text(generalized_mechanisms, encoding="utf-8")
    mechanism_application_prompt_path.write_text(mechanism_application_prompt, encoding="utf-8")
    answer_path.write_text(answer, encoding="utf-8")

    # 5. 打印结果
    print("\n" + "=" * 80)
    print("GENERALIZED MECHANISMS")
    print("=" * 80)
    print(generalized_mechanisms)

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(answer)

    print("\n[SAVE]")
    print(f"context:                       {context_path.resolve()}")
    print(f"mechanism extraction prompt:   {mechanism_extraction_prompt_path.resolve()}")
    print(f"generalized mechanisms:        {generalized_mechanisms_path.resolve()}")
    print(f"mechanism application prompt:  {mechanism_application_prompt_path.resolve()}")
    print(f"answer:                        {answer_path.resolve()}")

    return answer


# ============================================================
# 15. 运行
# ============================================================

async def main():
    # Read query3 from query_results.json
    query_results_path = PROJECT_DIR / "query_results.json"
    with open(query_results_path, "r", encoding="utf-8") as f:
        query_data = json.load(f)
    query3 = query_data.get("query3", "")
    if not query3:
        raise ValueError(f"query3 not found in {query_results_path}")

    print(f"[Step 0] query3: {query3[:150]}...")

    # Decompose into sub-questions
    sub_questions = _decompose_query(query3)
    print(f"[Step 0] Decomposed into {len(sub_questions)} sub-questions:")
    for i, q in enumerate(sub_questions, 1):
        print(f"  Q{i}: {q}")

    # For each sub-question, read evidence from evidence_qN/ and generate answer
    all_answers = {}
    for i, question in enumerate(sub_questions, 1):
        evidence_dir = PROJECT_DIR / f"evidence_q{i}"
        output_dir = PROJECT_DIR / f"answer_q{i}"
        output_dir.mkdir(parents=True, exist_ok=True)

        graphrag_json_path = evidence_dir / "graphrag_evidence.json"
        chroma_json_path = evidence_dir / "chroma_evidence.json"
        graphrag_txt_path = evidence_dir / "graphrag_evidence.txt"
        chroma_txt_path = evidence_dir / "chroma_evidence.txt"

        print(f"\n{'='*60}")
        print(f"  Q{i}: {question[:100]}...")
        print(f"  Evidence: {evidence_dir}")
        print(f"  Output:   {output_dir}")
        print(f"{'='*60}")

        if not graphrag_json_path.exists() and not chroma_json_path.exists():
            print(f"  [WARN] No evidence found in {evidence_dir}, skipping")
            continue

        # Temporarily redirect OUTPUT_DIR for this question
        global OUTPUT_DIR
        _saved_output_dir = OUTPUT_DIR
        OUTPUT_DIR = output_dir

        try:
            answer = await generate_answer_via_generalized_mechanisms(
                query=question,
                graphrag_json_path=graphrag_json_path,
                chroma_json_path=chroma_json_path,
                graphrag_txt_path=graphrag_txt_path,
                chroma_txt_path=chroma_txt_path,
            )
            all_answers[f"Q{i}"] = {"query": question, "answer": answer}
        finally:
            OUTPUT_DIR = _saved_output_dir

    # Save summary
    summary_path = PROJECT_DIR / "writing_results.json"
    summary_path.write_text(
        json.dumps(all_answers, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[Done] Results saved to {summary_path}")
    return all_answers


# ============================================================
# 16. LangGraph node wrapper
# ============================================================

async def _run_writing_async(state: dict) -> dict:
    """Async core: reads evidence dirs from state, writes answers, returns updated state."""
    project_dir = Path(__file__).resolve().parent

    # Read query_results.json
    query_results_path = project_dir / "query_results.json"
    if not query_results_path.exists():
        raise FileNotFoundError(f"query_results.json not found at {query_results_path}")
    query_data = json.loads(query_results_path.read_text(encoding="utf-8"))
    query3 = query_data.get("query3", "")
    if not query3:
        raise ValueError("query3 not found in query_results.json")

    print(f"[Writing Agent] query3: {query3[:150]}...")

    # Decompose into sub-questions
    sub_questions = _decompose_query(query3)
    print(f"[Writing Agent] Decomposed into {len(sub_questions)} sub-questions:")
    for i, q in enumerate(sub_questions, 1):
        print(f"  Q{i}: {q}")

    # For each sub-question, read evidence from evidence_qN/ and generate answer
    all_answers = {}
    for i, question in enumerate(sub_questions, 1):
        evidence_dir = project_dir / f"evidence_q{i}"
        output_dir = project_dir / f"answer_q{i}"
        output_dir.mkdir(parents=True, exist_ok=True)

        graphrag_json_path = evidence_dir / "graphrag_evidence.json"
        chroma_json_path = evidence_dir / "chroma_evidence.json"
        graphrag_txt_path = evidence_dir / "graphrag_evidence.txt"
        chroma_txt_path = evidence_dir / "chroma_evidence.txt"

        print(f"\n{'='*60}")
        print(f"  Q{i}: {question[:100]}...")
        print(f"{'='*60}")

        if not graphrag_json_path.exists() and not chroma_json_path.exists():
            print(f"  [WARN] No evidence found in {evidence_dir}, skipping")
            continue

        global OUTPUT_DIR
        _saved_output_dir = OUTPUT_DIR
        OUTPUT_DIR = output_dir

        try:
            answer = await generate_answer_via_generalized_mechanisms(
                query=question,
                graphrag_json_path=graphrag_json_path,
                chroma_json_path=chroma_json_path,
                graphrag_txt_path=graphrag_txt_path,
                chroma_txt_path=chroma_txt_path,
            )
            all_answers[f"Q{i}"] = {"query": question, "answer": answer}
        finally:
            OUTPUT_DIR = _saved_output_dir

    # Save summary
    summary_path = project_dir / "writing_results.json"
    summary_path.write_text(
        json.dumps(all_answers, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[Writing Agent] Results saved to {summary_path}")

    state["answers"] = all_answers
    state["writing_results_path"] = str(summary_path)
    return state


def run(state: dict) -> dict:
    """
    LangGraph node: Writing Agent.
    Reads query from query_results.json, evidence from evidence_qN/ dirs,
    generates structured explanations, saves to writing_results.json.
    """
    return asyncio.run(_run_writing_async(state))


if __name__ == "__main__":
    asyncio.run(main())