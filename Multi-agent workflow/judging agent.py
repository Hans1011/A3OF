import os
import json
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from openai import OpenAI


# =========================
# 1. Agicto API 设置
# =========================

AGICTO_API_KEY = os.getenv("AGICTO_API_KEY")
AGICTO_BASE_URL = os.getenv("AGICTO_BASE_URL", "https://api.agicto.cn/v1")

if not AGICTO_API_KEY:
    raise RuntimeError("请先设置环境变量 AGICTO_API_KEY")

client = OpenAI(
    api_key=AGICTO_API_KEY,
    base_url=AGICTO_BASE_URL
)


# =========================
# 2. Stage 1 Prompt
# =========================

EVIDENCE_GROUNDING_JUDGE_PROMPT = """
You are an expert judge tasked with evaluating whether an AI-generated explanation
is grounded in the provided evidence.

Analyze the provided INPUT, CONTEXT, and OUTPUT to determine whether the OUTPUT
introduces unsupported information or contradicts the CONTEXT.

Guidelines:
1. The OUTPUT must not introduce new information beyond what is provided in the CONTEXT.
2. The OUTPUT must not contradict any information given in the CONTEXT.
3. The OUTPUT should not contradict well-established facts or general knowledge.
4. Ignore the INPUT when evaluating faithfulness; it is provided for context only.
5. Consider partial hallucinations where some information is correct but other parts are not.
6. Check that the OUTPUT does not oversimplify or generalize information in a way that
   changes its meaning or accuracy.

Analyze the text thoroughly and assign a hallucination score between 0 and 1, where:
- 0.0: The OUTPUT is entirely faithful to the CONTEXT.
- 1.0: The OUTPUT is entirely unfaithful to the CONTEXT.

INPUT:
{input}

CONTEXT:
{context}

OUTPUT:
{output}

Provide your verdict in JSON format:
{{
    "score": <your score between 0.0 and 1.0>,
    "pass": <true if score is below the predefined threshold, otherwise false>,
    "reason": [
        <list your reasoning as bullet points>
    ]
}}
"""


# =========================
# 3. Stage 2 Prompt
# =========================

ASSOCIATION_CONSISTENCY_JUDGE_PROMPT = """
You are an expert judge tasked with evaluating whether an AI-generated explanation
is consistent with extracted scientific associations.

Analyze the provided EXTRACTED ASSOCIATIONS and OUTPUT to determine whether the
OUTPUT preserves the correct entity-feature, feature-effect, and condition-outcome
relationships.

Guidelines:
1. The OUTPUT must not contradict the extracted associations.
2. The OUTPUT must not assign attributes, actions, effects, or outcomes to the wrong entities.
3. The OUTPUT must preserve the direction of each association, such as positive,
   negative, enhancing, suppressing, increasing, or decreasing effects.
4. The OUTPUT must not conflate different entities, features, experimental conditions,
   or outcomes.
5. The OUTPUT must not overgeneralize an association beyond the conditions under which
   it was extracted.
6. Consider partial inconsistency where some associations are preserved but others are distorted.

EXTRACTED ASSOCIATIONS:
{associations}

OUTPUT:
{output}

Provide your verdict in JSON format:
{{
    "pass": <true or false>,
    "score": <your inconsistency score between 0.0 and 1.0>,
    "reason": [
        <list your reasoning as bullet points>
    ]
}}
"""


# =========================
# 4. 工具函数
# =========================

def read_txt(file_path):
    """
    读取 txt 文件内容。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def extract_json(text):
    """
    从 LLM 返回结果中提取 JSON。
    兼容 ```json ... ``` 或普通文本。
    """
    if isinstance(text, dict):
        return text

    text = text.strip()

    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Cannot parse JSON from LLM output:\n{text}")


def llm_call(prompt, model="gpt-4o-mini"):
    """
    调用 Agicto API。
    输入 prompt，返回模型输出文本。
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a strict scientific judging agent. Always return valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


def build_evidence_grounding_prompt(input_text, context, output):
    return EVIDENCE_GROUNDING_JUDGE_PROMPT.format(
        input=input_text,
        context=context,
        output=output
    )


def build_association_consistency_prompt(associations, output):
    return ASSOCIATION_CONSISTENCY_JUDGE_PROMPT.format(
        associations=associations,
        output=output
    )


# =========================
# 5. Two-stage vetting
# =========================

def two_stage_vetting(
    input_text,
    context,
    associations,
    output,
    model="gpt-4o-mini",
    grounding_threshold=0.3,
    association_threshold=0.3
):
    """
    Two-stage vetting process.

    Stage 1:
        Evidence-grounding check.
        检查 output 是否被 context 支持。

    Stage 2:
        Association-consistency check.
        检查 output 是否与 associations 一致。
    """

    # ---------- Stage 1 ----------
    grounding_prompt = build_evidence_grounding_prompt(
        input_text=input_text,
        context=context,
        output=output
    )

    grounding_raw = llm_call(
        prompt=grounding_prompt,
        model=model
    )

    grounding_result = extract_json(grounding_raw)

    # ---------- Stage 2 ----------
    association_prompt = build_association_consistency_prompt(
        associations=associations,
        output=output
    )

    association_raw = llm_call(
        prompt=association_prompt,
        model=model
    )

    association_result = extract_json(association_raw)

    grounding_score = float(grounding_result.get("score", 1.0))
    association_score = float(association_result.get("score", 1.0))

    grounding_pass = grounding_score <= grounding_threshold
    association_pass = association_score <= association_threshold

    final_result = {
        "grounding_check": grounding_result,
        "association_check": association_result,
        "thresholds": {
            "grounding_threshold": grounding_threshold,
            "association_threshold": association_threshold
        },
        "finalized": grounding_pass and association_pass
    }

    return final_result


# =========================
# 6. Auto-discover and judge all answers from writing agent
# =========================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_associations_text() -> str:
    """Load associations from associations.json."""
    path = os.path.join(PROJECT_DIR, "associations.json")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("associations", "")
    if isinstance(raw, list):
        return "\n\n".join(raw)
    return str(raw)


def _load_sub_questions() -> dict:
    """Load sub-questions from writing_results.json or decompose from query3."""
    # Try writing_results.json first
    path = os.path.join(PROJECT_DIR, "writing_results.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: v.get("query", "") for k, v in data.items()}

    # Fallback: read query3 and decompose
    query_path = os.path.join(PROJECT_DIR, "query_results.json")
    if not os.path.exists(query_path):
        raise FileNotFoundError("Neither writing_results.json nor query_results.json found")

    with open(query_path, "r", encoding="utf-8") as f:
        query_data = json.load(f)
    query3 = query_data.get("query3", "")
    if not query3:
        raise ValueError("query3 not found in query_results.json")

    def _to_snake_case(text):
        text = text.strip().lower()
        text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
        text = re.sub(r"_+", "_", text)
        return text.strip("_")

    prompt = f"""You are given a compound scientific question.
Your task is to decompose it into independent factor-response-outcome relationships.
Return ONLY valid JSON.
Required JSON format:
{{"relationships": [{{"factor": "...", "response_pattern": "...", "outcome": "..."}}]}}
Question: {query3}"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a scientific text understanding assistant. Return only the requested output."},
                  {"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content
    raw = re.sub(r"^```json", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"```$", "", raw).strip()
    data = json.loads(raw)
    relationships = data.get("relationships", [])

    result = {}
    for i, item in enumerate(relationships, 1):
        factor = _to_snake_case(item["factor"])
        pattern = item["response_pattern"].strip()
        outcome = item["outcome"].strip()
        result[f"Q{i}"] = f"Why does {factor} follow a {pattern} response, controlling {outcome}?"
    return result


def _format_report(results: dict, grounding_threshold: float, association_threshold: float) -> str:
    """Format judging results into a clean text report."""
    lines = []
    lines.append("=" * 70)
    lines.append("JUDGING REPORT")
    lines.append("=" * 70)
    lines.append(f"Grounding threshold: {grounding_threshold}")
    lines.append(f"Association threshold: {association_threshold}")
    lines.append("")

    overall_pass = True
    for qname in sorted(results.keys()):
        data = results[qname]
        g = data.get("grounding_check", {})
        a = data.get("association_check", {})
        g_score = g.get("score", 1.0)
        a_score = a.get("score", 1.0)
        g_pass = g_score <= grounding_threshold
        a_pass = a_score <= association_threshold
        q_pass = g_pass and a_pass
        if not q_pass:
            overall_pass = False

        lines.append("-" * 70)
        lines.append(f"  {qname}")
        lines.append(f"  Query: {data.get('query', '')[:200]}...")
        lines.append("")
        lines.append(f"  Stage 1 - Evidence Grounding:")
        lines.append(f"    Score: {g_score:.3f}  {'PASS' if g_pass else 'FAIL'}")
        for r in g.get("reason", [])[:3]:
            lines.append(f"    - {r}")
        lines.append("")
        lines.append(f"  Stage 2 - Association Consistency:")
        lines.append(f"    Score: {a_score:.3f}  {'PASS' if a_pass else 'FAIL'}")
        for r in a.get("reason", [])[:3]:
            lines.append(f"    - {r}")
        lines.append("")

    lines.append("=" * 70)
    lines.append(f"OVERALL: {'ALL PASS' if overall_pass else 'SOME FAIL'}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    grounding_threshold = 0.3
    association_threshold = 0.3

    sub_questions = _load_sub_questions()
    associations_text = _load_associations_text()

    print(f"Found {len(sub_questions)} questions to judge: {list(sub_questions.keys())}")
    print(f"Associations loaded: {len(associations_text)} chars")

    results = {}
    for qname in sorted(sub_questions.keys()):
        question = sub_questions[qname]
        qnum = qname.replace("Q", "")
        answer_dir = os.path.join(PROJECT_DIR, f"answer_q{qnum}")

        context_path = os.path.join(answer_dir, "combined_context_used.txt")
        output_path = os.path.join(answer_dir, "final_answer_from_generalized_mechanisms.txt")

        if not os.path.exists(output_path):
            print(f"\n  [{qname}] SKIP: no answer found at {output_path}")
            continue

        context = read_txt(context_path) if os.path.exists(context_path) else ""
        output = read_txt(output_path)

        print(f"\n  [{qname}] Judging...")
        print(f"    context: {len(context)} chars")
        print(f"    output:  {len(output)} chars")

        result = two_stage_vetting(
            input_text=question,
            context=context,
            associations=associations_text,
            output=output,
            model="gpt-4o-mini",
            grounding_threshold=grounding_threshold,
            association_threshold=association_threshold,
        )
        result["query"] = question
        results[qname] = result

        g_score = result["grounding_check"].get("score", "?")
        a_score = result["association_check"].get("score", "?")
        passed = result["finalized"]
        print(f"    grounding={g_score}  association={a_score}  pass={passed}")

    # Save JSON
    json_path = os.path.join(PROJECT_DIR, "judge_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Generate TXT report
    report = _format_report(results, grounding_threshold, association_threshold)
    report_path = os.path.join(PROJECT_DIR, "judging_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{report}")
    print(f"\nJSON saved to: {json_path}")
    print(f"Report saved to: {report_path}")


# =========================
# 7. LangGraph node wrapper
# =========================

def run(state: dict) -> dict:
    """
    LangGraph node: Judging Agent.
    Reads Writing agent output from state, runs two-stage vetting,
    and updates state with judge_results and final_report.
    """
    project_dir = Path(__file__).resolve().parent
    grounding_threshold = 0.3
    association_threshold = 0.3

    sub_questions_raw = state.get("answers", {})
    associations_text = state.get("associations_text", "")

    if not associations_text:
        associations_path = project_dir / "associations.json"
        if associations_path.exists():
            data = json.loads(associations_path.read_text(encoding="utf-8"))
            raw = data.get("associations", "")
            if isinstance(raw, list):
                associations_text = "\n\n".join(raw)
            else:
                associations_text = str(raw)
        state["associations_text"] = associations_text

    # Map Q1/Q2/Q3 to their answer directories
    sub_questions = {}
    for qname, answer_text in sub_questions_raw.items():
        # answer_text may be the answer string directly or a dict
        if isinstance(answer_text, dict):
            sub_questions[qname] = answer_text.get("query", answer_text.get("answer", str(answer_text)))
        else:
            sub_questions[qname] = str(answer_text)

    if not sub_questions:
        # Fallback: try reading from writing_results.json
        wr_path = project_dir / "writing_results.json"
        if wr_path.exists():
            wr_data = json.loads(wr_path.read_text(encoding="utf-8"))
            for k, v in wr_data.items():
                sub_questions[k] = v.get("query", v.get("answer", str(v)))

    print(f"Found {len(sub_questions)} questions to judge: {list(sub_questions.keys())}")
    print(f"Associations loaded: {len(associations_text)} chars")

    results = {}
    for qname in sorted(sub_questions.keys()):
        qnum = qname.replace("Q", "")
        answer_dir = project_dir / f"answer_q{qnum}"

        context_path = answer_dir / "combined_context_used.txt"
        output_path = answer_dir / "final_answer_from_generalized_mechanisms.txt"

        if not output_path.exists():
            print(f"\n  [{qname}] SKIP: no answer found at {output_path}")
            continue

        context = ""
        if context_path.exists():
            context = context_path.read_text(encoding="utf-8")
        output = output_path.read_text(encoding="utf-8")

        # Get the query from writing_results.json
        wr_path = project_dir / "writing_results.json"
        question = ""
        if wr_path.exists():
            wr_data = json.loads(wr_path.read_text(encoding="utf-8"))
            question = wr_data.get(qname, {}).get("query", "")

        print(f"\n  [{qname}] Judging...")
        print(f"    context: {len(context)} chars")
        print(f"    output:  {len(output)} chars")

        result = two_stage_vetting(
            input_text=question,
            context=context,
            associations=associations_text,
            output=output,
            model="gpt-4o-mini",
            grounding_threshold=grounding_threshold,
            association_threshold=association_threshold,
        )
        result["query"] = question
        results[qname] = result

        g_score = result["grounding_check"].get("score", "?")
        a_score = result["association_check"].get("score", "?")
        passed = result["finalized"]
        print(f"    grounding={g_score}  association={a_score}  pass={passed}")

    # Save JSON
    json_path = project_dir / "judge_results.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Generate TXT report
    report = _format_report(results, grounding_threshold, association_threshold)
    report_path = project_dir / "judging_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print(f"\n{report}")

    state["judge_results"] = results
    state["judge_results_path"] = str(json_path)
    state["final_report"] = report
    state["final_report_path"] = str(report_path)

    return state


if __name__ == "__main__":
    main()