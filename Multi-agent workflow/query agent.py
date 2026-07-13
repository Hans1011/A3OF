
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, List

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

_SCRIPT_DIR = Path(__file__).resolve().parent
INTERMEDIATE_INFER_SCRIPT = str(_SCRIPT_DIR / "Query intermediate variables.py")

PROJECT_DIR = str(_SCRIPT_DIR / "Model")

JSON_PATH = str(_SCRIPT_DIR / "associations.json")

SAVE_RESULTS_PATH = str(_SCRIPT_DIR / "query_results.json")

BASE_URL = os.getenv("AGICTO_BASE_URL", "https://api.agicto.cn/v1")

API_KEY = os.getenv("AGICTO_API_KEY")

MODEL_NAME = os.getenv("LLM_MODEL_LIGHT", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "You are a scientific reasoning assistant. "
    "Given experimental associations and an intermediate variable,"
    "generate the corresponding output question exactly and concisely."
)

PROMPT_HEADER = """
Based on the experimental context, nonlinear associations, and intermediate variable, generate the Output question.
We are studying water-oil interface. Through experiments, we have already found associations between input variables and output (MNP cluster crossing performance across water-oil interface), and we have also identified a possible intermediate variable linking both input and output.

Associations:
""".strip()

QUERY_CONFIGS = [
    {
        "name": "query1",
        "BASE_MODEL_PATH": os.path.join(PROJECT_DIR, "Qwen-2b"),
        "LORA_PATH": os.path.join(PROJECT_DIR, "qwen3b_lora_sft-10"),
    },
    {
        "name": "query2",
        "BASE_MODEL_PATH": os.path.join(PROJECT_DIR, "QWen-3b"),
        "LORA_PATH": os.path.join(PROJECT_DIR, "qwen3b_lora_sft-3"),
    },
    {
        "name": "query3",
        "BASE_MODEL_PATH": os.path.join(PROJECT_DIR, "Qwen-2b"),
        "LORA_PATH": os.path.join(PROJECT_DIR, "qwen3b_lora_sft-11"),
    },
]


def read_json_file(json_path: str) -> Dict[str, Optional[str]]:

    path = Path(json_path)

    if not path.exists():
        raise FileNotFoundError(f"Cannot find json file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw = data.get("associations", "")
    if isinstance(raw, list):
        associations_text = "\n\n".join(raw)
    else:
        associations_text = str(raw)

    # Check if intermediate_variable is present in the JSON
    intermediate_variable = data.get("intermediate_variable", None)

    return {
        "associations": associations_text,
        "intermediate_variable": intermediate_variable,
    }

VARIABLE_DISPLAY_MAP: Dict[str, str] = {
    "ion_concentration_in_water": "Ion concentration in water",
    "oil_type": "Oil type",
    "surfactant_in_water": "Aqueous surfactant",
    "surfactant_in_oil": "Surfactant in oil",
    "ratio_of_surfactant_in_water": "Aqueous surfactant ratio",
    "ratio_of_surfactant_in_oil": "Surfactant ratio in oil",
}


def _detect_main_variable(paragraph: str) -> Optional[str]:
    """Detect which of the six variables a paragraph is primarily about."""
    for var_name in VARIABLE_DISPLAY_MAP:
        if var_name.lower() in paragraph.lower():
            return var_name
    return None


def reformat_associations(associations_text: str, intermediate_variable: Optional[str]) -> tuple[str, Optional[str]]:

    text = associations_text.strip()

    if "Non-linear patterns observed" in text:
        return text, intermediate_variable

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if len(paragraphs) == 1:
        sentences = re.split(r'(?<=[.!?])\s+', paragraphs[0])
        # Group sentences by detected variable
        grouped: Dict[str, list[str]] = {}
        current_var = None
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            detected = _detect_main_variable(sent)
            if detected:
                current_var = detected
            if current_var:
                grouped.setdefault(current_var, []).append(sent)

        if grouped and len(grouped) <= 5:
            # Reform into per-variable paragraphs
            seen_vars: list[str] = []
            new_paragraphs: list[str] = []
            for var_name, sents in grouped.items():
                if var_name not in seen_vars:
                    seen_vars.append(var_name)
                    new_paragraphs.append(" ".join(sents))
            if new_paragraphs:
                paragraphs = new_paragraphs

    # Build formatted output
    output_parts = ["Non-linear patterns observed:", ""]
    used_vars: set[str] = set()
    pattern_num = 0

    for para in paragraphs:
        if not para.strip():
            continue
        var_name = _detect_main_variable(para)
        if var_name and var_name not in used_vars:
            used_vars.add(var_name)
            display = VARIABLE_DISPLAY_MAP.get(var_name, var_name)
            label = f"{pattern_num + 1}) {display} ({var_name})"
        else:
            pattern_num += 1
            label = f"{pattern_num + 1}) Pattern {pattern_num + 1}"

        pattern_num += 1
        output_parts.append(label)

        # Split paragraph into sentences and add as bullet points
        sentences = re.split(r'(?<=[.!?])\s+', para)
        for sent in sentences:
            sent = sent.strip()
            if sent:
                output_parts.append(f"- {sent}")
        output_parts.append("")

    formatted = "\n".join(output_parts).strip()

    return formatted, intermediate_variable

def build_query1_prompt(associations: str) -> str:

    return f"""
{PROMPT_HEADER}
{associations}
""".strip()


def build_user_prompt(associations: str, intermediate_variable: str) -> str:

    return f"""
{PROMPT_HEADER}
{associations}

Intermediate variable:{intermediate_variable}
""".strip()


def ask_external_py_with_query1_question(query1_question: str) -> str:

    script_path = Path(INTERMEDIATE_INFER_SCRIPT)

    if not script_path.exists():
        raise FileNotFoundError(
            f"Cannot find intermediate-variable inference script: {script_path}"
        )

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            input=query1_question,
            text=True,
            capture_output=True,
            check=True,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as e:
        print("===== External Script STDOUT =====")
        print(e.stdout)
        print()
        print("===== External Script STDERR =====")
        print(e.stderr)
        print()
        raise RuntimeError(
            "Failed to get answer by calling external intermediate-variable script."
        ) from e

    stdout = result.stdout.strip()
    answer = parse_intermediate_answer_from_script_output(stdout)
    return answer


def parse_intermediate_answer_from_script_output(stdout: str) -> str:

    if not stdout:
        raise ValueError("External script returned empty output.")

    match = re.search(
        r"(?is)Answer\s*[:：]\s*(.+)$",
        stdout,
    )

    if match:
        answer = match.group(1).strip()
    else:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        answer = lines[-1].strip()

    if not answer:
        raise ValueError("Failed to parse answer from external script output.")

    return answer


def extract_intermediate_variable(answer: str) -> str:

    text = answer.strip()

    text = re.sub(r"[*_`#]", "", text).strip()
    text = re.sub(r"^[-\d.\)\s]+", "", text).strip()

    patterns = [
        r"(?i)^the\s+most\s+likely\s+primary\s+intermediate\s+variable\s+is\s+",
        r"(?i)^the\s+primary\s+intermediate\s+variable\s+is\s+",
        r"(?i)^the\s+intermediate\s+variable\s+is\s+",
        r"(?i)^intermediate\s+variable\s*[:：]\s*",
        r"(?i)^answer\s*[:：]\s*",
    ]

    for pattern in patterns:
        text = re.sub(pattern, "", text).strip()

    text = text.split("\n", 1)[0].strip()

    sentence_end = re.search(r"(?<=[a-zA-Z\)])\.\s+", text)
    if sentence_end:
        text = text[: sentence_end.start() + 1].strip()

    text = text.rstrip(" .;；")

    return text


def load_model_and_tokenizer(base_model_path: str, lora_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
    )

    model.eval()

    return tokenizer, model


def run_inference(config: Dict[str, str], user_prompt: str) -> str:
    base_model_path = config["BASE_MODEL_PATH"]
    lora_path = config["LORA_PATH"]

    tokenizer, model = load_model_and_tokenizer(base_model_path, lora_path)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.02,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return response

def save_results(results: List[Dict[str, str]], save_path: str) -> None:
    """Save the three query outputs as a clean JSON file."""
    output = {item["name"]: item["response"] for item in results}
    Path(save_path).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def main():
    parsed = read_json_file(JSON_PATH)

    associations, intermediate_variable = reformat_associations(
        parsed["associations"],
        parsed["intermediate_variable"],
    )

    query1_question = ""
    gpt_answer = ""

    if intermediate_variable:
        pass
    else:

        query1_prompt = build_query1_prompt(associations)
        query1_question = run_inference(QUERY_CONFIGS[0], query1_prompt)


        gpt_answer = ask_external_py_with_query1_question(query1_question)


        intermediate_variable = extract_intermediate_variable(gpt_answer)

    user_prompt = build_user_prompt(
        associations=associations,
        intermediate_variable=intermediate_variable,
    )


    all_results = []

    for query_config in QUERY_CONFIGS:
        response = run_inference(query_config, user_prompt)
        all_results.append(
            {
                "name": query_config["name"],
                "response": response,
            }
        )

    save_results(all_results, SAVE_RESULTS_PATH)

    print("=" * 60)
    for item in all_results:
        print(f"\n[{item['name']}]")
        print(item["response"])


def run(state: dict) -> dict:
    """
    LangGraph node: Query Agent.
    Reads associations from state, generates query1/query2/query3
    using local Qwen+LoRA models, saves to query_results.json.
    Updates state with 'query1', 'query2', 'query3', 'intermediate_variable'.
    """
    global JSON_PATH
    project_dir = Path(__file__).resolve().parent

    # Override JSON_PATH with the one from state (or default)
    saved_json_path = JSON_PATH
    JSON_PATH = state.get("associations_path", str(project_dir / "associations.json"))

    try:
        main()
    finally:
        JSON_PATH = saved_json_path

    # Read results back
    results_path = project_dir / "query_results.json"
    if results_path.exists():
        data = json.loads(results_path.read_text(encoding="utf-8"))
        state["query1"] = data.get("query1", "")
        state["query2"] = data.get("query2", "")
        state["query3"] = data.get("query3", "")
    else:
        state["query1"] = ""
        state["query2"] = ""
        state["query3"] = ""

    state["query_results_path"] = str(results_path)
    return state


if __name__ == "__main__":
    main()
