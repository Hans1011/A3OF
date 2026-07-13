
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

# ---------------------------------------------------------------------------
# Project root & environment
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# LangGraph imports (install if missing)
try:
    from langgraph.graph import END, StateGraph
except ImportError:
    print(
        "LangGraph is not installed. Run:\n"
        "  pip install langgraph langgraph-checkpoint"
    )
    sys.exit(1)


# ====================================================================
# Pipeline State (shared TypedDict)
# ====================================================================

class PipelineState(TypedDict, total=False):
    """Shared state flowing through all 6 agents."""

    # ── Input files ──
    data_csv: str
    caption_csv: str
    pdf_dir: str
    skip_mining: bool

    # ── Extraction Agent outputs ──
    associations: List[str]
    associations_path: str
    associations_text: str

    # ── Query Agent outputs ──
    query1: str
    query2: str
    query3: str
    intermediate_variable: str
    query_results_path: str

    # ── Mining Agent outputs ──
    downloaded_papers_dir: str

    # ── Searching Agent outputs ──
    evidence_dirs: List[str]

    # ── Writing Agent outputs ──
    answers: Dict[str, Any]
    writing_results_path: str

    # ── Judging Agent outputs ──
    judge_results: Dict[str, Any]
    judge_results_path: str
    final_report: str
    final_report_path: str

    # ── Pipeline metadata ──
    current_stage: str
    errors: List[str]


# ====================================================================
# Module loader (supports filenames with spaces)
# ====================================================================

_AGENT_CACHE: Dict[str, Any] = {}


def _import_agent_module(agent_filename: str):
    """Import a Python module by filename (handles spaces in filename)."""
    if agent_filename in _AGENT_CACHE:
        return _AGENT_CACHE[agent_filename]

    filepath = PROJECT_ROOT / agent_filename
    if not filepath.exists():
        raise FileNotFoundError(f"Agent file not found: {filepath}")

    module_name = agent_filename.replace(" ", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # register so sub-imports work
    spec.loader.exec_module(module)
    _AGENT_CACHE[agent_filename] = module
    return module


# ====================================================================
# Node factory
# ====================================================================

def make_node(agent_filename: str, node_name: str):
    """
    Return a LangGraph node function.

    The agent file MUST export a `run(state: dict) -> dict` function.
    """

    def node(state: PipelineState) -> PipelineState:
        print(f"\n{'#' * 60}")
        print(f"#  NODE: {node_name}  ({agent_filename})")
        print(f"{'#' * 60}")

        state["current_stage"] = node_name.lower()

        try:
            agent = _import_agent_module(agent_filename)

            if not hasattr(agent, "run"):
                raise AttributeError(
                    f"{agent_filename} does not export a run(state) function. "
                    "Please add one."
                )

            new_state = agent.run(state)
            print(f"  [{node_name}] ✓ completed")
            return new_state

        except Exception as exc:
            print(f"  [{node_name}] ✗ ERROR: {exc}")
            errors: List[str] = state.get("errors", [])
            errors.append(f"{node_name}: {type(exc).__name__}: {exc}")
            state["errors"] = errors
            return state

    return node


# ====================================================================
# Conditional routing (skip Mining if requested)
# ====================================================================

def _route_after_query(state: PipelineState) -> str:
    """Decide whether to run Mining or jump straight to Searching."""
    if state.get("skip_mining"):
        print("\n  [Router] skip_mining=True → jumping to Searching")
        return "searching"
    return "mining"


# ====================================================================
# Build the LangGraph pipeline
# ====================================================================

def build_pipeline() -> StateGraph:
    """Construct the sequential 6-agent StateGraph."""

    builder = StateGraph(PipelineState)

    # ── Register nodes ──
    builder.add_node("extraction", make_node("Extraction agent.py", "Extraction"))
    builder.add_node("query", make_node("query agent.py", "Query"))
    builder.add_node("mining", make_node("mining agent.py", "Mining"))
    builder.add_node("searching", make_node("searching agent.py", "Searching"))
    builder.add_node("writing", make_node("writing agent.py", "Writing"))
    builder.add_node("judging", make_node("judging agent.py", "Judging"))

    # ── Edges ──
    builder.set_entry_point("extraction")
    builder.add_edge("extraction", "query")

    # Conditional: skip Mining if requested
    builder.add_conditional_edges(
        "query",
        _route_after_query,
        {"mining": "mining", "searching": "searching"},
    )
    builder.add_edge("mining", "searching")
    builder.add_edge("searching", "writing")
    builder.add_edge("writing", "judging")
    builder.add_edge("judging", END)

    return builder.compile()


# ====================================================================
# Main entry point
# ====================================================================

def main() -> PipelineState:
    parser = argparse.ArgumentParser(
        description="A3OF Multi-Agent Pipeline — LangGraph orchestration"
    )
    parser.add_argument(
        "--data_csv",
        default=str(PROJECT_ROOT / "augmented_data_with_pic.csv"),
        help="Path to augmented experimental data CSV (default: augmented_data_with_pic.csv)",
    )
    parser.add_argument(
        "--caption_csv",
        default=str(PROJECT_ROOT / "caption.csv"),
        help="Path to image caption CSV (default: caption.csv)",
    )
    parser.add_argument(
        "--pdf_dir",
        default=str(PROJECT_ROOT / "Papers"),
        help="Directory containing PDF papers for evidence retrieval (default: Papers/)",
    )
    parser.add_argument(
        "--skip_mining",
        action="store_true",
        help="Skip Consensus paper download (Mining Agent). Use if papers already in Papers/.",
    )
    parser.add_argument(
        "--stop_after",
        choices=["extraction", "query", "mining", "searching", "writing", "judging"],
        help="Stop pipeline after a specific agent (for debugging).",
    )

    args = parser.parse_args()

    # ── Validate inputs ──
    if not Path(args.data_csv).exists():
        print(f"[WARN] Data CSV not found: {args.data_csv}")
    if not Path(args.caption_csv).exists():
        print(f"[WARN] Caption CSV not found: {args.caption_csv}")

    # ── Build initial state ──
    initial_state: PipelineState = {
        "data_csv": args.data_csv,
        "caption_csv": args.caption_csv,
        "pdf_dir": args.pdf_dir,
        "skip_mining": args.skip_mining,
        "errors": [],
        "current_stage": "init",
    }

    graph = build_pipeline()

    print("=" * 70)
    print("  A3OF  Multi-Agent Pipeline")
    print("  Autonomous · Apprehensible · Accelerated Optimization")
    print("=" * 70)
    print(f"  Data CSV:      {args.data_csv}")
    print(f"  Caption CSV:   {args.caption_csv}")
    print(f"  PDF Dir:       {args.pdf_dir}")
    print(f"  Skip Mining:   {args.skip_mining}")
    if args.stop_after:
        print(f"  Stop after:    {args.stop_after}")
    print("=" * 70)
    print()
    print("  Agents (sequential):")
    print("    1. Extraction  — SHAP analysis + associations")
    print("    2. Query       — sub-query generation (Qwen LoRA)")
    if not args.skip_mining:
        print("    3. Mining      — Consensus API + PDF download")
    print(f"    {4 if args.skip_mining else 3}→{'4' if args.skip_mining else '5'}. Searching   — GraphRAG + Chroma evidence retrieval")
    print(f"    {'5' if args.skip_mining else '6'}. Writing     — mechanism extraction + explanation")
    print(f"    {'6' if args.skip_mining else '7'}. Judging     — two-stage hallucination vetting")
    print()
    print("=" * 70)

    # ── Run ──
    if args.stop_after:
        # Partial run for debugging
        final_state = initial_state
        stage_order = ["extraction", "query", "mining", "searching", "writing", "judging"]
        for stage in stage_order:
            if stage != "extraction":
                # Find the corresponding node
                pass
        print("[INFO] --stop_after not yet implemented for partial runs.")
        print("[INFO] Running full pipeline instead.")

    final_state = graph.invoke(initial_state)

    # ── Report ──
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)

    errors = final_state.get("errors", [])
    if errors:
        print(f"\n  ⚠ {len(errors)} error(s) encountered:")
        for e in errors:
            print(f"    - {e}")
    else:
        print("\n  ✓ All agents completed successfully.")

    print(f"\n  Output files:")
    print(f"    associations:      {final_state.get('associations_path', 'N/A')}")
    print(f"    queries:           {final_state.get('query_results_path', 'N/A')}")
    print(f"    downloaded papers: {final_state.get('downloaded_papers_dir', 'N/A')}")
    evidence = final_state.get("evidence_dirs", [])
    print(f"    evidence dirs:     {evidence if evidence else 'N/A'}")
    print(f"    writing results:   {final_state.get('writing_results_path', 'N/A')}")
    print(f"    final report:      {final_state.get('final_report_path', 'N/A')}")

    if final_state.get("final_report"):
        print(f"\n{'─' * 60}")
        print(final_state["final_report"])
        print(f"{'─' * 60}")

    return final_state


if __name__ == "__main__":
    main()
