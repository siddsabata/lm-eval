from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.analysis.stability import StabilityAnalyzer
from src.evaluation.judge import LMJudge
from src.evaluation.regex_evaluator import RegexEvaluator
from src.utils.config import load_model_config, load_benchmark_config, make_benchmark
from src.utils.io import append_jsonl, read_jsonl, write_json
from src.utils.naming import slugify


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate inference outputs and compute stability metrics.")
    p.add_argument("--inference-dir", type=Path, required=True, help="Directory with inference outputs")
    p.add_argument("--benchmarks", nargs="+", required=True, help="Benchmarks to evaluate")
    p.add_argument("--models", nargs="*", default=None, help="Optional model keys to evaluate (defaults to all)")
    p.add_argument("--judge-model", type=str, required=False, help="Judge model config name (for lm_judge)")
    p.add_argument("--judge-votes", type=int, default=3, help="Number of judge votes per item")
    p.add_argument("--mt-pass-threshold", type=float, default=7.0, help="MT-Bench pass threshold")
    p.add_argument("--output-dir", type=Path, default=Path("./outputs"), help="Output directory")
    p.add_argument("--dev-mode", action="store_true", help="Match dev-mode sampling used for inference")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return p.parse_args()


def _checkpoint_path(output_dir: Path, model_key: str, benchmark: str, run_idx: int) -> Path:
    return output_dir / "checkpoints" / f"evaluation_{slugify(model_key)}_{benchmark}_run{run_idx}.json"


def _load_checkpoint(output_dir: Path, model_key: str, benchmark: str, run_idx: int, judged_path: Path) -> int:
    cp = _checkpoint_path(output_dir, model_key, benchmark, run_idx)
    existing_len = len(read_jsonl(judged_path))
    if not cp.exists():
        return existing_len
    try:
        import json

        data = json.loads(cp.read_text(encoding="utf-8"))
        last = int(data.get("last_completed_item", -1))
        return max(existing_len, last + 1, 0)
    except Exception:  # noqa: BLE001
        return existing_len


def _save_checkpoint(output_dir: Path, model_key: str, benchmark: str, run_idx: int, item_idx: int, total_items: int) -> None:
    write_json(
        _checkpoint_path(output_dir, model_key, benchmark, run_idx),
        {
            "model": model_key,
            "benchmark": benchmark,
            "run_index": run_idx,
            "last_completed_item": item_idx,
            "total_items": total_items,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


def _format_mt_question_and_response(entry: dict[str, Any]) -> tuple[str, str]:
    turn1 = entry["messages"][0]["content"] if entry.get("messages") else ""
    turn2 = (entry.get("metadata") or {}).get("turn2")
    question = f"Turn 1:\n{turn1}"
    if turn2:
        question += f"\n\nTurn 2:\n{turn2}"

    resp = entry.get("response")
    if isinstance(resp, list):
        parts = [f"Turn {i+1} Answer:\n{r}" for i, r in enumerate(resp)]
        response = "\n\n".join(parts)
    else:
        response = str(resp)
    return question, response


async def amain(args: argparse.Namespace) -> None:
    inference_dir = args.inference_dir
    output_dir = args.output_dir
    judgments_dir = output_dir / "judgments"
    judgments_dir.mkdir(parents=True, exist_ok=True)

    model_keys = args.models
    if not model_keys:
        model_keys = [p.name for p in inference_dir.iterdir() if p.is_dir()]

    bench_cfgs = {b: load_benchmark_config(b) for b in args.benchmarks}
    benchmarks = {b: make_benchmark(b, bench_cfgs[b]) for b in args.benchmarks}

    needs_judge = any((bench_cfgs[b].get("evaluation") or {}).get("method", "") != "regex" for b in args.benchmarks)
    judge: LMJudge | None = None
    if needs_judge:
        if not args.judge_model:
            raise SystemExit("--judge-model is required for lm_judge benchmarks")
        judge_cfg = load_model_config(args.judge_model)
        judge = LMJudge(judge_cfg, votes_per_item=args.judge_votes)
        if not judge.start():
            raise SystemExit("Failed to start judge vLLM server")

    regex_eval = RegexEvaluator()
    analyzer = StabilityAnalyzer()

    final: dict[str, Any] = {
        "meta": {
            "run_id": f"eval_{time.strftime('%Y-%m-%d_%H%M%S', time.gmtime())}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stability_runs": None,
            "temperature": None,
            "dev_mode": bool(args.dev_mode),
            "dev_samples": None,
            "judge_model": args.judge_model,
            "judge_votes_per_item": args.judge_votes,
        },
        "models": {},
        "comparison": {},
    }

    try:
        for model_key in model_keys:
            model_dir = inference_dir / model_key
            if not model_dir.exists():
                continue
            final["models"][model_key] = {}

            for bench_name in args.benchmarks:
                _ = benchmarks[bench_name]  # keep for future benchmark-specific helpers
                bench_dir = model_dir / bench_name
                if not bench_dir.exists():
                    continue

                run_files = sorted(bench_dir.glob("run_*.jsonl"))
                if not run_files:
                    continue

                if final["meta"]["stability_runs"] is None:
                    final["meta"]["stability_runs"] = len(run_files)

                per_run_results: dict[int, list[bool | float]] = {}
                categories: list[str] | None = None
                question_ids: list[str] | None = None

                for run_file in run_files:
                    run_idx = int(run_file.stem.split("_")[-1])
                    judged_path = judgments_dir / model_key / bench_name / f"run_{run_idx}_judged.jsonl"
                    judged_path.parent.mkdir(parents=True, exist_ok=True)

                    entries = read_jsonl(run_file)
                    if entries and final["meta"]["temperature"] is None:
                        final["meta"]["temperature"] = (entries[0].get("inference_config") or {}).get("temperature")

                    start_idx = (
                        _load_checkpoint(output_dir, model_key, bench_name, run_idx, judged_path) if args.resume else 0
                    )

                    run_results: list[bool | float] = []
                    run_categories: list[str] = []
                    run_ids: list[str] = []

                    if start_idx > 0:
                        existing_judged = read_jsonl(judged_path)
                        for j, entry in enumerate(entries[: len(existing_judged)]):
                            meta = entry.get("metadata") or {}
                            category = meta.get("subject") or meta.get("category") or "unknown"
                            run_categories.append(str(category))
                            run_ids.append(str(entry.get("id", j)))
                        for rec in existing_judged:
                            if bench_name == "mt_bench":
                                run_results.append(float(rec.get("final_score", 0.0)))
                            else:
                                run_results.append(bool(rec.get("pass", False)))

                    for i in tqdm(range(start_idx, len(entries)), desc=f"judge:{model_key}:{bench_name}:run{run_idx}"):
                        entry = entries[i]
                        run_ids.append(str(entry.get("id", i)))
                        meta = entry.get("metadata") or {}
                        category = meta.get("subject") or meta.get("category") or "unknown"
                        run_categories.append(str(category))

                        if bench_name == "mmlu":
                            correct = str(meta.get("correct_answer") or entry.get("ground_truth") or "").upper()
                            passed = regex_eval.evaluate(str(entry.get("response") or ""), correct)
                            run_results.append(bool(passed))
                            append_jsonl(
                                judged_path,
                                {
                                    "id": entry.get("id"),
                                    "benchmark": bench_name,
                                    "model_key": model_key,
                                    "run_index": run_idx,
                                    "response": entry.get("response"),
                                    "pass": bool(passed),
                                    "ground_truth": correct,
                                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                },
                            )
                        elif bench_name == "mt_bench":
                            if judge is None:
                                raise RuntimeError("Judge not initialized")
                            question, response = _format_mt_question_and_response(entry)
                            judge_failed = False
                            try:
                                judged = await judge.judge_mt_bench(question, response)
                                score = float(judged["final_score"])
                                votes = judged["votes"]
                            except Exception as e:  # noqa: BLE001
                                judge_failed = True
                                votes = []
                                score = 0.0
                                logging.getLogger(__name__).warning("Judge failed for %s: %s", entry.get("id"), e)
                            run_results.append(score)
                            append_jsonl(
                                judged_path,
                                {
                                    "id": entry.get("id"),
                                    "benchmark": bench_name,
                                    "model_key": model_key,
                                    "run_index": run_idx,
                                    "response": entry.get("response"),
                                    "judge_model": args.judge_model,
                                    "judge_votes": votes,
                                    "final_score": score,
                                    "pass": bool(score >= float(args.mt_pass_threshold)),
                                    "pass_threshold": float(args.mt_pass_threshold),
                                    "judge_failed": judge_failed,
                                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                },
                            )
                        elif bench_name == "multichallenge":
                            if judge is None:
                                raise RuntimeError("Judge not initialized")
                            rubric = str(meta.get("instance_rubric") or "")
                            conversation = entry.get("messages") or []
                            judge_failed = False
                            try:
                                judged = await judge.judge_multichallenge(
                                    conversation, str(entry.get("response") or ""), rubric
                                )
                                passed = bool(judged["pass"])
                                votes = judged["votes"]
                            except Exception as e:  # noqa: BLE001
                                judge_failed = True
                                votes = []
                                passed = False
                                logging.getLogger(__name__).warning("Judge failed for %s: %s", entry.get("id"), e)
                            run_results.append(passed)
                            append_jsonl(
                                judged_path,
                                {
                                    "id": entry.get("id"),
                                    "benchmark": bench_name,
                                    "model_key": model_key,
                                    "run_index": run_idx,
                                    "response": entry.get("response"),
                                    "judge_model": args.judge_model,
                                    "judge_votes": votes,
                                    "pass": passed,
                                    "judge_failed": judge_failed,
                                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                },
                            )
                        else:
                            raise ValueError(f"Unsupported benchmark: {bench_name}")

                        if (i + 1) % 100 == 0 or i == len(entries) - 1:
                            _save_checkpoint(output_dir, model_key, bench_name, run_idx, i, total_items=len(entries))

                    per_run_results[run_idx] = run_results
                    if categories is None:
                        categories = run_categories
                    if question_ids is None:
                        question_ids = run_ids

                stability = analyzer.analyze(per_run_results, question_ids=question_ids)
                per_cat = analyzer.analyze_by_category(per_run_results, categories) if categories else {}

                if bench_name == "mt_bench":
                    stability_out = {
                        "per_run_score": stability["per_run"],
                        "mean": stability["mean"],
                        "std": stability["std"],
                        "coefficient_of_variation": stability["coefficient_of_variation"],
                        "min": stability["min"],
                        "max": stability["max"],
                    }
                    per_cat_out = {
                        k: {
                            "n_questions": len([c for c in categories or [] if c == k]),
                            "per_run_score": v["per_run"],
                            "mean": v["mean"],
                            "std": v["std"],
                        }
                        for k, v in per_cat.items()
                    }
                else:
                    stability_out = {
                        "per_run_accuracy": stability["per_run"],
                        "mean": stability["mean"],
                        "std": stability["std"],
                        "coefficient_of_variation": stability["coefficient_of_variation"],
                        "min": stability["min"],
                        "max": stability["max"],
                    }
                    per_cat_out = {
                        k: {
                            "n_questions": len([c for c in categories or [] if c == k]),
                            "per_run_accuracy": v["per_run"],
                            "mean": v["mean"],
                            "std": v["std"],
                        }
                        for k, v in per_cat.items()
                    }

                method = (bench_cfgs[bench_name].get("evaluation") or {}).get("method", "regex")
                block: dict[str, Any] = {
                    "evaluation_method": method,
                    "total_questions": len(categories or []),
                    "stability": stability_out,
                    "per_category": per_cat_out,
                }
                if "question_stability" in stability:
                    block["question_stability"] = stability["question_stability"]
                if method != "regex":
                    block["judge_config"] = {"model": args.judge_model, "votes_per_item": args.judge_votes}
                final["models"][model_key][bench_name] = block

        for bench_name in args.benchmarks:
            ranking = []
            for model_key in final["models"].keys():
                bench_block = final["models"][model_key].get(bench_name)
                if not bench_block:
                    continue
                mean = float(bench_block["stability"]["mean"])
                std = float(bench_block["stability"]["std"])
                ranking.append({"model": model_key, "mean": mean, "std": std})
            ranking.sort(key=lambda x: x["mean"], reverse=True)
            if ranking:
                final["comparison"][bench_name] = {
                    "best_model": ranking[0]["model"],
                    "best_mean": ranking[0]["mean"],
                    "model_ranking": ranking,
                }

        write_json(output_dir / "final_evaluation.json", final)
    finally:
        if judge is not None:
            judge.stop()


def main() -> None:
    args = parse_args()
    _configure_logging(args.log_level)
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
