from src.benchmarks.mmlu import MMLUBenchmark
from src.benchmarks.mt_bench import MTBenchBenchmark
from src.benchmarks.multichallenge import MultiChallengeBenchmark


def test_mmlu_prompt_format():
    b = MMLUBenchmark({"dataset": {"source": "huggingface", "path": "cais/mmlu", "split": "test"}})
    prompt = b.format_prompt("What is 2+2?", ["1", "2", "3", "4"], "math")
    assert "multiple choice question about math" in prompt
    assert "A. 1" in prompt
    assert "D. 4" in prompt
    assert prompt.strip().endswith("Answer:")


def test_mt_bench_judge_prompt_contains_rating_marker():
    b = MTBenchBenchmark({"dataset": {"source": "local", "path": "dummy"}})
    s = b.get_judge_prompt("Q", "R")
    assert "[[rating]]" in s
    assert "[Question]" in s


def test_multichallenge_judge_prompt_yes_no():
    b = MultiChallengeBenchmark({"dataset": {"source": "local", "path": "dummy"}})
    prompt = b.get_judge_prompt(
        conversation=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        response="ok",
        rubric="Did it say hello?",
    )
    assert "Answer only YES or NO" in prompt

