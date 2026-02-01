from src.evaluation.regex_evaluator import RegexEvaluator


def test_extract_answer_basic():
    ev = RegexEvaluator()
    assert ev.extract_answer("A") == "A"
    assert ev.extract_answer("Answer: B") == "B"
    assert ev.extract_answer("The answer is (c).") == "C"
    assert ev.extract_answer("Final: D.") == "D"


def test_evaluate():
    ev = RegexEvaluator()
    assert ev.evaluate("Answer: A", "A") is True
    assert ev.evaluate("Answer: B", "A") is False

