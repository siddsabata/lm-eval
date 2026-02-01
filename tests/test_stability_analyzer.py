from src.analysis.stability import StabilityAnalyzer


def test_analyze_bool_stability():
    analyzer = StabilityAnalyzer()
    results = {
        0: [True, False, True, False],
        1: [True, False, False, False],
        2: [True, False, True, False],
    }
    out = analyzer.analyze(results, question_ids=["q1", "q2", "q3", "q4"])
    assert out["per_run"] == [0.5, 0.25, 0.5]
    assert out["question_stability"]["always_correct"] == 0.25
    assert out["question_stability"]["always_incorrect"] == 0.5
    assert out["question_stability"]["inconsistent"] == 0.25
    assert out["question_stability"]["inconsistent_questions"] == ["q3"]


def test_analyze_float_stability():
    analyzer = StabilityAnalyzer()
    results = {0: [7.0, 8.0], 1: [6.0, 8.0]}
    out = analyzer.analyze(results)
    assert out["per_run"] == [7.5, 7.0]

