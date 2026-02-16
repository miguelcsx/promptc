from promptc.dspy.programs import _required_float, _required_json_list


def test_required_json_list_accepts_json_list() -> None:
    assert _required_json_list('["a", "b"]') == ["a", "b"]


def test_required_json_list_accepts_python_list_string() -> None:
    assert _required_json_list("['a', 'b']") == ["a", "b"]


def test_required_json_list_accepts_bullet_list_string() -> None:
    assert _required_json_list("- a\n- b") == ["a", "b"]


def test_required_json_list_accepts_comma_separated_string() -> None:
    assert _required_json_list("a, b, c") == ["a", "b", "c"]


def test_required_json_list_accepts_single_text_item() -> None:
    assert _required_json_list("single item") == ["single item"]


def test_required_float_accepts_float_like_word_fraction() -> None:
    assert _required_float("0. nine") == 0.9


def test_required_float_accepts_embedded_numeric_token() -> None:
    assert _required_float("score=0.85 (good)") == 0.85
