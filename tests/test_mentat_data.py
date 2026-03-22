from nanochat.mentat.data import generate_trace_example


def test_generate_trace_example_is_deterministic():
    ex1 = generate_trace_example(7, "train")
    ex2 = generate_trace_example(7, "train")
    assert ex1.text == ex2.text
    assert ex1.outputs == ex2.outputs
    assert ex1.errors == ex2.errors


def test_train_and_val_examples_differ():
    train_ex = generate_trace_example(7, "train")
    val_ex = generate_trace_example(7, "val")
    assert train_ex.text != val_ex.text


def test_generated_trace_has_expected_sections():
    ex = generate_trace_example(0, "train")
    assert "program:" in ex.text
    assert "trace:" in ex.text
    assert "final:" in ex.text
    assert "outputs=" in ex.text
