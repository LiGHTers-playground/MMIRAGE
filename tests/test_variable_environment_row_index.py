from mmirage.core.process.variables import InputVar, VariableEnvironment


def test_row_index_defaults_to_none_and_stays_out_of_the_variables():
    env = VariableEnvironment({"text": "a"})
    assert env.row_index is None
    assert "row_index" not in env.to_dict()


def test_with_variable_propagates_row_index():
    env = VariableEnvironment({"text": "a"}, row_index=7)
    derived = env.with_variable("score", 1).with_variable("tag", "x")

    assert derived.row_index == 7
    assert dict(derived.to_dict()) == {"text": "a", "score": 1, "tag": "x"}
    assert "row_index" not in derived.to_dict()


def test_from_batch_input_variables_takes_indices():
    batch = {"text": ["a", "b", "c"]}
    input_vars = [InputVar(name="text", key="text")]

    without = VariableEnvironment.from_batch_input_variables(batch, input_vars)
    assert [e.row_index for e in without] == [None, None, None]

    with_indices = VariableEnvironment.from_batch_input_variables(
        batch, input_vars, indices=[10, 11, 12]
    )
    assert [e.row_index for e in with_indices] == [10, 11, 12]
    assert [e.get("text") for e in with_indices] == ["a", "b", "c"]
