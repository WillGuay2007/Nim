from nim import ai


def test_get_q_value_invalid_input():
    model = ai.QLearning()
    state = (100, 2)
    action = (0, 100)

    assert model.get_q_value(state, action) == 0