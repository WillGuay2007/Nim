from nim import ai


def test_get_q_value_invalid_input():
    model = ai.QLearning()
    assert model.get_q_value((1, 1), (1, 1)) == 0

def test_get_q_value_with_set_q():
    model = ai.QLearning()
    model.q[((1, 1), (0, 1))] = 1
    assert model.get_q_value((1, 1), (0, 1)) == 1