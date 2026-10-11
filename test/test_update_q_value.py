from nim import ai


def test_update_q_value():
    model = ai.QLearning(alpha=0.5, gamma=1.0)
    state = (1, 2)
    action = (0, 1)

    model.update_q_value(state, action, 0, 1, 0)

    assert model.q[(state, action)] == 0.5