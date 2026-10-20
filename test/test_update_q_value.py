from nim import ai


def test_update_q_value():
    model = ai.QLearning(alpha=0.5, gamma=1.0)
    state = (1, 2)
    action = (0, 1)

    model.update_q_value(state, action, 0, 1, 0)

    assert model.q[(state, action)] == 0.5


def test_update_q_value_with_old_q():
    model = ai.QLearning(alpha=0.5, gamma=1.0)
    state = (1, 2)
    action = (0, 1)

    model.update_q_value(state, action, 0.5, 1, 0)

    assert model.q[(state, action)] == 0.75


def test_update_q_value_negative_reward():
    model = ai.QLearning(alpha=0.5, gamma=1.0)
    state = (1, 2)
    action = (0, 1)

    model.update_q_value(state, action, 0, -1, 0)

    assert model.q[(state, action)] == -0.5


def test_update_q_value_with_future_reward():
    model = ai.QLearning(alpha=0.5, gamma=1.0)
    model.update_q_value((1, 2), (0, 1), 0, 0, 1)
    assert model.q[(1, 2), (0, 1)] == 0.5


def test_update_q_value_with_no_gamma():
    model = ai.QLearning(alpha=0.5, gamma=0)
    model.update_q_value((1, 2), (0, 1), 0, 0, 1)
    assert model.q[(1, 2), (0, 1)] == 0


def test_update_q_value_with_low_alpha():
    model = ai.QLearning(alpha=0.05, gamma=1.0)
    state = (1, 2)
    action = (0, 1)

    model.update_q_value(state, action, 0, 1, 0)

    assert model.q[(state, action)] == 0.05
