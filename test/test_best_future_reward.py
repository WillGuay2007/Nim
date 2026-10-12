from nim import ai


def test_best_future_reward():
    model = ai.QLearning()
    model.q[(1, 0), (0, 1)] = -1
    assert model.best_future_reward((1, 0)) == -1

def test_best_future_reward_no_q_value():
    model = ai.QLearning()
    assert model.best_future_reward((1, 0)) == 0

def test_best_future_reward_no_actions():
    model = ai.QLearning()
    assert model.best_future_reward((0, 0)) == 0

def test_best_future_reward_pick_max():
    model = ai.QLearning()
    model.q[(2, 0), (0, 1)] = 1
    assert model.best_future_reward((2, 0)) == 1

def test_best_future_reward_untried_beat_negative():
    model = ai.QLearning()
    model.q[(2, 0), (0, 2)] = -1
    assert model.best_future_reward((2, 0)) == 0