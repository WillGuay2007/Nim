from nim import ai

def test_get_q_value_invalid_input():
    model = ai.QLearning()
    model.q[(1, 0), (0, 1)] = 1

    assert model.best_future_reward((1, 0)) == 1