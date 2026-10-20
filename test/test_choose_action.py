from nim import ai

def test_choose_action_no_train():
    model = ai.QLearning()
    model.q[((3, 3), (0, 1))] = 1
    action = model.choose_action((3, 3), False)
    assert (action == (0, 1))

def test_choose_action_with_train():
    model = ai.QLearning(epsilon = 0.5)
    model.q[((1, 1), (0, 1))] = 1
    action = model.choose_action((1, 1), True)
    assert (action == (0, 1) or action == (1, 1))

def test_choose_action_no_train_equal_q():
    model = ai.QLearning()
    action = model.choose_action((1, 1), False)
    assert (action == (0, 1) or action == (1, 1)) #techniquement premier, pas random mais bon, il va se faire update de toute facon