from nim import transition


def test_transition():
    new_board = transition((1, 1), (0, 1))
    assert new_board == (0, 1)


def test_transition_invalid_heap():
    try:
        transition((1, 1), (10, 0))
    except IndexError:
        assert True
    else:
        assert False


def test_transition_less_than_1_picked():
    try:
        transition((1, 1), (0, -1))
    except ValueError:
        assert True
    else:
        assert False


def test_transition_take_more_than_heap_size():
    try:
        transition((1, 1), (0, 10))
    except ValueError:
        assert True
    else:
        assert False
