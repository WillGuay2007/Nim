from nim import available_actions


def test_empty_board_falsy():
    empty_board = ()
    assert not available_actions(empty_board)

def test_empty_board_empty():
    empty_board = ()
    assert len(available_actions(empty_board)) == 0

def test_documented_example():
    assert {(0, 1), (1, 1), (0, 2)} == set(available_actions([2, 1, 0, 0]))

def test_last_move():
    assert {(0, 1)} == set(available_actions([1]))

def test_no_move():
    assert set() == set(available_actions([0]))

def test_one_heap_multiple_sticks():
    assert {(0, 1), (0, 2), (0, 3)} == set(available_actions([3]))

def test_heaps_gap_in_middle():
    assert {(0, 1), (0, 2), (2, 1), (2, 2)} == set(available_actions([2, 0, 2]))

def test_big_heap():
    assert {(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6) } == set(available_actions([6]))

def test_no_duplicate_actions():
    assert len(available_actions([2, 3, 1])) == 6
