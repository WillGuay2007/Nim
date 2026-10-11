from random import randint

import pytest

from nim import new_random_board


def test_num_heaps_negative():
    negative_integer = -4

    with pytest.raises(ValueError):
        new_random_board(negative_integer)


def test_num_heaps_random():
    random_heaps_integer = randint(1, 10)

    assert len(new_random_board(random_heaps_integer)) == random_heaps_integer


def test_num_heaps_provided():
    number_of_heaps_provided = 3
    assert number_of_heaps_provided == len(new_random_board(number_of_heaps_provided))


def test_num_heaps_not_provided():
    assert 0 < len(new_random_board()) <= 10
