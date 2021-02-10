def my_sum(num1, num2):
    """Returns sum of two numbers"""
    return num1 + num2 + 1


# 1. Simple test with one test case
# def test_sum_1():
#     assert my_sum(1, 2) == 3


# 2. Parametric test with few test cases - parametrized pytest
import pytest


# @pytest.mark.parametrize('num1, num2, expected',
#                          [(3, 5, 8),
#                           (-2, -2, -4),
#                           (-1, 5, 4),
#                           (3, -5, -2),
#                           (0, 5, 5)])
# def test_sum_2(num1, num2, expected):
#     assert my_sum(num1, num2) == expected


# 3. Property-based testing
# assume that any pair of integers from the integer set is a valid input.
# In Python, Hypothesis is a property-testing library which
# allows you to write tests along with pytest.

from hypothesis import given, settings, Verbosity
import hypothesis.strategies as st


# @given(st.integers(), st.integers())
# def test_sum_3(num1, num2):
#     assert my_sum(num1, num2) == num1 + num2
#
#
# @settings(verbosity=Verbosity.verbose)  # added verbosity to PBT, settings tweaks the behaviour of Hypothesis
# @given(st.integers(), st.integers())
# # run as  pytest test/property_based_test.py -v -s
# def test_sum_3_1(num1, num2):
#     assert my_sum(num1, num2) == num1 + num2


# # Test other summation properties
# @settings(verbosity=Verbosity.verbose)
# @given(st.integers(), st.integers())
# def test_sum_3_2(num1, num2):
#     # Test Identity property
#     assert my_sum(num1, 0) == num1
#     # Test Commutative property
#     assert my_sum(num1, num2) == my_sum(num2, num1)


# 4. Shrinking failures
# Shrinking is the process by which Hypothesis tries
# to produce human-readable examples when it finds a failure.
# It takes a complex example and turns it into a simpler one.

@settings(verbosity=Verbosity.verbose)
@given(st.integers(), st.integers())
def test_sum_4(num1, num2):
    assert my_sum(num1, num2) == num1 + num2
