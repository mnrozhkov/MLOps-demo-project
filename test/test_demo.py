import pytest


def inc(x):
    """Function to test (might be imported as a module)"""
    return x + 1.0


def test_inc_1():
    """Simple test 1"""
    assert inc(3) == 4


# def test_inc_2():
#     """Simple test 2 - set to fail"""
#     assert inc(3) == 4


# Demonstrate fixtures in pytest:
# --------------------------------
@pytest.fixture(scope='function')  # runs once per function
def get_test_data():
    return [(0, 1),
            (-2, -1),
            (-1, 0),
            (3, 4), 
            (-1.0, 0.0)]


# A fixture can be marked as autouse=True,
# which will make every test in your suite use it by default:
# pytest test_example.py -v -s -- run with -s option to to print to stdout
@pytest.fixture(autouse=True)
def setup_and_teardown():
    """It will automatically be called before and after each test run: autouse=True. """
    print('\nFetching data from db')
    yield  # Anything written after yield is executed after the tests finish executing.
    print('\nSaving test run data in db')


def test_inc_type(get_test_data):
    for data in get_test_data:
        num = data[0]
        expected = data[1]
        assert isinstance(inc(num), float)


def test_inc_value(get_test_data):
    for data in get_test_data:
        num = data[0]
        expected = data[1]
        assert inc(num) == expected

