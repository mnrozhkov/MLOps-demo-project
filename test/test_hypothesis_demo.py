from hypothesis import given
from hypothesis import strategies as st


def backwards_all_caps(text):
    return text[::-1].upper()


@given(st.text())
def test_backwards_all_caps(input_string):
    modified = backwards_all_caps(input_string)
    assert input_string.upper() == ''.join(reversed(modified))
