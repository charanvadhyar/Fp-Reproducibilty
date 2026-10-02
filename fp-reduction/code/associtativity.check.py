"""
Proof that fp32 addition is not associative.

Reproduces the hand derivation: a = 2^24, b = c = 1.
Expected: (a+b)+c = 16777216  but  a+(b+c) = 16777218.

Run:  python assoc_check.py
"""

import numpy as np


def demo(a, b, c):
    """Compare the two groupings for one triple of fp32 values."""
    # np.float32(...) forces every value and every intermediate into fp32.
    # Without it, Python would compute in fp64 and the effect would vanish.
    a = np.float32(a)
    b = np.float32(b)
    c = np.float32(c)

    left  = np.float32(np.float32(a + b) + c)   # (a + b) + c
    right = np.float32(a + np.float32(b + c))   # a + (b + c)

    print(f"a = {a:.0f}, b = {b:.0f}, c = {c:.0f}")
    print(f"  (a + b) + c = {left:.0f}")
    print(f"  a + (b + c) = {right:.0f}")
    print(f"  difference  = {np.float32(left - right):.0f}")
    print(f"  associative? {bool(left == right)}")
    print()
    return left, right


if __name__ == "__main__":
    # The hand-derived case: 2^24 is where fp32's ULP becomes 2.
    left, right = demo(2**24, 1, 1)

    # assert turns the claim into a test: the script fails loudly if the
    # behaviour ever changes (e.g. run on a machine with different defaults).
    assert left == np.float32(16777216), "left grouping unexpected"
    assert right == np.float32(16777218), "right grouping unexpected"
    assert left != right, "expected non-associativity, got equality"

    print("Verified: fp32 addition is not associative for this case.")

"""
Output:
a = 16777216, b = 1, c = 1
  (a + b) + c = 16777216
  a + (b + c) = 16777218
  difference  = -2
  associative? False

Verified: fp32 addition is not associative for this case.
"""
