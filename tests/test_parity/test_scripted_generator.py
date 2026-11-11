"""The ScriptedGenerator that replays R's random numbers in the L1 tests.

It must serve each pool in order and in the requested shape, refuse to serve
more than was scripted, log every call with the arguments that shape the
draw, and, in strict mode, refuse every draw that no pool serves.
"""

import numpy as np
import pytest

from ._helpers import ScriptedGenerator

pytestmark = pytest.mark.parity


# ---------------------------------------------------------------------------
# the four pools
# ---------------------------------------------------------------------------


def test_chisq_pool_in_order_and_shape():
    rng = ScriptedGenerator(chisq=[3.0, 4.0, 5.0, 6.0])
    first = rng.chisquare(2)
    assert np.ndim(first) == 0 and first == 3.0
    np.testing.assert_array_equal(rng.chisquare(2, size=3), [4.0, 5.0, 6.0])
    assert rng.unused == (0, 0)


def test_normal_pool_in_order_and_shape():
    rng = ScriptedGenerator(normals=[0.1, 0.2, 0.3, 0.4, 0.5, -1.0, 2.0])
    out = rng.standard_normal((2, 2))
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, [[0.1, 0.2], [0.3, 0.4]])
    assert rng.standard_normal() == 0.5
    np.testing.assert_array_equal(rng.normal(loc=1.0, scale=2.0, size=2), [1.0 + 2.0 * -1.0, 1.0 + 2.0 * 2.0])
    assert rng.unused == (0, 0)


def test_uniform_pool_in_order_and_shape():
    rng = ScriptedGenerator(uniforms=[0.25, 0.5, 0.75, 0.1, 0.9, 0.2, 0.3])
    np.testing.assert_array_equal(rng.random(3), [0.25, 0.5, 0.75])
    assert rng.random() == 0.1
    assert rng.uniform(2.0, 4.0) == 2.0 + (4.0 - 2.0) * 0.9
    out = rng.uniform(size=(2, 1))
    assert out.shape == (2, 1)
    np.testing.assert_array_equal(out, [[0.2], [0.3]])
    rng.assert_exhausted()


def test_integer_pool_in_order_and_shape():
    rng = ScriptedGenerator(integers=[4, 0, 2, 3, 1])
    out = rng.integers(0, 5, size=3)
    assert out.dtype == np.int64 and out.shape == (3,)
    np.testing.assert_array_equal(out, [4, 0, 2])
    single = rng.integers(4)
    assert np.ndim(single) == 0 and single == 3
    np.testing.assert_array_equal(rng.integers(1, 1, size=(1, 1), endpoint=True), [[1]])
    rng.assert_exhausted()


@pytest.mark.parametrize(
    "pools, draw",
    [
        ({"chisq": [1.0]}, lambda rng: rng.chisquare(2, size=2)),
        ({"normals": [0.1, 0.2]}, lambda rng: rng.standard_normal(3)),
        ({}, lambda rng: rng.normal()),
        ({"uniforms": [0.5]}, lambda rng: rng.uniform(size=2)),
        ({"uniforms": []}, lambda rng: rng.random()),
        ({"integers": [0, 1]}, lambda rng: rng.integers(0, 2, size=3)),
    ],
    ids=["chisquare", "standard_normal", "normal", "uniform", "random", "integers"],
)
def test_asking_for_more_than_is_left_raises(pools, draw):
    rng = ScriptedGenerator(**pools)
    with pytest.raises(AssertionError, match="scripted left"):
        draw(rng)


@pytest.mark.parametrize(
    "value, low, high, endpoint",
    [(5, 0, 5, False), (-1, 0, 5, False), (0, 1, 4, False), (6, 1, 5, True)],
)
def test_scripted_integer_outside_the_requested_range_raises(value, low, high, endpoint):
    rng = ScriptedGenerator(integers=[value])
    with pytest.raises(AssertionError, match="outside the requested range"):
        rng.integers(low, high, endpoint=endpoint)


def test_scripted_integer_outside_range_of_one_argument_call_raises():
    rng = ScriptedGenerator(integers=[3])
    with pytest.raises(AssertionError, match=r"outside the requested range \[0, 3\)"):
        rng.integers(3)


# ---------------------------------------------------------------------------
# the call log
# ---------------------------------------------------------------------------


def test_calls_are_logged_with_their_arguments():
    rng = ScriptedGenerator(chisq=[2.5], normals=[0.1, 0.2, 0.3], uniforms=[0.3, 0.6], integers=[1, 0, 2, 5])
    rng.chisquare(7)
    rng.standard_normal(1)
    rng.normal(1.0, 2.0, size=2)
    rng.random()
    rng.uniform(-1.0, 1.0)
    rng.integers(0, 2, size=2)
    rng.integers(3)
    rng.integers(5, 5, endpoint=True)
    assert rng.calls == [
        ("chisquare", 1, {"df": 7}),
        ("standard_normal", 1, {}),
        ("normal", 2, {"loc": 1.0, "scale": 2.0}),
        ("random", 1, {}),
        ("uniform", 1, {"low": -1.0, "high": 1.0}),
        ("integers", 2, {"low": 0, "high": 2, "endpoint": False}),
        ("integers", 1, {"low": 0, "high": 3, "endpoint": False}),
        ("integers", 1, {"low": 5, "high": 5, "endpoint": True}),
    ]
    first = rng.calls[0]
    assert (first.method, first.n, first.args["df"]) == ("chisquare", 1, 7)


def test_order_merges_consecutive_calls_on_one_pool():
    rng = ScriptedGenerator(chisq=[1.0], normals=np.zeros(6), uniforms=np.full(4, 0.5), integers=[0, 1, 2, 3])
    rng.chisquare(3)
    rng.standard_normal(3)
    rng.normal(size=2)  # the same pool as standard_normal
    rng.random(2)
    rng.uniform()
    rng.standard_normal()
    rng.integers(0, 4, size=4)
    rng.random()
    assert rng.order == [("chisq", 1), ("normal", 5), ("uniform", 3), ("normal", 1), ("integer", 4), ("uniform", 1)]
    rng.assert_exhausted()


def test_assert_exhausted_names_what_is_left():
    rng = ScriptedGenerator(chisq=[1.0], normals=[0.1, 0.2], uniforms=[0.5], integers=[0, 1, 2])
    assert rng.unused == (1, 2)
    with pytest.raises(AssertionError, match=r"^scripted values left over: 1 chisq, 2 normal, 1 uniform, 3 integer$"):
        rng.assert_exhausted()
    rng.chisquare(1)
    rng.standard_normal(2)
    rng.random()
    rng.integers(0, 3, size=2)
    with pytest.raises(AssertionError, match=r"^scripted values left over: 1 integer$"):
        rng.assert_exhausted()
    rng.integers(0, 3)
    rng.assert_exhausted()
    ScriptedGenerator().assert_exhausted()  # nothing scripted


# ---------------------------------------------------------------------------
# unscripted draws
# ---------------------------------------------------------------------------


def test_unscripted_draws_come_from_pcg64():
    """Without ``strict``, other methods and the methods of a pool that was not
    given draw from PCG64 with ``seed``, and nothing is logged."""
    rng = ScriptedGenerator(chisq=[1.0], seed=3)
    reference = np.random.Generator(np.random.PCG64(3))
    np.testing.assert_array_equal(rng.choice(10, size=5), reference.choice(10, size=5))
    np.testing.assert_array_equal(rng.permutation(6), reference.permutation(6))
    np.testing.assert_array_equal(rng.random(3), reference.random(3))
    assert rng.uniform(2.0, 3.0) == reference.uniform(2.0, 3.0)
    np.testing.assert_array_equal(rng.integers(0, 9, size=4), reference.integers(0, 9, size=4))
    assert rng.calls == []
    assert rng.chisquare(4) == 1.0


@pytest.mark.parametrize(
    "name, draw",
    [
        ("choice", lambda rng: rng.choice(3)),
        ("permutation", lambda rng: rng.permutation(3)),
        ("binomial", lambda rng: rng.binomial(1, 0.5)),
        ("random", lambda rng: rng.random()),
        ("uniform", lambda rng: rng.uniform()),
        ("integers", lambda rng: rng.integers(0, 3)),
    ],
    ids=["choice", "permutation", "binomial", "random", "uniform", "integers"],
)
def test_strict_refuses_unscripted_draws(name, draw):
    rng = ScriptedGenerator(chisq=[1.0], normals=[0.5], strict=True)
    with pytest.raises(AssertionError, match=rf"rng\.{name}\(\) is not scripted"):
        draw(rng)


def test_strict_serves_the_scripted_pools():
    rng = ScriptedGenerator(chisq=[1.0], normals=[0.5], uniforms=[0.25], integers=[2], strict=True)
    assert isinstance(rng.bit_generator, np.random.PCG64)
    assert rng.chisquare(3) == 1.0
    assert rng.standard_normal() == 0.5
    assert rng.random() == 0.25
    assert rng.integers(0, 3) == 2
    rng.assert_exhausted()
