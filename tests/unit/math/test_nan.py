from src.math.nan import nan

def test_nan_ops():
    assert 1 + nan() == nan()
    assert 1 - nan() == nan()
    assert 1 * nan() == nan()
    assert 1 / nan() == nan()
    assert 1 ** nan() == nan()

    assert nan() + 1 == nan()
    assert nan() - 1 == nan()
    assert nan() * 1 == nan()
    assert nan() / 1 == nan()
    assert nan() ** 1 == nan()

    val = nan()
    val += 1
    assert val == nan()
    val -= 1
    assert val == nan()
    val *= 1
    assert val == nan()
    val /= 1
    assert val == nan()

    assert abs(nan()) == nan()
    assert -nan() == nan()

