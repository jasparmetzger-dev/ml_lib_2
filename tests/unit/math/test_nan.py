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
    assert val.__ipow__(1) == nan()

    one = 1
    one /= nan()
    assert one == nan()

    assert abs(nan()) == nan()
    assert -nan() == nan()

def test_nan_repr():
    assert nan().__repr__() == 'nan'

def test_nan_comparison():
    assert (nan() > 1) == False
    assert (nan() >= 1) == False
    assert (nan() < 1) == False
    assert (nan() <= 1) == False

    assert (1 > nan()) == False
    assert (1 >= nan()) == False
    assert (1 < nan()) == False
    assert (1 <= nan()) == False
