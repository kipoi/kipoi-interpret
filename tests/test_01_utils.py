from kipoi_interpret.utils import apply_within
import numpy as np
from collections import OrderedDict
import pytest

def test_apply_within():
    values = [np.array([1, 2, 3]), np.array([5, 6, 7])]
    fn = lambda x,y: x+y
    values_exp = [el*2 for el in values]
    a = OrderedDict()
    a_exp = OrderedDict()
    for k, val, val_exp in zip(['first', 'second'], values, values_exp):
        a[k] = val
        a_exp[k] = val_exp

    # general functionality:
    a_ret = apply_within(a, a, fn)
    assert set(a.keys()) == set(a_ret.keys())
    assert all([np.all(a_ret[k] == a_exp[k]) for k in a_exp])

    values_ret = apply_within(values, values, fn)
    assert len(values_ret) == len(values_exp)
    assert all([np.all(el1 == el2) for el1, el2 in zip(values_ret, values_exp)])

    assert np.all(values_exp[0] == apply_within(values[0], values[0], fn))

    # type mismatch error:
    with pytest.raises(Exception):
        _ = apply_within(a, values, fn)

    # list length mismatch error:
    with pytest.raises(Exception):
        _ = apply_within(values[:1], values, fn)

    # dict key mismatch error
    with pytest.raises(Exception):
        b = OrderedDict()
        b['first'] = a['first']
        _ = apply_within(a, b, fn)