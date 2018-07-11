import numpy as np
import collections
from collections import OrderedDict

def get_model_input_special_type(model, input_id):
    inputs = model.schema.inputs
    if isinstance(inputs, collections.Mapping) or isinstance(inputs, collections.Sequence):
        return inputs[input_id].special_type
    else:
        return inputs.special_type

def get_model_input(batch, input_id=None):
    """
    Get model input from batch
    batch: batch of model input samples
    """
    if isinstance(batch, dict) or isinstance(batch, list):
        assert input_id is not None
        return batch[input_id]
    else:
        return batch



def set_model_input(batch, value, input_id=None):
    if isinstance(batch, dict) or isinstance(batch, list):
        assert input_id is not None
        batch[input_id] = value
        return batch
    else:
        return value



def apply_within(data1, data2, function, **kwargs):
    if not type(data1) == type(data2):
        raise Exception("data1 and data2 have to be of the same type!")
    if isinstance(data1, dict):
        if not set(data1.keys()) == set(data2.keys()):
            raise Exception("data1 and data2 dictionaries have to have the same keys!")
        out = {}
        if isinstance(data1, OrderedDict):
            out = OrderedDict()
        for k in data1:
            out[k] = apply_within(data1[k], data2[k], function, **kwargs)
    elif isinstance(data1, list):
        if not len(data1) == len(data2):
            raise Exception("data1 and data2 lists have to have the same length!")
        out = []
        for el1, el2 in zip(data1, data2):
            out.append(apply_within(el1, el2, function, **kwargs))
    else:
        out = function(data1, data2, **kwargs)
    return out
