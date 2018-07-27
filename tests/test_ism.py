import numpy as np
import pytest
from kipoi_interpret.importance_scores.ism import Mutation
import kipoi
from kipoi.specs import NestedMappingField, ArraySchema
from related import from_yaml
import related
from kipoi.external.related.mixins import RelatedConfigMixin
from kipoi.external.flatten_json import flatten


class DummyContainer():
    pass

class DummyModel():
    def predict_on_batch(self, in_batch):
        return in_batch


@related.immutable(strict=True)
class DummyModelSchema(RelatedConfigMixin):
    """Describes the model schema
    """
    # can be a dictionary, list or a single array
    inputs = NestedMappingField(ArraySchema, keyword="shape", key="name")


input_examples = {'dict':"""
inputs:
    dat1:
        shape: (4, None)
        special_type: DNASeq
""",
'list':"""
inputs:
    - name: dat1
      shape: (4, None)
      special_type: DNASeq
""",
'arr':"""
inputs:
    name: dat1
    shape: (4, None)
    special_type: DNASeq
"""}

def get_dummy_model(input_schema = "dict"):
    dm = DummyModel()
    dm.schema = DummyContainer()
    dm.schema.inputs = DummyModelSchema.from_config(from_yaml(input_examples[input_schema])).inputs
    return dm


arr_elm = np.array([[[0, 0, 1, 0], [0, 1, 0, 0]]])
batch_inputs = {'dict': {"dat1": arr_elm},
                'list': [arr_elm],
                'arr': arr_elm}

def test_get_correct_model_input():
    for model_input_type in input_examples:
        # define batch input data:
        batch_input = batch_inputs[model_input_type]
        m = Mutation(get_dummy_model(model_input_type), "dat1", ['diff'])
        idx = m.get_correct_model_input_id('dat1')
        if model_input_type == "dict":
            assert idx == 'dat1'
        else:
            assert idx == 0


def test_score():
    for model_input_type in input_examples:
        # define batch input data:
        batch_input = batch_inputs[model_input_type]
        m = Mutation(get_dummy_model(model_input_type), "dat1", ['diff'])
        scores_ret = m.score(batch_input)
        # expected output:
        for smpl_i, smpl in enumerate(scores_ret):
            for i in range(len(smpl)):
                for j in range(len(smpl[i])):
                    bi = batch_input
                    if model_input_type == "dict":
                        bi = batch_input['dat1']
                    elif model_input_type == "list":
                        bi = batch_input[0]
                    exp = (np.arange(0, 4) == j).astype(int) - bi[smpl_i, i, :]
                    if np.all(exp == 0):
                        assert smpl[i][j] is None
                    else:
                        smpl_diff = smpl[i][j][0]  # select the score i,j and the score 0 which is 'diff' here
                        model_out_diff = smpl_diff
                        if model_input_type == "dict":
                            model_out_diff = smpl_diff['dat1']
                        elif model_input_type == "list":
                            model_out_diff = smpl_diff[0]
                        assert np.all(model_out_diff[i, :] == exp)
        # test with selector_fn
        if model_input_type == "dict":
            sel_fn = lambda x: x['dat1']
        elif model_input_type == "list":
            sel_fn = lambda x: x[0]
        else:
            sel_fn = lambda x: x
        m = Mutation(get_dummy_model(model_input_type), "dat1", ['diff'], output_sel_fn=sel_fn)
        scores_ret = m.score(batch_input)
        for smpl_i, smpl in enumerate(scores_ret):
            for i in range(len(smpl)):
                for j in range(len(smpl[i])):
                    bi = batch_input
                    if model_input_type == "dict":
                        bi = batch_input['dat1']
                    elif model_input_type == "list":
                        bi = batch_input[0]
                    exp = (np.arange(0, 4) == j).astype(int) - bi[smpl_i, i, :]
                    if np.all(exp == 0):
                        assert smpl[i][j] is None
                    else:
                        smpl_diff = smpl[i][j][0]  # select the score i,j and the score 0 which is 'diff' here
                        model_out_diff = smpl_diff
                        assert np.all(model_out_diff[i, :] == exp)
        # test the test_ref_ref functionality
        m = Mutation(get_dummy_model(model_input_type), "dat1", ['diff'], test_ref_ref=True)
        scores_ret = m.score(batch_input)
        # expected output:
        for smpl_i, smpl in enumerate(scores_ret):
            for i in range(len(smpl)):
                for j in range(len(smpl[i])):
                    bi = batch_input
                    if model_input_type == "dict":
                        bi = batch_input['dat1']
                    elif model_input_type == "list":
                        bi = batch_input[0]
                    exp = (np.arange(0, 4) == j).astype(int) - bi[smpl_i, i, :]
                    # if test_ref_ref is set the score has to be returned for all!
                    smpl_diff = smpl[i][j][0]  # select the score i,j and the score 0 which is 'diff' here
                    model_out_diff = smpl_diff
                    if model_input_type == "dict":
                        model_out_diff = smpl_diff['dat1']
                    elif model_input_type == "list":
                        model_out_diff = smpl_diff[0]
                    assert np.all(model_out_diff[i, :] == exp)


def test_mutate():
    example = np.array([[0, 0, 1, 0], [0, 1, 0, 0]])
    m = Mutation(get_dummy_model(), "dat1", ['diff'])
    for ret, idxs in m._mutate_sample(example):
        assert example[idxs[0], idxs[1]] == 0
        kept_sel = np.arange(example.shape[0]) != idxs[0]
        sel_j = np.arange(example.shape[1]) == idxs[1]
        assert np.all(example[kept_sel, :] == ret[kept_sel, :])
        assert np.all(ret[~kept_sel, sel_j] == 1)
        assert np.all(ret[~kept_sel, ~sel_j] != 1)


def test_is_compatible():
    is_compat_model = kipoi.get_model("tests/models/pyt", "dir")
    m = Mutation(is_compat_model, "dat1", ['diff'])
    assert m.is_compatible(is_compat_model)
