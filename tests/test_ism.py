import numpy as np
import pytest
from kipoi_interpret.importance_scores.ism import Mutation
import kipoi

class DummyContainer():
    pass

class DummyModel():
    def predict_on_batch(self, in_batch):
        return in_batch

def get_dummy_model():
    dm = DummyModel()
    dm.schema = DummyContainer()
    dm.schema.inputs = {"dat1"}
    return dm


def test_score():
    # define batch input data:
    batch_input = {"dat1":np.array([[[0,0,1,0], [0,1,0,0]]])} # 1 sample, seqlen 2, onehot encoded
    m = Mutation(get_dummy_model(), "dat1", ['diff'])
    scores_ret = m.score(batch_input)
    # expected output:
    for smpl_i, smpl in enumerate(scores_ret):
        for i in range(len(smpl)):
            for j in range(len(smpl[i])):
                exp =  (np.arange(0, 4) == j).astype(int) - batch_input['dat1'][smpl_i, i,:]
                if np.all(exp==0):
                    assert smpl[i][j] is None
                else:
                    smpl_diff = smpl[i][j][0]  # select the score i,j and the score 0 which is 'diff' here
                    model_out_diff = smpl_diff['dat1']
                    assert np.all(model_out_diff[i,:] == exp)
    # test with selector_fn
    sel_fn = lambda x: x['dat1']
    m = Mutation(get_dummy_model(), "dat1", ['diff'], output_sel_fn=sel_fn)
    scores_ret = m.score(batch_input)
    for smpl_i, smpl in enumerate(scores_ret):
        for i in range(len(smpl)):
            for j in range(len(smpl[i])):
                exp =  (np.arange(0, 4) == j).astype(int) - batch_input['dat1'][smpl_i, i,:]
                if np.all(exp==0):
                    assert smpl[i][j] is None
                else:
                    smpl_diff = smpl[i][j][0]  # select the score i,j and the score 0 which is 'diff' here
                    model_out_diff = smpl_diff
                    assert np.all(model_out_diff[i,:] == exp)


def test_mutate():
    example = np.array([[0,0,1,0], [0,1,0,0]])
    m = Mutation(get_dummy_model(), "dat1", ['diff'])
    for ret, idxs in m.mutate_sample(example):
        assert example[idxs[0], idxs[1]] == 0
        kept_sel = np.arange(example.shape[0]) != idxs[0]
        sel_j = np.arange(example.shape[1]) == idxs[1]
        assert np.all(example[kept_sel, :] == ret[kept_sel, :])
        assert np.all(ret[~kept_sel, sel_j] == 1)
        assert np.all(ret[~kept_sel, ~sel_j] != 1)


def test_is_compatible():
    is_compat_model = kipoi.get_model("DeepSEA/variantEffects")
    m = Mutation(is_compat_model, "dat1", ['diff'])
    assert m.is_compatible(is_compat_model)
