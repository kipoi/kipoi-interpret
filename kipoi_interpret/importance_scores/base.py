from __future__ import division, absolute_import, print_function
from tqdm import tqdm
from kipoi.utils import merge_dicts
from kipoi.data_utils import numpy_collate_concat
import abc
# Abstract declaration of different interpretation APIs, plus any
# common functionality


# Abstract importance score
class ImportanceScore(object):

    @classmethod
    def is_compatible(self, model):
        """Test if the model is compatible with the API
        """
        raise NotImplementedError()

    def score(self, input_batch):
        """Score a particular example
        """
        raise NotImplementedError()


# Importance score that requires also the reference
class ImportanceScoreWRef(ImportanceScore):

    @abc.abstractmethod
    def score(self, input_batch, input_ref):
        # read the data loaders into arrays as needed
        # instantiate the reference generator using the kwargs
        # call score_func accordingly
        # TODO: implement
        raise NotImplementedError()


# --------------------------------------------

def available_methods():
    """Get all available importance scores
    """
    from . import ism, gradient, referencebased
    int_modules = [ism, gradient, referencebased]

    available_methods = {}
    for m in int_modules:
        available_methods = merge_dicts(available_methods, m.METHODS)
    return available_methods


def get_importance_score(importance_score):
    """Get the importance score
    """
    if isinstance(importance_score, ImportanceScore):
        return importance_score
    methods = available_methods()
    if importance_score not in methods:
        raise ValueError("Importance score {0} not found. Valid scores are: {1}".
                         format(importance_score, methods))
    return methods[importance_score]


# --------------------------------------------


def feature_importance(model,
                       dataloader,
                       importance_score,
                       importance_score_kwargs={},
                       batch_size=32,
                       num_workers=0):
    """Return feature importance scores

    # Arguments
        model: kipoi model (obtained by `kipoi.get_model()`)
        dataloader: instantiated kipoi dataloder (obtained by `kipoi.get_dataloader_factory()(**dl_kwargs)`
           or `model.default_dataloader(**dl_kwargs)`
        importance_score (`str` or `ImportanceScore`): which importance score to use
        importance_score_kwargs (dict): kwargs passed to the importance score
        batch_size: run scoring and data-loading in batches
        num_workers: number of workers for parallel data-loading. Passed to `dataloader.batch_iter(...)`

    # Returns
        (dict of np.arrays): dataset returned by the dataloader (dict with keys `inputs`, `targets`, `metadata`)
           but with an additional `importance_scores` key 

    """
    ImpScore = get_importance_score(importance_score)
    if not ImpScore.is_compatible(model):
        raise ValueError("model not compatible with score: {0}".format(importance_score))
    impscore = ImpScore(model, **importance_score_kwargs)

    def append_key(d, k, v):
        d[k] = v
        return d

    # TODO - handle the reference-based importance scores...
    return numpy_collate_concat([append_key(batch, "importance_scores", impscore.score(batch['inputs']))
                                 for batch in tqdm(dataloader.batch_iter(batch_size=batch_size,
                                                                         num_workers=num_workers))])
