from __future__ import division, absolute_import, print_function
from .base import ImportanceScore
import numpy as np
from kipoi.data_utils import numpy_collate_concat, numpy_collate
from kipoi_interpret.utils import get_model_input, set_model_input
from kipoi.data_utils import get_dataset_item
from kipoi_interpret.utils import apply_within
from .ism_scores import Score
import copy


def parse_score_str(score, **init_kwargs):
    if isinstance(score, Score):
        return score
    else:
        return Score.from_string(score)(**init_kwargs)


def parse_scores(scores, score_kwargs):
    if score_kwargs is not None:
        if not len(scores) == len(score_kwargs):
            raise Exception("If score_kwargs is not none then at least an empty dictionary has to be passed for "
                            "every entry in `scores`.")
        return [parse_score_str(score, **kwargs) for score, kwargs in zip(scores, score_kwargs)]
    else:
        return [parse_score_str(score) for score in scores]


class Mutation(ImportanceScore):
    """ISM for working with one-hot encoded inputs. 
    """

    def __init__(self, model, model_input, scores=['diff'], score_kwargs=None, batch_size=32, output_sel_fn = None,
                 category_dim = 1):
        """
        Args:
          model: Kipoi model
          model_input: which model input to mutate
          scores: a list of score names or score instances
          batch_size: batch size for calls to prediction. This is independent from the size of batch
            used with the `score` method.
          score_kwargs: Initialisation keyword arguments for `scores`. If not None then it is a list of kwargs 
            dictionaries of the same length as `scores`
          output_sel_fn: Function used to select a model output. Only the selected output will be reported as a return 
            value.
          category_dim: Dimension in which the the one-hot category is stored. e.g. for a one-hot encoded DNA-sequence
            array with input shape (1000, 4) for a single sample, `category_dim` is 1, for (4, 1000) `category_dim`
            is 0. In the given dimension only one value is allowed to be non-zero, which is the selected one.
        """
        self.model = model
        self.model_input = model_input
        self.batch_size = batch_size
        self.scores = parse_scores(scores, score_kwargs)
        self.output_sel_fn = output_sel_fn
        self.category_dim = category_dim

    def is_compatible(self, model):
        return True

    def mutate_sample(self, onehot_input):
        it = np.ndenumerate(onehot_input)
        for idx, in_val in it:
            if in_val == 1:
                continue
            zero_sel = list(idx)
            zero_sel[self.category_dim] = slice(None)
            output_onehot = onehot_input.copy()
            output_onehot.__setitem__(tuple(zero_sel), 0)
            output_onehot.__setitem__(idx, 1)
            yield output_onehot, idx

    def mutate_sample_batched(self, onehot_input):
        return_samples = []
        return_indexes = []
        for mutated, indexes in self.mutate_sample(onehot_input):
            return_samples.append(mutated)
            return_indexes.append(indexes)
            if len(return_samples) == self.batch_size:
                yield return_samples, return_indexes
                return_samples = []
                return_indexes = []
        if len(return_samples) != 0:
            yield return_samples, return_indexes


    def score(self, input_batch):
        """
        Args:
          input_batch: Input batch that should be scored.
                
        Returns:
          A list of length: len(`scores`). Every element of the list is a stacked list of depth D if the model input
          is D-dimensional with identcal shape. Every entry of that list then contains the scores of the model output 
          selected by `output_sel_fn`. Values are `None` if the input_batch already had a `1` at that position.
        """

        ref = self.model.predict_on_batch(input_batch)
        scores = []
        for sample_i in range(input_batch[self.model_input].shape[0]):
            # get the full set of model inputs for the selected sample
            sample_set = get_dataset_item(input_batch, sample_i)
            # get the reference output for this sample
            ref_sample_pred = get_dataset_item(ref, sample_i)
            # Apply the output selection function if defined
            if self.output_sel_fn is not None:
                ref_sample_pred = self.output_sel_fn(ref_sample_pred)
            # get the one-hot encoded reference input array
            input_sample = get_model_input(sample_set, input_id=self.model_input)
            # where we keep the scores - scores are lists (ordered by diff method) of ndarrays, lists or dictionaries - whatever is returned by the model
            score = np.empty(input_sample.shape, dtype=object)
            score[:] = None
            for alt_batch, alt_idxs in self.mutate_sample_batched(input_sample):
                #
                num_samples = len(alt_batch)
                mult_set = numpy_collate([sample_set]*num_samples)
                mult_set = set_model_input(mult_set, numpy_collate(alt_batch), input_id=self.model_input)
                alt = self.model.predict_on_batch(mult_set)
                for alt_sample_i in range(num_samples):
                    alt_sample = get_dataset_item(alt, alt_sample_i)
                    # Apply the output selection function if defined
                    if self.output_sel_fn is not None:
                        alt_sample = self.output_sel_fn(alt_sample)
                    # Apply scores across all model outputs for ref and alt
                    output_scores = [apply_within(ref_sample_pred, alt_sample, scr) for scr in self.scores]
                    score.__setitem__(alt_idxs[alt_sample_i], output_scores)
            scores.append(score.tolist())

        return scores

    # def score_from_cli(self, input_onehot_fasta_data_loader_config,
    #                    io_batch_size=100,
    #                    **other_kwargs):
    #     # read the data to create input_onehot_fasta
    #     # call score_func accordingly
    #     assert False

# 'zero': (DummyZero, 0),
# 'saliency': (Saliency, 1),
# 'grad*input': (GradientXInput, 2),
# 'intgrad': (IntegratedGradients, 3),
# 'elrp': (EpsilonLRP, 4),
# 'deeplift': (DeepLIFTRescale, 5),
# 'occlusion': (Occlusion, 6)


class Occlusion(ImportanceScore):

    def __init__(self, model, model_input, scores=['diff']):
        """
        Args:
          model: Kipoi model
          model_input: which model input to mutate
          scores: a list of score names or score instances
        """
        self.model = model
        self.model_input = model_input
        self.scores = parse_scores(scores)

    def is_compatible(self, model):
        # ISM is compatible with all the models
        return True

    def score(self, input_batch):
        # TODO - loop through all the possible elements of the array and set them to zero
        def get_mask(idx, shape):
            ones = np.ones(shape)
            ones[idx] = 0
            return ones

        input_batch[self.model_input]
        pass


METHODS = {"occlusion": Occlusion,
           "mutation": Mutation}
