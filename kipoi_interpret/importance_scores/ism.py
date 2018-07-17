from __future__ import division, absolute_import, print_function
from .base import ImportanceScore
import numpy as np
from kipoi.data_utils import numpy_collate_concat, numpy_collate
from kipoi_interpret.utils import get_model_input, set_model_input
from kipoi.data_utils import get_dataset_item
from kipoi_interpret.utils import apply_within, get_model_input_special_type
from kipoi.components import ArraySpecialType
from .ism_scores import Score


def parse_score_str(score, **init_kwargs):
    if isinstance(score, Score):
        return score
    else:
        return Score.from_string(score)(**init_kwargs)


def parse_scores(scores):
    return [parse_score_str(score) for score in scores]


class Mutation(ImportanceScore):
    """ISM for working with one-hot encoded inputs
    """

    def __init__(self, model, model_input, scores=['diff'], score_kwargs=None, batch_size=32, output_sel_fn = None):
        """
        Args:
          model: Kipoi model
          model_input: which model input to mutate
          scores: a list of score names or score instances
          batch_size: batch size for calls to prediction. This is independent from the size of batch
            used with the `score` method.
        """
        self.model = model
        self.model_input = model_input
        self.batch_size = batch_size
        self.scores = parse_scores(scores)
        self.output_sel_fn = output_sel_fn

    def is_compatible(self):
        return get_model_input_special_type(self.model, self.model_input) is ArraySpecialType.DNASeq

    @staticmethod
    def mutate_sample(onehot_input):
        for i in range(onehot_input.shape[0]):
            for j in range(onehot_input.shape[1]):
                if onehot_input[i, j] == 1:
                    continue
                output_onehot = onehot_input.copy()
                output_onehot[i, :] = 0
                output_onehot[i, j] = 1
                yield output_onehot, (i, j,)

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
        # perturb the sequences according to some rule
        # call score_func to get new prediction on perturbed seqs
        # compile the stuff together to get the scores

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
            score = [[None for _2 in range(input_sample.shape[1])] for _ in range(input_sample.shape[0])]
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
                    ### TODO: Implement a function that 1) will select a score 2) can summarise the model output
                    ### TODO: (ctd.) to a single value! For now just return as is
                    idx_i, idx_j = alt_idxs[alt_sample_i]
                    score[idx_i][idx_j] = output_scores
            scores.append(score)

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
