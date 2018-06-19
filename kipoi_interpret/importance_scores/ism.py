from __future__ import division, absolute_import, print_function
from .common import ImportanceScore
import numpy as np


def parse_score_str(score):
    # TODO - implement
    if isinstance(score, Score):
        return score
    else:
        return Score.from_string(score)


def parse_scores(scores):
    return [parse_score_str(score) for score in scores]


class Mutation(ImportanceScore):
    """ISM for working with one-hot encoded inputs
    """

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
        for inp in self.model.inputs:
            if isinstance(inp, "OneHot"):
                # TODO - check if the model has a one-hot encoded input
                return True
        return False

    def score(self, input_batch):
        # perturb the sequences according to some rule
        # call score_func to get new prediction on perturbed seqs
        # compile the stuff together to get the scores
        # TODO: implement
        assert False

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
