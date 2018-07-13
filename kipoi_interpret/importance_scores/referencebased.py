from __future__ import division, absolute_import, print_function
from .base import ImportanceScoreWRef
from .gradient import Gradient

import tempfile
import numpy as np


# Other proposal (this object is passed as an argument to compile())
class DeepLift(ImportanceScoreWRef):

    def __init__(self, model, output_layer,
                 task_idx, preact, mxts_mode = 'deeplift'):

        import deeplift
        from deeplift.conversion import kerasapi_conversion as kc
        from deeplift.layers import NonlinearMxtsMode

        def get_mxts_mode(mode_name):
            mxts_modes = {'deeplift': NonlinearMxtsMode.DeepLIFT_GenomicsDefault,
                          'revealcancel': NonlinearMxtsMode.RevealCancel,
                          'rescale': NonlinearMxtsMode.Rescale,
                          'gradient': NonlinearMxtsMode.Gradient,
                          'guidedbackprop': NonlinearMxtsMode.GuidedBackprop}
            return mxts_modes[mode_name]

        # TODO: create and return the deeplift func, which
        # takes arguments "input_data_list" and "input_references_list"
        self.model = model
        if not self.is_compatible(model):
            raise Exception("Model not compatible with DeepLift")

        self.task_idx = task_idx


        weight_f = tempfile.mktemp()
        arch_f = tempfile.mktemp()
        model.model.save_weights(weight_f)
        with open(arch_f, "w") as ofh:
            ofh.write(model.model.to_json())

        self.deeplift_model = kc.convert_model_from_saved_files(weight_f, json_file=arch_f,
                                                                nonlinear_mxts_mode=get_mxts_mode(mxts_mode))

        # Compile the function that computes the contribution scores
        # For sigmoid or softmax outputs, target_layer_idx should be -2 (the default)
        # (See "3.6 Choice of target layer" in https://arxiv.org/abs/1704.02685 for justification)
        # For regression tasks with a linear output, target_layer_idx should be -1
        # (which simply refers to the last layer)
        # If you want the DeepLIFT multipliers instead of the contribution scores, you can use get_target_multipliers_func
        self.deeplift_contribs_func = self.deeplift_model.get_target_contribs_func(
            find_scores_layer_idx=0,
            target_layer_idx=output_layer)


    @classmethod
    def is_compatible(cls, model):
        # TODO: implement check for required functions
        # specifically, a "save_in_keras2" func that saves in the keras2
        # format, and also test that the conversion works

        if model.type != "keras":
            # Support only keras models
            return False
        # Check the Keras backend
        import keras.backend as K

        # TODO - check the Keras version
        if model.backend is None:
            backend = K.backend()
        else:
            backend = model.backend
        return backend == "tensorflow"

    def score(self, input_batch, input_ref):
        x_standardized = self.model._batch_to_list(input_batch)

        # Taken from DeepLIFT github readme
        scores = self.deeplift_contribs_func(task_idx=self.task_idx,
                                                 input_data_list=x_standardized,
                                                 batch_size=10,
                                                 progress_update=1000)


        # re-format the list-type input back to how the input_batch was:
        scores = self.model._match_to_input(scores, input_batch)
        return scores


class IntegratedGradients(ImportanceScoreWRef, Gradient):

    def score(self, input_batch, input_ref):
        grads = super().score(input_batch)
        # TODO implement integrated gradients
        # https://github.com/marcoancona/DeepExplain/blob/master/deepexplain/tensorflow/methods.py#L208-L225
        pass


class GradientXInput(ImportanceScoreWRef, Gradient):
    # AbstractGrads implements the

    def score(self, input_batch, input_ref):
        # TODO - handle also the case where input_ref is not a simple array but a list
        return super().score(input_batch) * input_ref


METHODS = {"deeplift": DeepLift,
           "grad*input": GradientXInput,
           "intgrad": IntegratedGradients}
