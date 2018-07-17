from __future__ import division, absolute_import, print_function
from .base import ImportanceScoreWRef
from .gradient import Gradient
from kipoi.model import KerasModel, TensorFlowModel
from kipoi.data_utils import numpy_collate

import tempfile
import numpy as np


# Other proposal (this object is passed as an argument to compile())
class DeepLift(ImportanceScoreWRef):
    """
    Wrapper around DeepLIFT
    """

    def __init__(self, model, output_layer,
                 task_idx, preact, mxts_mode='rescale_conv_revealcancel_fc',
                 batch_size=32):
        """
        Args:
          model: Kipoi model
          output_layer (int): selected Keras layer with respect to which the scores should be calculated
          task_idx (int): Node/Neuron within the selected layer with respect to which the score should be calculated
          preact: !NOT YET IMPLEMENTED! Use values prior to activation - for now the default is True!
          mxts_mode: Selected score
          batch_size: Batch size for scoring
        """
        from deeplift.conversion import kerasapi_conversion as kc
        from deeplift.layers import NonlinearMxtsMode

        def get_mxts_mode(mode_name):
            # Labels from examples:
            mxts_modes = {'rescale_conv_revealcancel_fc': NonlinearMxtsMode.DeepLIFT_GenomicsDefault,
                          'revealcancel_all_layers': NonlinearMxtsMode.RevealCancel,
                          'rescale_all_layers': NonlinearMxtsMode.Rescale,
                          'grad_times_inp': NonlinearMxtsMode.Gradient,
                          'guided_backprop': NonlinearMxtsMode.GuidedBackprop}
            return mxts_modes[mode_name]

        self.model = model
        if not self.is_compatible(model):
            raise Exception("Model not compatible with DeepLift")

        self.task_idx = task_idx
        self.batch_size = batch_size

        weight_f = tempfile.mktemp()
        arch_f = tempfile.mktemp()
        model.model.save_weights(weight_f)
        with open(arch_f, "w") as ofh:
            ofh.write(model.model.to_json())
        self.deeplift_model = kc.convert_model_from_saved_files(weight_f, json_file=arch_f,
                                                                nonlinear_mxts_mode=get_mxts_mode(mxts_mode))

        # TODO this code may be useful for future when functional models can be handled too
        self.input_layer_idxs = [0]
        self.output_layers_idxs = [-1]
        """
        input_names = self.model._get_feed_input_names()
        self.input_layer_idxs = []
        self.output_layers_idxs = []
        for input_name in input_names:
            input_layer_name = input_name[:-len("_input")] if input_name.endswith("_input") else input_name
            for i, l in enumerate(self.model.model.layers):
                if l.name == input_layer_name:
                    self.input_layer_idxs.append(i)
        """

        self.fwd_predict_fn = None

        # Now try to find the correct layer:
        if not isinstance(output_layer, int):
            raise Exception("output_layer has to be an integer index of the Keras layer in the Keras model.")

        # TODO: DeepLIFT does not guarantee that the layer naming recapitulates the Keras layer order.
        if output_layer < 0:
            output_layer = len(model.model.layers) + output_layer
        target_layer_idx = [i for i, l in enumerate(self.deeplift_model.get_layers()) if l.name == str(output_layer)][0]

        # Compile the function that computes the contribution scores
        # For sigmoid or softmax outputs, target_layer_idx should be -2 (the default)
        # (See "3.6 Choice of target layer" in https://arxiv.org/abs/1704.02685 for justification)
        # For regression tasks with a linear output, target_layer_idx should be -1
        # (which simply refers to the last layer)
        # If you want the DeepLIFT multipliers instead of the contribution scores, you can use get_target_multipliers_func
        self.deeplift_contribs_func = self.deeplift_model.get_target_contribs_func(
            find_scores_layer_idx=self.input_layer_idxs,
            target_layer_idx=target_layer_idx)

    @classmethod
    def is_compatible(cls, model):
        if model.type != "keras":
            # Support only keras models
            return False
        # Check the Keras backend
        import keras
        import keras.backend as K

        if not int(keras.__version__.split(".")[0]) == 2:
            return False

        # Can only support sequential model since the layer ordering is not 1:1
        if not isinstance(model.model, keras.Sequential):
            return False

        # Backend has to be tensorflow
        if model.backend is None:
            backend = K.backend()
        else:
            backend = model.backend
        return backend == "tensorflow"

    def score(self, input_batch, input_ref):
        """
        Calculate DeepLIFT scores of a given input sequence.
        Args:
          input_batch: Model input data 
        Returns:
          DeepLIFT scores in the same shape / same containers as the input batch.
        """
        x_standardized = self.model._batch_to_list(input_batch)
        ref_standaradized = None
        if input_ref is not None:
            ref_standaradized = self.model._batch_to_list(input_ref)

        scores = self.deeplift_contribs_func(task_idx=self.task_idx,
                                             input_data_list=x_standardized,
                                             input_references_list=ref_standaradized,
                                             batch_size=self.batch_size,
                                             progress_update=1000)

        # TODO DeepLIFT error when using batched execution:
        """
        # run_function_in_batches fails for 
        scores = run_function_in_batches(
            func=self.deeplift_contribs_func,
            input_data_list=x_standardized,
            batch_size=self.batch_size,
            progress_update=1000,
            task_idx=self.task_idx)
        """

        # DeepLIFT returns all samples as a list of individual samples
        scores = [numpy_collate(el) for el in scores]

        # re-format the list-type input back to how the input_batch was:
        scores = self.model._match_to_input(scores, input_batch)
        return scores

    def predict_on_batch(self, input_batch):
        """
        Function that can be used to check the successful model conversion. The output of this function should match 
        the output of the original model when executing .predict(input_batch)
        Args:
          input_batch: Model input data 
        Returns:
          Model predictions
        """
        from deeplift.util import run_function_in_batches
        from deeplift.util import compile_func
        x_standardized = self.model._batch_to_list(input_batch)
        if self.fwd_predict_fn is None:
            # TODO: Once DeepLIFT layer annotation works integrate it here too:
            """
            # identify model output layers:
            self.output_layers_idxs = []
            for output_name in self.model.model.output_names:
                for i, l in enumerate(self.model.model.layers):
                    if l.name == output_name:
                        self.output_layers_idxs.append(i)
            """
            inputs = [self.deeplift_model.get_layers()[i].get_activation_vars() for i in self.input_layer_idxs]
            outputs = [self.deeplift_model.get_layers()[i].get_activation_vars() for i in self.output_layers_idxs]
            self.fwd_predict_fn = compile_func(inputs, outputs)

        preds = run_function_in_batches(
            input_data_list=x_standardized,
            func=self.fwd_predict_fn,
            batch_size=self.batch_size,
            progress_update=None)

        preds = np.array(preds)
        if len(self.output_layers_idxs) == 1:
            preds = preds[0, ...]

        return preds


class IntegratedGradients(ImportanceScoreWRef, Gradient):

    def score(self, input_batch, input_ref):
        grads = super(IntegratedGradients, self).score(input_batch)
        # TODO implement integrated gradients
        # https://github.com/marcoancona/DeepExplain/blob/master/deepexplain/tensorflow/methods.py#L208-L225
        pass


class GradientXInput(ImportanceScoreWRef, Gradient):
    # AbstractGrads implements the

    def score(self, input_batch, input_ref):
        # TODO - handle also the case where input_ref is not a simple array but a list
        return super(GradientXInput, self).score(input_batch) * input_ref


METHODS = {"deeplift": DeepLift,
           "grad*input": GradientXInput,
           "intgrad": IntegratedGradients}
