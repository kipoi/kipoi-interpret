from __future__ import division, absolute_import, print_function
from .common import ImportanceScore
import numpy as np


class Gradient(ImportanceScore):

    def __init__(self, model,
                 model_input=None,
                 model_output=None,
                 filter_idx=None,
                 avg_func="sum",
                 layer=None,
                 selected_fwd_node=None,
                 pre_nonlinearity=False):
        """
        Args:
          model: Kipoi model
          model_input: Which model input to use (list index or a dictionary name)
          model_output: Which model output to use (op name)
          layer": Which output layer to use to make the predictions. If not specified, the final layer will be used.
          pre_nonlinearity: boolean flag indicating that it should checked whether the selected output is post activation
                   function. If a non-linear activation function is used attempt to use its input.
          filter_idx: Filter index that should be inspected with gradients. If not set all filters will be used.
          avg_func: Averaging function to be applied across selected filters (`--filter_idx`) in layer `--layer`."
          selected_fwd_node: If the selected layer has multiple inbound connections in
             the graph then those can be selected here with an integer
             index. Not necessarily supported by all models.
        """
        self.model = model
        self.model_input = model_input

        self.filter_idx = filter_idx
        self.avg_func = avg_func
        self.layer = layer
        self.selected_fwd_node = selected_fwd_node
        self.pre_nonlinearity = pre_nonlinearity

    @classmethod
    def is_compatible(self, model):
        """Requires the gradient method to be implemented

        Args:
          model: Model instance or ModelDescription
        """
        if hasattr(model, 'input_grad'):
            return True
        if model.type in ["keras", "pytorch", "tensorflow"]:
            return True
        return False

    def score(self, input_batch):
        return self.model.input_grad(input_batch,
                                     filter_idx=self.filter_idx,
                                     avg_func=self.avg_func,
                                     layer=self.layer,
                                     final_layer=self.layer is None,
                                     selected_fwd_node=self.selected_fwd_node,
                                     pre_nonlinearity=self.pre_nonlinearity)


class Saliency(Gradient):

    def score(self, input_batch):
        # TODO - apply the abs function to all list/dict elements if necessary
        return np.abs(super().score(self, input_batch))


METHODS = {"saliency": Saliency,
           "grad": Gradient}
