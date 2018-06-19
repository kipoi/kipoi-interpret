from __future__ import division, absolute_import, print_function
from .common import ImportanceScoreWRef
from .gradient import Gradient


# Other proposal (this object is passed as an argument to compile())
class DeepLift(ImportanceScoreWRef):

    def __init__(self, model, output_layer,
                 task_idx, preact):
        # TODO: create and return the deeplift func, which
        # takes arguments "input_data_list" and "input_references_list"
        assert False
        self.compiled_func = get_compiled_fn(model.model, ...)

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
        return self.compiled_func(
            input_data_list=input_batch,
            input_references_list=input_ref)


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
