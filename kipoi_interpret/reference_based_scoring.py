from __future__ import division, absolute_import, print_function
from .common import CompiledApi


class SingleReferenceApi(CompiledApi):

    def get_scoring_func(self, model, **compilation_kwargs):
        #expects a scoring function to be returned with the API:
        #   Inputs:
        #       input_data_list: the input values
        #       input_references_list: the reference values
        #   Returns:
        #       A list of numpy arrays containing the scores
        raise NotImplementedError()    

    def is_compatible(self, model):
        raise NotImplementedError()

    def score(self, input_data_list,
                    input_references_list,
                    batch_size=100, progress_update=None,
                    **compilation_kwargs):
        self.compile_if_needed(**compilation_kwargs)
        #call score_func accordingly in batches
        #TODO: implement
        assert False

    def score_from_cli(self, input_dataloader_config,
                             input_references_config,
                             io_batch_size=100,
                             **other_kwargs):
        #read the data loaders into arrays as needed
        #call score_func accordingly
        assert False        


class MultipleReferencesApi(CompiledApi):

    def get_scoring_func(self, model, **compilation_kwargs):
        #(Same as for SingleReferenceApi)
        #expects a scoring function to be returned with the API:
        #   Inputs:
        #       input_data_list: the input values
        #       input_references_list: the reference values
        #   Returns:
        #       A list of numpy arrays containing the scores
        raise NotImplementedError()    

    def is_compatible(self, model):
        raise NotImplementedError()

    def score(self, input_data_list,
                    reference_generator,
                    num_refs_per_input,
                    batch_size=100,
                    progress_update=None,
                    **compilation_kwargs):
        self.compile_if_needed(**compilation_kwargs)
        #generate multiple references per input
        #call score_func accordingly in batches
        #TODO: implement
        assert False

    def score_from_cli(self, input_dataloader_config,
                             reference_generator_kwargs,
                             io_batch_size=100,
                             **other_kwargs):
        #read the data loaders into arrays as needed
        #instantiate the reference generator using the kwargs
        #call score_func accordingly
        #TODO: implement
        assert False        


class DeepLiftScoringFuncMixin(object): 

    def get_scoring_func(self, model, output_layer,
                               task_idx, preact):
        #TODO: create and return the deeplift func, which
        #takes arguments "input_data_list" and "input_references_list"
        assert False

    def is_compatible(self, model):
        #TODO: implement check for required functions
        #specifically, a "save_in_keras2" func that saves in the keras2
        #format, and also test that the conversion works
        assert False


class DeepLiftSingleReference(
        DeepLiftScoringFuncMixin, SingleReferenceApi)


class DeepLiftMultipleReferences(
        DeepLiftScoringFuncMixin, MultipleReferencesApi)


class IntGradScoringFuncMixin(object): 

    def get_scoring_func(self, model, output_layer,
                               task_idx, preact, num_intervals):
        #num_intervals is the number of intervals for integrated
        #gradients per example
        #TODO: create and return the integrated gradient func, which
        #takes arguments "input_data_list" and "input_references_list"
        assert False

    def is_compatible(self, model):
        #TODO: implement check for required functions
        #(gradients and also possibly activations of the layer if getting
        # scores on an intermediate layer)
        assert False


class IntGradSingleReference(
        IntGradScoringFuncMixin, SingleReferenceApi)


class IntGradMultipleReferences(
        IntGradScoringFuncMixin, MultipleReferencesApi)


class GradTimeDiffRef(SingleReferenceApi):

    def get_scoring_func(self, model, output_layer, task_idx, preact):
        #expects a scoring function to be returned with the API:
        #   Inputs:
        #       input_data_list: the input values
        #       input_references_list: the reference values
        #   Returns:
        #       A list of numpy arrays containing the scores
        #TODO: implement something that does grad*diff_ref
        assert False 

    def is_compatible(self, model):
        #TODO: implement check for required functions
        #(just the gradients function)
        assert False

