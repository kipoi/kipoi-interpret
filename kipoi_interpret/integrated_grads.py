from __future__ import division, absolute_import, print_function
from .common_interp_apis import SingleReferenceApi, MultipleReferencesApi


class IntGradScoringFuncMixin(object): 

    def get_scoring_func(self, model, output_layer,
                               task_idx, preact, num_intervals):
        #num_intervals is the number of intervals for integrated
        #gradients per example
        #TODO: implement the integrated gradient func, which
        #takes arguments "input_data_list" and "input_references_list"
        assert False


class IntGradSingleReference(
        IntGradScoringFuncMixin, SingleReferenceApi)


class IntGradMultipleReferences(
        IntGradScoringFuncMixin, MultipleReferencesApi)
