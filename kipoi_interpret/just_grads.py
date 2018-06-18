from __future__ import division, absolute_import, print_function
from .common import CompiledApi


class JustGrads(CompiledApi):

    def get_scoring_func(self, model, **compilation_kwargs):
        #expects a scoring function to be returned with the API:
        #   Inputs:
        #       input_data_list: the input values
        #   Returns:
        #       A list of numpy arrays containing the gradients
        #TODO: implement 

    def is_compatible(self, model):
        #check the presence of gradients being implemented

    def score(self, input_data_list,
                    batch_size=100, progress_update=None,
                    **compilation_kwargs):
        self.compile_if_needed(**compilation_kwargs)
        #call score_func accordingly in batches
        #TODO: implement
        assert False

    def score_from_cli(self, input_dataloader_config,
                             io_batch_size=100,
                             **other_kwargs):
        #read the data loaders into arrays as needed
        #call score_func accordingly
        assert False 
