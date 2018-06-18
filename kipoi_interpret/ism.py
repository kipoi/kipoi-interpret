from __future__ import division, absolute_import, print_function
from .common_interp_apis import CompiledApi


class SimpleISM(CompiledApi):

    def get_scoring_func(self, model, output_layer, task_idx, preact):
        #expects a scoring function to be returned with the API:
        #   Inputs:
        #       input_data_list: the input values
        #   Returns:
        #       The model prediction
        
        #todo: Implement me
        assert False 

    def score(self, input_onehot_fasta,
                    batch_size=100, progress_update=None,
                    **compilation_kwargs):
        self.compile_if_needed(**compilation_kwargs)
        #perturb the sequences according to some rule
        #call score_func to get new prediction on perturbed seqs
        #compile the stuff together to get the scores
        #TODO: implement
        assert False

    def score_from_cli(self, input_onehot_fasta_data_loader_config,
                             io_batch_size=100,
                             **other_kwargs):
        #read the data to create input_onehot_fasta
        #call score_func accordingly
        assert False        
