from __future__ import division, absolute_import, print_function
from .common import CompiledApi


class DeltaSvm(CompiledApi):
    
    def get_scoring_func(self, model, **compilation_kwargs):
        raise NotImplementedError() 

    def score(self, input_fastas, **compilation_kwargs):
        self.compile_if_needed(**compilation_kwargs)
        #TODO: implement
        assert False 

    def score_from_cli(self, input_fastas_dataloader_config, **other_kwargs):
        #TODO: implement
        assert False 

#Command line call would look like:

#kipoi interpret delta_svm.DeltaSVM --args "{...}"
