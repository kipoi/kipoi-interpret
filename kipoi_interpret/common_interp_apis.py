from __future__ import division, absolute_import, print_function

#Abstract declaration of different interpretation APIs, plus any
#common functionality


class CompiledApi(object)

    def __init__(self):
        self._score_func = None

    def compile(self, model, **compilation_kwargs):
       
        self._score_func = self.get_scoring_func(model=model,
                                                 **compilation_kwargs) 

    def get_scoring_func(self, model, **compilation_kwargs):
        raise NotImplementedError()

    def score(self, **compilation_kwargs):
        raise NotImplementedError()

    def is_compiled(self):
        return self._score_func is not None

    def compile_if_needed(self, **compilation_kwargs):
        if (len(optional_compilation_kwargs) > 0):
            self.compile(**compilation_kwargs)    
        if (self.is_compiled() == False):
            print("Model was not compiled - attempting compilation")
            self.compile(**compilation_kwargs)


class SingleReferenceApi(CompiledApi):

    def get_scoring_func(self, model, **compilation_kwargs):
        #expects a scoring function to be returned with the API:
        #   Inputs:
        #       input_data_list: the input values
        #       input_references_list: the reference values
        #   Returns:
        #       A list of numpy arrays containing the scores
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

