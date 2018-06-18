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

    def is_compatible(self, model):
        raise NotImplementedError()

    def score(self, **compilation_kwargs):
        raise NotImplementedError()

    def is_compiled(self):
        return self._score_func is not None

    def compile_if_needed(self, **compilation_kwargs):
        if (len(compilation_kwargs) > 0):
            self.compile(**compilation_kwargs)    
        elif (self.is_compiled() == False):
            print("Model was not compiled - attempting compilation")
            self.compile(**compilation_kwargs)

