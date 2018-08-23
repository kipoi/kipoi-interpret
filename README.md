# kipoi_interpret
> Model interepretation plugin for Kipoi.

<a href='https://circleci.com/gh/kipoi/kipoi-interpret'>
	<img alt='CircleCI' src='https://circleci.com/gh/kipoi/kipoi-interpret.svg?style=svg' style="max-height:20px;width:auto">
</a>
<a href=https://coveralls.io/github/kipoi/kipoi-interpret?branch=master>
	<img alt='Coverage status' src=https://coveralls.io/repos/github/kipoi/kipoi-interpret/badge.svg?branch=master style="max-height:20px;width:auto;">
</a>


## Installation

```sh
pip install kipoi_interpret
```

## Usage example

```python
# list all available methods
kipoi_interpret.importance_scores.available_methods()
```

Available methods:
```python
# Gradient-based methods
from kipoi_interpret.importance_scores.gradient import Gradient, GradientXInput
# In-silico mutagenesis-based methods
from kipoi_interpret.importance_scores.ism import Mutation
# DeepLift
from kipoi_interpret.importance_scores.referencebased import DeepLift
```

Gradient * input example
```python
# seqa = one-hot-encoded DNA sequence
import kipoi
model = kipoi.get_model("<my-model>")
ginp = GradientXInput(model)
val = ginp.score(batch_input)  # val is an array of importance scores
```

See [notebooks/1-DNA-seq-model-example.ipynb](notebooks/1-DNA-seq-model-example.ipynb) for an example.

<!-- ## Development setup -->

<!-- Describe how to install all development dependencies and how to run an automated test-suite of some kind. Potentially do this for multiple platforms. -->

<!-- ```sh -->
<!-- TODO -->
<!-- ``` -->


## Release History

* 0.1.0
    * First release to PyPI
