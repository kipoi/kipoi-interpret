# kipoi_interpret
> Model interepretation plugin for Kipoi.

<a href='https://circleci.com/gh/kipoi/kipoi-interpret'>
	<img alt='CircleCI' src='https://circleci.com/gh/kipoi/kipoi-interpret.svg?style=svg' style="max-height:20px;width:auto">
</a>
<a href=https://coveralls.io/github/kipoi/kipoi-interpret?branch=master>
	<img alt='Coverage status' src=https://coveralls.io/repos/github/kipoi/kipoi-interpret/badge.svg?branch=master style="max-height:20px;width:auto;">
</a>


> [!WARNING]
> ### **Kipoi Project - Sunset Announcement**
> 
> After several impactful years, we have made the decision to **archive the Kipoi repositories and end active maintenance** of the project.
> 
> This is a bittersweet moment. While it’s always a little sad to sunset a project, the field of machine learning in genomics has evolved rapidly, with new technologies and platforms emerging that better meet current needs. Kipoi played an important role in its time, helping researchers **share, reuse, and benchmark trained models** in regulatory genomics. We’re proud of what it accomplished and grateful for the strong community support that made it possible.
> 
> Kipoi’s impact continues, however:
> 
> *   [The Kipoi webinar series](seminar.html) will carry on, supporting discussions around model reuse and interpretability.
> *   [Kipoiseq](https://github.com/kipoi/kipoiseq), our standard set of data-loaders for sequence-based modeling, also remains active and relevant.
> 
> Thanks to everyone who contributed, used, or supported Kipoi. It’s been a fantastic journey, and we're glad the project helped shape how models are shared in the field.
> 
> \- The Kipoi Team


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
