# Setup

```
pip install pytorch-transformers==1.2.0
cd subnetwork-probing/transformer_lens
pip install -e .
```

Then you will need to install `mlab2/` (which requires Redwood or REMIX access, it's NOT the same repo as this: https://github.com/redwoodresearch/mlab2)

In the directory containing `mlab2/`, make a file called `setup.py` and add this:

```
import setuptools

setuptools.setup(
    name="mlab2",
    version="0.1",
    packages=["mlab2"],
    package_dir={"mlab2": "mlab2"},
)
```

Then run `pip install -e .` in this directory

# TODO

[ ] Sort out `transformer_lens` as a submodule for code release.
[ ] Tune learning rate
[ ] tests for initialisation
[ ] MLP masked hook points
[ ] tests for Gumbel 

# subnetwork-probing
This is code for the paper:  
[Low-Complexity Probing via Finding Subnetworks](https://github.com/stevenxcao/subnetwork-probing)  
Steven Cao, Victor Sanh, Alexander M. Rush  
NAACL-HLT 2021  
### Dependencies
This code was tested with `python 3.6`, `pytorch 1.1`, and `pytorch-transformers 1.2`.
### Data
This paper uses the [Universal Dependencies](https://universaldependencies.org/) dataset for dependencies and part-of-speech tagging, and the [CoNLL 2003 NER Shared Task](https://www.clips.uantwerpen.be/conll2003/ner/) for named entity recognition. 
### Running the code
To run the main experiment in the paper, run `python main.py PATH_TO_SAVE_DIRECTORY`.
