This is a fixed version of the original repository. It contains additional extensions, however if you are just interested in reproducing the original paper, then revert back to commit *70fbaea3cfab2633c9e2cac7616e75b15a3773c2*.

# Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data
This is an implementation of the PAC-Bayes generalization bound optimization for stochastic neural networks, as described in the article "[Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data](https://arxiv.org/pdf/1703.11008.pdf)" by Dziugaite and Roy, published in *Uncertainty in AI* (2017).

## Setting up the environment

1. Install Python 3.5.5 through Anaconda (e.x. follow https://stackoverflow.com/a/56134984/7343355)
2. pip install -r requirements.txt
3. Set PYTHONPATH

## Requirements
Python 3.5: absl-py==0.11.0, astor==0.8.1, certifi==2020.6.20, gast==0.4.0, grpcio==1.33.2, h5py==2.10.0, importlib-metadata==2.0.0, Keras==2.2.2, Keras-Applications==1.0.4, Keras-Preprocessing==1.0.2, Markdown==3.2.2, numpy==1.14.5, protobuf==3.14.0, PyYAML==5.3.1, scipy==1.4.1, six==1.15.0, tensorboard==1.10.0, tensorflow==1.10.0, termcolor==1.1.0, Werkzeug==1.0.1, wincertstore==0.2, zipp==1.2.0

## Instructions
Running the code involves 2 steps:
1. SGD optimization, which saves initial and final network weights to `snn/experiments`  
2. PAC-Bayes optimization, which first loads the weights saved in the previous step, and then optimizes the PAC-Bayes bound over the weights and variances.

### SGD Optimization
To run SGD on a fully connected neural network consisting of a hidden layer with 600 neurons for 20 epochs on binary MNIST, execute the following command:

`python3.5 snn/experiments/run_sgd.py fc --layers 600 --sgd_epochs 20 --binary`

The code will throw a `FileExistsError` if a checkpoint already exists. To overwrite an existing checkpoint, use the following command: 

`python3.5 snn/experiments/run_sgd.py fc --layers 600 --sgd_epochs 20 --overwrite --binary`

### PAC-Bayes Optimization

The following command can be used to run the PAC-Bayes optimization for 1000 epochs on the saved checkpoint: 

`python3.5 snn/experiments/run_pacb.py fc --layers 600 --sgd_epochs 20 --pacb_epochs 1000 --lr 0.001 --drop_lr 250 --lr_factor 0.1 --binary`

The learning rate starts at 0.001 and is dropped to 0.0001 after 250 epochs.

## Acknowledgments

The development of this code was initiated by [Gintare Karolina Dziugaite](https://gkdz.org) and [Daniel M. Roy](http://danroy.org), while they were visiting the Simons Institute for the Theory of Computing at U.C. Berkeley. During this course of this research project, GKD was supported by an EPSRC studentship; DMR was supported by an NSERC Discovery Grant, Connaught Award, and U.S. Air Force Office of Scientific Research grant #FA9550-15-1-0074.

Waseem Gharbieh (Element AI) and Gabriel Arpino (University of Toronto) contributed to improving and testing the code, and helped produce this code release.

## BiBTeX

    @inproceedings{DR17,
            title = {Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data},
           author = {Gintare Karolina Dziugaite and Daniel M. Roy},
             year = {2017},
        booktitle = {Proceedings of the 33rd Annual Conference on Uncertainty in Artificial Intelligence (UAI)},
    archivePrefix = {arXiv},
           eprint = {1703.11008},
    }
