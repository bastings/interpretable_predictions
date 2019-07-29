# Interpretable Neural Predictions with Differentiable Binary Variables

This is the repository for [Interpretable Neural Predictions with Differentiable Binary Variables](https://www.aclweb.org/anthology/P19-1284), accepted at ACL 2019.
The models in this repository learn to select a rationale, a portion of the input text, serving as an explanation of what is important for classification.
Rationales are trained jointly with classification, either with REINFORCE (for the Lei et al. method), or simply using backpropagation (for our method using the HardKuma distribution). 

If you use this code in your work, then please cite our paper [[bib]](https://www.aclweb.org/anthology/papers/P/P19/P19-1284.bib).

# Notebooks

We curate an updated HardKuma implementation at [https://github.com/probabll/sparse-distributions](https://github.com/probabll/sparse-distributions).
You can also find other distributions there. 

# Instructions

## Installation

You need to have Python 3.6 or higher installed.
It is recommended that you use a virtual environment:
```
sudo pip3 install -U virtualenv
virtualenv --system-site-packages -p python3 ./my_venv
source ./my_venv/bin/activate
```

Then, install all required Python packages using:
```
pip install -r requirements.txt
```

Finally, clone the repository:

```
git clone https://github.com/joeynmt/joeynmt.git
```


## Multi-aspect Sentiment Analysis (Beer Advocate)
See [beer](latent_rationale/beer) directory.

There are three models that you can choose from using `--model`:
1. `baseline` (just classify the training instances)
2. `rl` (the bernoulli baseline / Lei et al. reimplementation)
3. `latent` (hardkuma)


To train on a single aspect (as in Table 2), e.g. aspect 0 (look):

```
python -m latent_rationale.beer.train \
    --model rl \
    --aspect 0 \
    --train_path data/beer/reviews.aspect0.train.txt.gz
    --dev_path data/beer/reviews.aspect0.heldout.txt.gz  
```

To train on **all** aspects:

```
python -m latent_rationale.beer.train \
    --model baseline \
    --aspect -1 \
    --train_path data/beer/reviews.260k.train.txt.gz
    --dev_path data/beer/reviews.260k.heldout.txt.gz  
```

For help/more options:

```
python -m latent_rationale.beer.train -h
```

To predict:

```
python -m latent_rationale.beer.predict --ckpt path/to/model/directory
```

## Stanford Sentiment (SST)

To train the latent rationale model to select 30% of text:

```
python -m latent_rationale.sst.train --model latent --selection 0.3 --save_path results/sst/latent_30pct
```

To train the Bernoulli (REINFORCE) model with L0 penalty weight 0.01:

```
python -m latent_rationale.sst.train --model latent --sparsity 0.01 --save_path results/sst/bernoulli_sparsity01
```

## SNLI

To train our reimplemenation of [Decomposable Attention](https://www.aclweb.org/anthology/D16-1244) (our baseline):

```
python -m latent_rationale.snli.train --save_path results/snli/da --model decomposable
```

(You can enable the self-attention option using --self-attention.)

To train the latent rationale model to select 10% of attention cells:
```
python -m latent_rationale.snli.train --save_path results/snli/latent_10pct --model decomposable --dist hardkuma --selection 0.10
```

Lastly there is a recurrent baseline model as well:
```
python -m latent_rationale.snli.train --save_path results/snli/rnn --model recurrent
```


