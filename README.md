
# A*NN*aMorphose
Neural Network models for Analogy in Morphology (A*NN*aMorphose) is an approach using deep learning models for the identification and inference of analogies in morphology.
The approach presented here is described in the article "Tackling Morphological Analogies Using Deep Learning" (Anonymous, 2022).

To cite this repository, use the following reference:
```bib
@article{MarquerAMAI22,
  author    = {Esteban Marquer and Miguel Couceiro},
  title     = {Solving morphological analogies: from retrieval to generation},
  eprint={2303.18062},
  archivePrefix={arXiv},
  url       = {https://arxiv.org/abs/2303.18062},
  pdf       = {https://arxiv.org/pdf/2303.18062.pdf},
}
@data{I5ED78_2023,
  author = {Marquer, Esteban and Miguel Couceiro},
  publisher = {Université de Lorraine},
  title = {{Données de réplication pour : ``Solving morphological analogies: from retrieval to generation''}},
  UNF = {UNF:6:JOLK+VaihCp0kH5dTg5ROA==},
  year = {2023},
  version = {VERSION PROVISOIRE},
  doi = {10.12763/I5ED78},
  url = {https://doi.org/10.12763/I5ED78}
}
```

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Install Instructions](#install-instructions)
  - [Cloning the Repository](#cloning-the-repository)
  - [Installing the Dependencies](#installing-the-dependencies)
    - [Anaconda](#anaconda)
  - [Setting up the Siganalogies, Sigmorphon2016, and Sigmorphon2019](#setting-up-the-siganalogies-sigmorphon2016-and-sigmorphon2019)
- [Model files](#model-files)
- [General Usage](#general-usage)
- [Reproducing Experiments](#reproducing-experiments)
  - [Running the Symbolic Baselines](#running-the-symbolic-baselines)
  - [Running the semantic baselines](#running-the-semantic-baselines)
  - [Training the models (the recommended way)](#training-the-models-the-recommended-way)
  - [Training and running the models one by one](#training-and-running-the-models-one-by-one)
    - [Running 3CosMul and 3CosAdd Baselines](#running-3cosmul-and-3cosadd-baselines)
    - [Classification Model CNN+ANNc](#classification-model-cnnannc)
    - [Retrieval Model CNN+ANNr](#retrieval-model-cnnannr)
    - [Retrieval Model CNN+ANNr](#retrieval-model-cnnannr-1)
- [Files and Folders (not up to date)](#files-and-folders-not-up-to-date)

## Install Instructions
The following installation instruction are designed for command line on Unix systems. Refer to the instructions for Git and Anaconda on your exploitation system for the corresponding instructions.

### Cloning the Repository
Clone the repository on your local machine, using the following command:

```bash
git clone https://github.com/EMarquer/nn-morpho-analogy.git
git submodule update --init siganalogies
```

### Installing the Dependencies
#### Anaconda
1.  Install [Anaconda](https://www.anaconda.com/) (or miniconda to save storage space).
2.  Then, create a conda environement (for example `morpho-analogy`) and install the dependencies, using the following commands:
    ```bash
    conda env create --name morpho-analogy python=3.9
    ```
3.  Use one of the following, depending on whether you have a GPU available (you can try newer version of PyTorch, but compatibility is not guaranteed):
    ```bash
    # cuda 11.6
    conda install -y --name morpho-analogy pytorch==1.13.1 torchtext=0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia
    # cuda 11.7
    conda install -y --name morpho-analogy pytorch==1.13.1 torchtext=0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
    # cpu
    conda install -y --name morpho-analogy pytorch==1.13.1 torchtext=0.14.1 cpuonly -c pytorch
    ```
4.  Finally, install the other required libraries:
    ```bash
    # extra conda libs
    conda install -y --name morpho-analogy -c conda-forge pytorch-lightning=1.9.3 torchmetrics=0.11.1
    conda install -y --name morpho-analogy -c conda-forge pandas=1.4.4 seaborn=0.12.2 scikit-learn=1.0.2
    conda install -y --name morpho-analogy -c conda-forge tabulate=0.9.0
    conda install -y --name morpho-analogy tensorboard
    pip install pebble
    ```
5.  All the following commands assume that you have activated the environment you just created. This can be done with the following command (using our example `morpho-analogy`):
    ```bash
    conda activate morpho-analogy
    ```

### Setting up the Siganalogies, Sigmorphon2016, and Sigmorphon2019
To install the Siganalogies data, run at the root of the repository:
- `git submodule update --init siganalogies` for Siganalogies code;
- `git submodule update --init sigmorphon2016` for Sigmorphon 2016;
- `git submodule update --init sigmorphon2019` for Sigmorphon 2019.
- `cp siganalogies/japanese-task1-train sigmorphon2016/data/` to copy the Japanese data, followintg the instructions of Siganalogies.

The Japanese data is stored as a Sigmorphon2016-style data file `japanese-task1-train` at the root of the directory, and should be moved to `sigmorphon2016/data`, the Sigmorphon2016 data folder. There is no test nor development set. For the training and evaluation, the file `japanese-task1-train` is split: 70\% of the analogies for the training and 30\% for the evaluation. The split is always the same for reproducibility, using random seed 42.

The Japanese data was extracted from the original [Japanese Bigger Analogy Test Set](https://vecto-data.s3-us-west-1.amazonaws.com/JBATS.zip).

## Model files
Most of the model files reported in the article "Solving morphological analogies: from retrieval to generation" will be made available through Dorel.
Download all the files into:
- `models/` should contain the data extracted from `models.zip` (ensure that there is no nesting, you should have from the root of the repository: `models/ret/2016/...`)
- `logs/` should contain the files that are not in a ZIP file in Dorel
- `results/` should contain the data extracted from `results.zip` (ensure that there is no nesting, you should have from the root of the repository: `results/ret/2016/...`)

Here is a more detailed description of the files and folders:
- The folders `models` and `result`, respectively found as `model.zip` and `results.zip`, contain the path structure `[model]/[dataset]/[langage]/[random_initialization_id]`, where `[dataset]` follows the Siganalogies labels: `2016` for Sigmorphon2016 and JBATS; `2019` for Sigmorphon2019. The data contained is as follows:
        `models/[...]/model.pkl`: PyTorch model file;
        `models/[...]/summary.csv`: file containing the evaluation results and other metadata about the training and the structure of the model, as well as the timestamp at which the model finished training;
        `models/[...]/version_1.0/`: PyTorch-Lightning training logs viewable by Tensorboard;
        `models/[...]/fails.csv`: enumeration of all the test analogies that the model did not manage to predict correctly, in an extensive format (for most purposes, it is not necessary to consult Siganalogies to analyse the results).
    The two folders cover the following models:
        `clf`: CNN+ANNc for classification;
        `ret`: CNN+ANNr for retrieval;
        `3cosmul`: CNN+3CosMul for retrieval, only contains `summary.csv` and reuses the embedding model of `clf`;
        `ret-annc`: CNN+ANNc for retrieval, only contains `summary.csv` and reuses the embedding model of `clf`.
    The folder `logs` has been unpacked in Dorel, and each file can be found separately. The path structure used follows `ae_annr/[dataset]/[langage]/model[random_initialization_id]-data[random_data_split_id]`. The data contained is as follows:
        `logs/ae_annr/[...]/debug/checkpoints/[...].pkl`: PyTorch-Lightning model file;
        `logs/ae_annr/[...]/summary.csv`: file containing the evaluation results and other metadata about the training and the structure of the model, as well as the timestamp at which the model finished training;
        `logs/ae_annr/[...]/debug/`: PyTorch-Lightning training logs viewable by Tensorboard;
        `logs/ae_annr/[...]/fails.csv`: enumeration of all the test analogies that the model did not manage to predict correctly, in an extensive format (for most purposes, it is not necessary to consult Siganalogies to analyse the results).
    This folder only covers the AE+ANNr model.



## General Usage
For each of the experiments files, it is not necessary to fill the parameters when you run the code, default values are used. You can use the `--help` flag to print the help message and detail available arguments.

## Reproducing Experiments
This section explains how to reproduce step by step the experiments reported in the article.

### Running the Symbolic Baselines
To run the baselines, run `python symbolic_baseline/run_baseline.py -d <dataset> -l <language> -m <algorithm>` (ex: `python baseline/run_baseline.py -l arabic -d 2016 -m kolmo` to run *Kolmo* on the 2016 version of Arabic).
This will output a summary in the command line interface as well as a CSV log file in the baseline folder (ex: `baseline/murena/2016/arabic`).

The available languages are in `siganalogies.config.SIG2016_LANGUAGES` and `siganalogies.config.SIG2019_HIGH`.

The available baselines are:
- `alea` for the approach of P. Langlais, F. Yvon, and P. Zweigenbaum *"Improvements in analogical learning: Application to translating multi-terms of the medical domain"*;
- `kolmo` or `kolmogorov` or `murena` for the approach of P.-A. Murena, M. Al-Ghossein, J.-L. Dessalles, and A. Cornuéjols *"Solving analogies on words based on minimal complexity transformation"*;
- `lepage` for the annalogy classifier of the toolset presented in R. Fam and Y. Lepage *"Tools for the production of analogical grids and a resource of n-gram analogical grids in 11 languages"*; the tools must be installed beforehand as described below.
  1. First, check that you have `swig` installed (`sudo apt install swig` on Ubuntu-based systems).
  2.  Then, run the following from the home directory to install Lepage's algorithm:
      ```bash
      cd symbolic_baseline/lepage
      conda activate morpho-analogy
      pip install -e .
      ```

### Running the semantic baselines
To run the baselines using pretrained embedding on English, run `python sem_baseline/train_all_clf.py`.

### Training the models (the recommended way)
To train all the models CNN+ANNc, run (in order):
- `python train_all_clf.py` to train CNN+ANNc for classification
- `python test_all_ret_annc.py` to test CNN+ANNc on retrieval (can take a long time)
- 
To train all the models CNN+ANNr, run:
- `python train_all_annr_ret.py` to train CNN+ANNr on retrieval, and if necessary will pre-train CNN+ANNc

For the AE+ANNr model, it is necessary to first train the AE:
- `cd genmorpho ; git submodule update --init siganalogies` to set up siganalogies
- `cp ../genmorpho-siganalogies.cfg.json siganalogies.cfg.json` to set up siganalogies configuration
- `cp ../genmorpho-train_all_autoencoders.py train_all_autoencoders.py` to update the languages on which the AE is trained
- `python train_all_autoencoders.py` to train the AE
- `cd ..` to go back to the root repository
- `python train_all_annr_gen.py` to train AE+ANNr

### Training and running the models one by one
#### Running 3CosMul and 3CosAdd Baselines
To run the baselines, run `python baseline/_3cos.py -d <dataset> -l <language> -M <"3CosAdd" or "3CosMul">`

#### Classification Model CNN+ANNc
To train a classifier and the corresponding embedding model for a language, run the following (all parameters are optional, shorthands can be seen with the `--help` flag of the command):

```bash
python train_clf.py -l <language> -d <dataset> -n <number of analogies in training set> -v <number of analogies in validation/development set> -t <number of analogies in test set>  --max_epochs <number of training epochs>
```

Examples: 
- `python train_clf.py -l arabic` to train on Arabic (by default on Sigmorphon 2016), using up to 50000 analogies (default) for 20 epochs (default);
- `python train_clf.py -l german -d 2019 -n 50000 --max_epochs 20` to train on German of Sigmorphon 2019, using up to 50000 analogies for 20 epochs.

#### Retrieval Model CNN+ANNr
To train a regression model and the corresponding embedding model for a language, run the following (all parameters are optional, shorthands can be seen with the `--help` flag of the command):

```bash
python train_ret.py -l <language> -d <dataset> -n <number of analogies in training set> -v <number of analogies in validation/development set> -t <number of analogies in test set>  --max_epochs <number of training epochs> -C <one of "cosine embedding loss", "relative shuffle", "relative all", "all">
```

Examples: 
- `python train_ret.py -l arabic` to train on Arabic (by default on Sigmorphon 2016), using up to 50000 analogies (default) for 20 epochs (default);
- `python train_ret.py -l german -d 2019 -n 50000 --max_epochs 20` to train on German of Sigmorphon 2019, using up to 50000 analogies for 20 epochs.

#### Retrieval Model CNN+ANNr
To train a regression model and the corresponding embedding model for a language, run the following (all parameters are optional, shorthands can be seen with the `--help` flag of the command):

```bash
python train_ret.py -l <language> -d <dataset> -n <number of analogies in training set> -v <number of analogies in validation/development set> -t <number of analogies in test set>  --max_epochs <number of training epochs> -C <one of "cosine embedding loss", "relative shuffle", "relative all", "all">
```

Examples: 
- `python train_ret.py -l arabic` to train on Arabic (by default on Sigmorphon 2016), using up to 50000 analogies (default) for 20 epochs (default);
- `python train_ret.py -l german -d 2019 -n 50000 --max_epochs 20` to train on German of Sigmorphon 2019, using up to 50000 analogies for 20 epochs.

## Files and Folders (not up to date)
Folders in the directory:
- `baseline`: scripts to run each of the 3 baselines (`alea`, `kolmo`, and `lepage`);
- `embeddings`: scripts (and, once downloaded, model files) of the pre-trained embeddings used in our early experiments;
- `utils.py`: generic tools used throught the code;
- `logs`: files generated by the training scripts, also contains the trained models;
- `results`: some other files generated by the training scripts, contains pre-aggregated results;
- `sigmorphon2016`: data files of the Sigmorphon2016 dataset;
- `sigmorphon2019`: data files of the Sigmorphon2019 dataset;
- `snippets`: several scripts used through the development of the approach, typically for evaluation and plotting.

Files at the root of the directory:
- `analogy_clf.py`: analogy classifier model definition;
- `analogy_reg.py`: analogy regression model definition;
- `cnn_embeddings.py`: morphological embedding model definition;
- `config.py`: file centralizing the most important paths and path templates of the project;
- `environment.yml`: file containing the environment data to create the anaconda environment;
- `japanese-task1-train`: backup of the japanese data extracted from the Japanese Bigger Analogy Test Set, in the Sigmorphon2016 task 1 format; to be used if Sigmorphon2016 has to be downloaded manually;
- `README.md`: this file;
- `train_clf.py`: file to train a classification model and the corresponding embedding model on a given language;
- `train_clf_transfer.py`: file to train a classification model and the corresponding embedding model in a transfer learning setting;
- `train_ret.py`: file to train a retrieval model and the corresponding embedding model on a given language.