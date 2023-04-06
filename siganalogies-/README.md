# `siganalogies` for morphological analogies using Sigmorphon 2016 and 2019
The `siganalogies` package is design to manipulate morphological analogies built upon Sigmorphon 2016 and Sigmorphon 2019 in PyTorch.

- [Changelog](#changelog)
- [How to cite](#how-to-cite)
- [References](#references)
- [Requirements](#requirements)
- [Setup](#setup)
  - [\[OPTIONAL\] Get pre-computed dataset files for analysis or faster first loading](#optional-get-pre-computed-dataset-files-for-analysis-or-faster-first-loading)
- [Basic usage](#basic-usage)
  - [Config file](#config-file)
  - [Dataset object](#dataset-object)
  - [Factories](#factories)
    - [Generic factory](#generic-factory)
    - [Dataset-specific factories](#dataset-specific-factories)
  - [Data encoding](#data-encoding)
  - [Data augmentation](#data-augmentation)
    - [Augmented forms (i.e. permutations)](#augmented-forms-ie-permutations)
    - [Augmented forms (i.e. permutations) with central permutation not accepted as a property of analogy](#augmented-forms-ie-permutations-with-central-permutation-not-accepted-as-a-property-of-analogy)
      - [Augmented forms (i.e. permutations) with central permutation considered a non-property of analogy](#augmented-forms-ie-permutations-with-central-permutation-considered-a-non-property-of-analogy)
- [Dataset description](#dataset-description)
- [Publications using this dataset](#publications-using-this-dataset)
- [\[NOT RECOMMENDED\] Minimal usage with `pickle` and dataset serialization files](#not-recommended-minimal-usage-with-pickle-and-dataset-serialization-files)

## Changelog
Comming soon:
- bilingual datasets (analogies between two similar languages);
- examples of usage;
- spliting of the data folowing the procédure of the articles by Alsaidi *et al.*

## How to cite
To cite this dataset, use the following reference:
- For this dataset:
  ```bib
  @misc{
      To be completed in the very near future.
  }
  ```

## References
The references of the datasets on wich the present dataset bases itself are as follow:
- For Sigmorphon 2016:
  ```bib
  @InProceedings{cotterell2016sigmorphon,
    author    = {Cotterell, Ryan and Kirov, Christo and Sylak-Glassman, John and Yarowsky, David and Eisner, Jason and Hulden, Mans},
    title     = {The {SIGMORPHON} 2016 Shared Task---Morphological Reinflection},
    booktitle = {Proceedings of the 2016 Meeting of {SIGMORPHON}},
    month     = {August},
    year      = {2016},
    address   = {Berlin, Germany},
    publisher = {Association for Computational Linguistics}
  }
  ```
- For the Japanese data added to Sigmorphon 2016, from the Japanese Bigger Analogy Test Set:
  ```bib
  @inproceedings{jap-data:2018:karpinska,
    author = {Marzena Karpinska and Bofang Li and Anna Rogers and Aleksandr Drozd},
    title = {Subcharacter Information in Japanese embeddings: when is it worth it?},
    year = {2018},
    booktitle = {Workshop on the Relevance of Linguistic Structure in Neural Architectures for NLP},
    address = {Melbourne, Australia},
    pages = {28-37},
    publisher = {ACL}
  }
  ```
- For Sigmorphon 2019:
  ```bib
  @inproceedings{mccarthy-etal-2019-sigmorphon,
    title = "The {SIGMORPHON} 2019 Shared Task: Morphological Analysis in Context and Cross-Lingual Transfer for Inflection",
    author = "McCarthy, Arya D.  and
      Vylomova, Ekaterina  and
      Wu, Shijie  and
      Malaviya, Chaitanya  and
      Wolf-Sonkin, Lawrence  and
      Nicolai, Garrett  and
      Kirov, Christo  and
      Silfverberg, Miikka  and
      Mielke, Sabrina J.  and
      Heinz, Jeffrey  and
      Cotterell, Ryan  and
      Hulden, Mans",
    booktitle = "Proceedings of the 16th Workshop on Computational Research in Phonetics, Phonology, and Morphology",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-4226",
    doi = "10.18653/v1/W19-4226",
    pages = "229--244",
  }
  ```

## Requirements
Python >= 3.8 and PyTorch are required for `siganalogies`.
The code was designed with PyTorch 1.10, but most versions equiped with the `torch.utils.data.Dataset` class should work.

## Setup
The recommended file structure is as follows:
```
my_project/
├── siganalogies/ (this package)
│   ├── precomputed/ (serialized datasets)
│   │   ├── 2016
│   │   └── 2019
│   ├── __init__.py
│   └── ...
├── sigmorphon2016/
│   ├── data
│   └── ...
├── sigmorphon2019/
│   ├── task1
│   └── ...
└── ...
```

To use, either copy all the files in a folder named `siganalogies` or clone this repository into your project (typically as a submodule).

You will also need to clone Sigmorphon 2016 and Sigmorphon 2019 into corresponding folders.

This can be done using the following commands, to be run from your project directory.
```bash
git clone git@github.com:EMarquer/siganalogies
git clone git@github.com:ryancotterell/sigmorphon2016 sigmorphon2016
git clone git@github.com:sigmorphon/2019.git sigmorphon2019
cp siganalogies/japanese-task1-train sigmorphon2016/data/japanese-task1-train
```

If you are in a repository, we recommend that you use `git submodule add` instead of `git clone`.

### [OPTIONAL] Get pre-computed dataset files for analysis or faster first loading
When using the dataser factory described in [Factories](#factories), if you pass `download=True` (the default) the corresponding file will be downloaded if it does not exist and `force_rebuild` is `False` (the default).

If you also specify `serialization=False`, no data will be saved locally. This could be useful for applications on devices with limited storage space.

**Be careful. Some pre-computed dataset files do not exist on the remote.**

## Basic usage
To manipulate the analogies, you will first need to load a [dataset object](#dataset-object) using the [dataset factories](#factories).
You will then be able to use the dataset as any other `torch.utils.data.Dataset`, each element being a quadruple.

The dataset object contains analogies of the form $`A:B::C:D`$ and $`A:B::A:B`$, but not the corresponding [augmented forms (i.e. permutations)](#augmented-forms-ie-permutations). We recommend to apply [data augmentation](#data-augmentation) to add said [augmented forms](#augmented-forms-ie-permutations).

### Config file
Some configuration, in particular data paths, can be changed for every iteration.
A cleaner option is to create a `siganalogies.cfg.json` file in your project and specify the configuration here. The format of this JSON file is a dictionary with as keys the name of the configuration, followed by the configuraion new default value.

The JSON file can be located either in your working directory (such that, from your script, `open('siganalogies.cfg.json')` can find the file) or in the `siganalogies` package folder (not recommended).

For example, let us change the value of `siganalogies.config.SIG2016_DATASET_PATH` and `siganalogies.config.SIG2019_DATASET_PATH`.
The corresponding `siganalogies.cfg.json` file will be:
```json
{
  "SIG2016_DATASET_PATH": "new/path/to/sigmorphon2016/data/",
  "SIG2019_DATASET_PATH": "new/path/to/sigmorphon2019/task1/"
}
```

Supported configuration names are:
- `SERIALIZATION` (default `True`);
- `DOWNLOAD` (default `True`);
- `AUTO_DOWNLOAD_SIG` (default `False`, not used yet);
- `DATASET_PATH` (default `<siganalogies root>/precomputed/`);
- `SIG2016_DATASET_PATH` (default `<siganalogies root>/sigmorphon2016/data/`);
- `SIG2016_SERIALIZATION_PATH` (default `DATASET_PATH/2016/`);
- `SIG2019_DATASET_PATH` (default `<siganalogies root>/sigmorphon2019/task1/`);
- `SIG2019_SERIALIZATION_PATH` (default `DATASET_PATH/2019/`).

Other configurations in `siganalogies.config` should not be modified.

[COMING SOON] If `AUTO_DOWNLOAD_SIG` is set to `True`, when trying to access Sigmorphon 2019 or 2016, if the files are missing, they will be downloaded.

### Dataset object
Dataset objects are subclasses of `torch.utils.data.Dataset` and should be created using [factories](#factories).

A dataset object has the following attributes:
- `language`: the dataset language;
- `mode`: the subset of the language data, using the separation done in Sigmorphon (for Sigmorphon 2016: `train`, `dev`, `test`; for Sigmorphon 2019 high resource languages: `train-high`; for Sigmorphon 2019 low resource languages: `train-low`, `dev`, `test`; test-covered is supported for neither dataset);
- `word_encoder`: the word encoding strategy, used by the dataset object (`char` to encode using character IDs based on the characters of the training dataset, or `none` to return the text itself, which is useful when using a custom encoding strategy); available Encoders are in the `siganalogies.encoders` sub-package;
- several other statistic-oriented attributes are computed when building the dataset:
  - `features`: a list of all the features in the dataset;
  - `word_voc`: a list of all the words in the dataset;
  - `features_with_analogies`: a list of all the features in the dataset which are present in at least one analogy;
  - `words_with_analogies`: a list of all the words in the dataset which are present in at least one analogy.

A dataset object has the key methods of a PyTorch Dataset object (`__len__` and `__getitem__`) as well as the following methods, none of which should be used in a standard usage (trust the factories to deal with things accordingly):
- `state_dict`, and the corresponding static method `Dataset.from_state_dict`, used in background by the factories to save and load datasets;
- `prepare_encoder` to prepare the word encoding;
- `set_analogy_classes` to compute the analogical pairs;

### Factories
Manually creating a dataset using its `__init__` call is not recommended, use the [generic factory](#generic-factory) or the [dataset-specific factories](#dataset-specific-factories), as they handle saving and loading the data.

If you are unsure the data saved is correct, you can specify `force_rebuild=True` to the factory function.

You can download the pre-computed dataset file if it does not already exits using `download=True`, to avoid the time required for the first initialization.
See [[OPTIONAL] Get pre-computed dataset files for analysis or faster first loading](#optional-get-pre-computed-dataset-files-for-analysis-or-faster-first-loading) for details.

To load the metadata without loading the data, you can specify `load_data=False`.
Combined with `serialization=False` and `download=True`, it alows to analyse some properties of the dataset without having the Sigmorphon files.

If you do not want to use serialized dataset (hence recompute the analogies each time) you can specify `serialization=False` to the factory function.
To disable precomputed dataset files once, specify `serialization=False` in the factory. To disable dataset files globaly, change `SERIALIZATION=True` to `SERIALIZATION=False` in `siganalogies/config.py` or in the config file.
Unless you also specify `force_rebuild=True`, the existing serialized datasets will still be used.


#### Generic factory
The generic factory is used to unify calls to dataset-specific factories.
Typical usage is when Sigmorphon 2016 and Sigmorphon 2019 will be used interchangeably.
To use the generic factory, refer to the **dataset** specific explanation of the key-word arguments.

```python
from siganalogies import dataset_factory

dataset = dataset_factory(dataset="2016", **kwargs)
```

#### Dataset-specific factories
Specify `dataset_pkl_folder` if you do not use the recommended structure to store the precomputed dataset files.
Defaults are: 
- `siganalogies.config.SIG2016_SERIALIZATION_PATH="siganalogies/precomputed/2016"`
- `siganalogies.config.SIG2019_SERIALIZATION_PATH="siganalogies/precomputed/2019"`

Specify `dataset_folder` if you do not use the recommended structure to store the datasets. 
Defaults are:
- `siganalogies.config.SIG2016_DATASET_PATH="siganalogies/sigmorphon2016/data"`
- `siganalogies.config.SIG2019_DATASET_PATH="siganalogies/sigmorphon2019/task1"`

For all the datasets, `word_encoding` can be either `"char"` for character-based encoding, or either `None` or `"none"`, if no encoding is applied and the raw text data is returned.

For **Sigmorphon 2016**, the available `language`s are listed in `siganalogies.config.SIG2016_LANGUAGES`. The available `mode`s are `train`, `dev`, and `test`, also listed in `siganalogies.config.SIG2016_MODES`.

```python
from siganalogies import dataset_factory_2016, SIG2016_SERIALIZATION_PATH, SIG2016_DATASET_PATH, SERIALIZATION

dataset = dataset_factory(
    language="german",
    mode="train",
    word_encoding="none",
    dataset_pkl_folder=SIG2016_SERIALIZATION_PATH,
    dataset_folder=SIG2016_DATASET_PATH,
    force_rebuild=False,
    serialization=SERIALIZATION)
```

For **Sigmorphon 2019**, the available `language`s are split in two categories:
- high ressource languages, listed in `siganalogies.config.SIG2019_HIGH`, and the corresponding `mode` are `train-high`, also listed in `siganalogies.config.SIG2019_HIGH_MODES`;
- low ressource languages, listed in `siganalogies.config.SIG2019_LOW`, and the corresponding `mode`s are `train`, `dev`, and `test`, also listed in `siganalogies.config.SIG2019_LOW_MODES`.

```python
from siganalogies import dataset_factory_2019, SIG2019_SERIALIZATION_PATH, SIG2019_DATASET_PATH, SERIALIZATION

dataset = dataset_factory(
    language="german",
    mode="train-high",
    word_encoding="none",
    dataset_pkl_folder=SIG2019_SERIALIZATION_PATH,
    dataset_folder=SIG2019_DATASET_PATH,
    force_rebuild=False,
    serialization=SERIALIZATION)
```

### Data encoding
There are two strategies available for word encoding.
1. The first strategy is to provide, depending on the needs, an existing or custom `encoders.Encoder` object, `"char"`, or `"none"`/`None`/`id` to the Sigmorphon Dataset object. This will apply the encoding at each call of the dataset's `__getitem__` method.
2. The second strategy is to do the encoding in a collate function of PyTorch or the equivalent in whatever language you are using. There, you can also use an existing or custom `encoders.Encoder` object, or define the collate function to fit your needs.

### Data augmentation
`utils.py` and `utils_no_cp.py` contain implementations of the data augmentation process, respectively with (default setting) or without central permutations among the predicates of analogy.

There are 3 key functions in each file:
- `enrich(a, b, c, d)` generates equivalent/valid/positive permutations from a valid permutation;
- `generate_negative(a, b, c, d)` generates invalid/negative permutations from a valid/positive one;
- `n_pos_n_neg(a, b, c, d, n=-1)` combines the two, and computes the invalid (i.e., negative) analogies from each permutation generated by `enrich(a, b, c, d)`.
  Using a positive value for `n` results in a sampling process such that we end up with `n` positive and `n` negative permutations. After all the valid and invalid permutations are computed, for each of the two sets of permutations:
  - if there are exactly `n` permutations, we return them;
  - if there are more than `n` permutations available, we sample `n` among them;
  - if there is less than `n` available, we sample additional permutations among until we have a total of `n`.

#### Augmented forms (i.e. permutations)
From $`A:B::C:D`$ we can have the following equivalent permutations (or forms):
- $`A:B::C:D`$ (the base form)
- $`A:C::B:D`$
- $`B:A::D:C`$
- $`B:D::A:C`$
- $`C:D::A:B`$
- $`C:A::D:B`$
- $`D:C::B:A`$
- $`D:B::C:A`$

We can also compute forms which should not be valid analogies, by permuting each analogical form $`A:B::C:D`$ above to obtain:
- $`A:A::C:D`$
- $`B:A::C:D`$
- $`C:B::A:D`$

The above process is implemented in `utils.py`.

#### Augmented forms (i.e. permutations) with central permutation not accepted as a property of analogy
From $`A:B::C:D`$ we can have the following equivalent permutations (or forms):
- $`A:B::C:D`$ (the base form)
- $`B:A::D:C`$
- $`C:D::A:B`$
- $`D:C::B:A`$

We can also compute forms which should not be valid analogies, by permuting each analogical form $`A:B::C:D`$ above to obtain:
- $`A:A::C:D`$
- $`B:A::C:D`$

The above process is implemented in `utils_no_cp.py`.

##### Augmented forms (i.e. permutations) with central permutation considered a non-property of analogy
If we consider central permutation a property analogy must not have, we add the following forms (involving central permutation) which should not be valid analogies:
- $`C:B::A:D`$
- $`A:C::B:D`$

The above process is implemented in `utils_no_cp.py`, but requires specifying the keyword argument `cp_undefined=False`. In other words, not only central permutation is not valid, but it is considered a non-property instead of merely being undefined.

## Dataset description
See [`siganalogies_description.pdf`](siganalogies_description.pdf).

## Publications using this dataset
To be completed.

## [NOT RECOMMENDED] Minimal usage with `pickle` and dataset serialization files
It is possible to access the pre-computed data directly, using the following:
```python
from pickle import load

data = load(open("precomputed/2019/adyghe-train-high-none.pkl", "rb"))
```

**The pickled data files do not contain the data itself**, even if they are usable without the actual data from Sigmorphon 2016 and Sigmorphon 2019.

The pickled data files follow the following pattern: `<language>-<mode>-<word encoding>.pkl`.

`data` will be a dictionary with the following keys:
- `timestamp`: the timestamp of the creation of the pickled data;
- all the key attributes of a dataset object:
  `language`, `mode`, `word_encoding`, `char_voc` and `char_voc_id` (if `word_encoding` is `char`), `features`,`word_voc`, `features_with_analogies`, and `words_with_analogies`.