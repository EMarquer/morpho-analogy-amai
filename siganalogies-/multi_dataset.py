from os.path import exists, join, dirname
from os import makedirs
#from torch.utils.data import Dataset
from .config import CUSTOM_SERIALIZATION_PATH, DATASET_PATH, DOWNLOAD, SERIALIZATION, SIG2016_DATASET_PATH, SIG2019_HIGH_LOW_PAIRS, SIG2019_LANGUAGES, SIG2019_DATASET_PATH, SIG2019_SERIALIZATION_PATH, SIG2019_HIGH, SIG2019_LOW, SIG2019_HIGH_MODES, SIG2019_LOW_MODES, dorel_pkl_url
from .abstract_analogy_dataset import AbstractAnalogyDataset, StateDict, save_state_dict, load_state_dict, Encoder, EncodedWord
from .sig_2016 import _load_data as _load_data_2016
from .sig_2019 import _load_data as _load_data_2019
from datetime import datetime
from typing import Iterable, Tuple, Type, Union, List, Dict, TypeVar
import logging

LangIdx = int

# create logger
module_logger = logging.getLogger(__name__)

class MultilingualDataset(AbstractAnalogyDataset):
    @staticmethod
    def from_state_dict(state_dict: StateDict, sig2016_dataset_folder=SIG2016_DATASET_PATH,
            sig2019_dataset_folder=SIG2019_DATASET_PATH, load_data=True, return_dataset_idx=False):
        """Create a dataset from saved data."""
        dataset = MultilingualDataset(datasets_kwargs=state_dict["datasets_kwargs"], strategy=state_dict["strategy"],
            word_encoder=state_dict["word_encoder"], building=False, state_dict=state_dict,
            sig2016_dataset_folder=sig2016_dataset_folder,
            sig2019_dataset_folder=sig2019_dataset_folder, load_data=load_data, return_dataset_idx=False)
        return dataset

    def state_dict(self) -> StateDict:
        "Return a data dictionary, loadable for future use of the dataset."
        state_dict = {
            "timestamp": datetime.utcnow(),
            "datasets_kwargs": self.datasets_kwargs,
            "strategy": self.strategy,
            "analogies": self.analogies,
            "word_encoder": self.word_encoder,
            "word_voc": self.word_voc, 
            #"word_voc_lists": self.word_voc_lists, # list lang_id: set of words with analogies
            "features": self.features,
            #"features_lists": self.word_voc_lists, # list lang_id: set of features with analogies
            "features_with_analogies": self.features_with_analogies,
            "words_with_analogies": self.words_with_analogies,
            "analogies_by_feature": self.analogies_by_feature
            #"words_with_analogies_lists": self.words_with_analogies_lists # list lang_id: set of words with analogies
        }
        return state_dict

    def __init__(self, datasets_kwargs, word_encoder="none", strategy="pairwise", building=True, 
            state_dict: StateDict=None, sig2019_dataset_folder=SIG2019_DATASET_PATH, 
            sig2016_dataset_folder=SIG2016_DATASET_PATH, load_data=True, return_dataset_idx=False, **kwargs):
        """
        :param word_encoder: The word encoder to use. By design, a single encoder is used for all languages. If this is
        not the desired behavior, rely on the language index to specify specific encoders externally, coupled with the 
        None encoder.

        :param datasets_kwargs: sequence of dictionaries with the following structure:
            - "dataset": either "2016" or "2019"
            - "language": a language of the dataset
            - "mode": a valid mode depending on "dataset" and "language"
        
        >>> datasets_kwargs=[
            {"dataset": 2019, "language": "german", "mode": "train-high"},
            {"dataset": 2019, "language": "middle-high-german", "mode": "train-low"}
            ]

        :param strategy: Merging strategy. Either a string ("concat", "pairwise", "pairwise-bidir") or a list of pairs:
            - index of the dataset for A:B in datasets_kwargs
            - index of the dataset for C:D in datasets_kwargs
        By default "pairwise" only creates either A:B::C:D or C:D::A:B given two languages. "pairwise-bidir" enforces 
        having both. 
        """
        super().__init__(word_encoder=word_encoder)
        
        self.datasets_kwargs = list(datasets_kwargs)
        self.strategy = strategy
        self.return_dataset_idx = return_dataset_idx
        
        # loading the data
        if load_data or building:
            self.raw_data = []
            for dataset_kwargs in self.datasets_kwargs:
                if str(dataset_kwargs["dataset"]) == "2016":
                    data = _load_data_2016(language=dataset_kwargs["language"], mode=dataset_kwargs["mode"], dataset_folder=sig2016_dataset_folder)
                else:
                    data = _load_data_2019(language=dataset_kwargs["language"], mode=dataset_kwargs["mode"], dataset_folder=sig2019_dataset_folder)
                self.raw_data.append(data)
                
        # building the analogies
        if building:
            self.set_analogy_classes()
            self.prepare_encoder()
        elif state_dict is not None:
            self.analogies = state_dict["analogies"]
            self.word_voc = set(state_dict["word_voc"])
            self.features = set(state_dict["features"])
            self.features_with_analogies = set(state_dict["features_with_analogies"])
            self.words_with_analogies = set(state_dict["words_with_analogies"])
            self.analogies_by_feature = state_dict["analogies_by_feature"]

    def set_analogy_classes(self):
        """Go through the data to extract the vocabulary, the available features, and build analogies."""
        # a. interpreting the strategy
        if self.strategy == "concat":
            dataset_pairs = [(i, i) for i in range(len(self.datasets_kwargs))]
        elif self.strategy == "pairwise":
            dataset_pairs = [(i, j) for i in range(len(self.datasets_kwargs)) for j in range(len(self.datasets_kwargs)) if j > i]
        else:
            dataset_pairs = self.strategy

        # b. actually building the analogies
        self.analogies = []
        self.word_voc = set()
        self.features = set()
        self.features_with_analogies = set()
        self.words_with_analogies = set()
        
        self.analogies_by_feature = dict()
        def add_analogy(i, j, lang_i, lang_j, a, b, c, d, feature):
            self.analogies.append(((lang_i, i), (lang_j, j)))
            self.features_with_analogies.add(feature)
            self.words_with_analogies.add(a)
            self.words_with_analogies.add(b)
            self.words_with_analogies.add(c)
            self.words_with_analogies.add(d)

            if feature not in self.analogies_by_feature.keys():
                self.analogies_by_feature[feature] = []
            self.analogies_by_feature[feature].append(len(self.analogies)-1)



        for lang_i, lang_j in dataset_pairs:
            for i, (word_a_i, feature_b_i, word_b_i) in enumerate(self.raw_data[lang_i]):
                self.word_voc.add(word_a_i)
                self.word_voc.add(word_b_i)
                self.features.add(feature_b_i)
                if lang_i==lang_j:
                    self.analogies.append(((lang_i, i), (lang_i, i))) # add the identity
                    for j, (word_a_j, feature_b_j, word_b_j) in enumerate(self.raw_data[lang_j][i+1:]):
                        if feature_b_i == feature_b_j:
                            add_analogy(i, i+1+j, lang_i, lang_j, word_a_i, word_b_i, word_a_j, word_b_j, feature_b_i)

                else:
                    for j, (word_a_j, feature_b_j, word_b_j) in enumerate(self.raw_data[lang_j]):
                        self.word_voc.add(word_a_i)
                        self.word_voc.add(word_b_i)
                        self.features.add(feature_b_i)
                        if feature_b_i == feature_b_j:
                            add_analogy(i, j, lang_i, lang_j, word_a_i, word_b_i, word_a_j, word_b_j, feature_b_i)

    def __getitem__(self, index) -> Union[Tuple[EncodedWord, EncodedWord, EncodedWord, EncodedWord],
            Tuple[Tuple[EncodedWord, EncodedWord, EncodedWord, EncodedWord], Tuple[LangIdx, LangIdx]]]:
        """Returns the index-th analogy built in self.set_analogy_classes as a quadruple A, B, C, D for a quadruple
        A:B::C:D. Also return the language indices if self.return_dataset_idx is True."""
        (ab_lang, ab_index), (cd_lang, cd_index) = self.analogies[index]
        a, feature_b, b = self.raw_data[ab_lang][ab_index]
        c, feature_d, d = self.raw_data[cd_lang][cd_index]
        if self.return_dataset_idx:
            return self.encode(a, b, c, d), (ab_lang, cd_lang)
        else:
            return self.encode(a, b, c, d)

def dataset_factory(datasets_kwargs, strategy="pairwise", file_name=None, word_encoder: Union[Type, Encoder, str, None]=None,
        dataset_pkl_folder=CUSTOM_SERIALIZATION_PATH, sig2019_dataset_folder=SIG2019_DATASET_PATH, 
        sig2016_dataset_folder=SIG2016_DATASET_PATH, force_rebuild=False, serialization=SERIALIZATION,
        load_data=True, return_dataset_idx=False) -> MultilingualDataset:
    """If aFile path takes precedence."""
    filepath = None
    if file_name is not None:
        filepath = join(dataset_pkl_folder, file_name)

    # pickle does not exist, create it
    if force_rebuild or file_name is None or not exists(filepath):
        module_logger.info(f"Starting building the dataset {filepath or datasets_kwargs}...")
        dataset = MultilingualDataset(datasets_kwargs, word_encoder=word_encoder, strategy=strategy,
            sig2016_dataset_folder=sig2016_dataset_folder, sig2019_dataset_folder=sig2019_dataset_folder,
            load_data=load_data, return_dataset_idx=return_dataset_idx)
        module_logger.info(f"Dataset {filepath or datasets_kwargs} built.")
        
        if serialization and file_name is not None:
            module_logger.info(f"Saving the dataset to {filepath}...")
            state_dict = dataset.state_dict()
            save_state_dict(state_dict, filepath)
            module_logger.info(f"Dataset saved to {filepath}.")
    
    # pickle exists
    else:
        state_dict = load_state_dict(filepath)
        assert len(state_dict["datasets_kwargs"]) == len(datasets_kwargs)
        assert state_dict["strategy"] == strategy
        assert all(loaded_kwargs == input_kwargs for loaded_kwargs, input_kwargs in zip(state_dict["datasets_kwargs"], datasets_kwargs))

        dataset = MultilingualDataset.from_state_dict(state_dict=state_dict, sig2016_dataset_folder=sig2016_dataset_folder,
            sig2019_dataset_folder=sig2019_dataset_folder, load_data=load_data, return_dataset_idx=return_dataset_idx)
        module_logger.info(f"Dataset {filepath} loaded.")
    return dataset
