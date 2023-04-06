from .config import DOWNLOAD, SERIALIZATION, SIG2016_LANGUAGES, SIG2016_DATASET_PATH, SIG2016_MODES, SIG2016_SERIALIZATION_PATH, JBATS_USE_KANJI, dorel_pkl_url
from os.path import exists, join, dirname
from os import makedirs
from datetime import datetime
from .abstract_analogy_dataset import AbstractAnalogyDataset, EncodedWord, StateDict, save_state_dict, load_state_dict
import logging
import typing as t
from .encoders import Encoder, encoder_as_string

# create logger
module_logger = logging.getLogger(__name__)

def _load_data(language="german", mode="train", task=1, dataset_folder=SIG2016_DATASET_PATH):
    '''Load the data from the sigmorphon files in the form of a list of triples (lemma, target_features, target_word).'''
    assert language in SIG2016_LANGUAGES, f"Language '{language}' is unkown, allowed languages are {SIG2016_LANGUAGES}"
    assert mode in SIG2016_MODES, f"Mode '{mode}' is unkown, allowed modes are {SIG2016_MODES}"
    if language == 'japanese':
        assert mode == 'train', f"Mode '{mode}' is unkown for Japanese, the only allowed mode is 'train'"
    filename = f"{language}-task{task}-{mode}"
    with open(join(dataset_folder, filename), "r", encoding="utf-8") as f:
        if language == 'japanese':
            lines = []
            for line in f:
                a, f, b = line.strip().split('\t')
                if "/" in b: # kanji /  hiragana_katakana 
                    kanji, hiragana_katakana = b.split("/")
                    lines.append([a, f, kanji if JBATS_USE_KANJI else hiragana_katakana])  
                else:
                    lines.append([a, f, b]) 
            return [line.strip().split('\t') for line in f]
        else:
            return [line.strip().split('\t') for line in f]

class Sig2016Dataset(AbstractAnalogyDataset):
    @staticmethod
    def from_state_dict(state_dict: StateDict, dataset_folder=SIG2016_DATASET_PATH, load_data=True):
        dataset = Sig2016Dataset(
            language=state_dict["language"],
            mode=state_dict["mode"],
            word_encoder=state_dict["word_encoder"],
            building=False, state_dict=state_dict, dataset_folder=dataset_folder, load_data=load_data)
        return dataset

    def state_dict(self) -> StateDict:
        "Return a data dictionary, loadable for future use of the dataset."
        state_dict = {
            "timestamp": datetime.utcnow(),
            "language": self.language,
            "mode": self.mode,
            "word_encoder": self.word_encoder.state_dict(),
            "analogies": self.analogies,
            "word_voc": self.word_voc,
            "features": self.features,
            "features_with_analogies": self.features_with_analogies,
            "words_with_analogies": self.words_with_analogies
        }
        return state_dict

    """A dataset class for manipultating files of task 1 of Sigmorphon2016."""
    def __init__(self, language="german", mode="train", word_encoder: t.Union[t.Type, Encoder, str, None]=None, 
            building=True, state_dict: StateDict=None, dataset_folder=SIG2016_DATASET_PATH, load_data=True, **kwargs):
        super().__init__(word_encoder=word_encoder)
        self.language = language
        self.mode = mode
        if load_data or building:
            self.raw_data = _load_data(language=language, mode=mode, task=1, dataset_folder=dataset_folder)

        if building:
            self.set_analogy_classes()
            self.prepare_encoder()
        elif state_dict is not None:
            self.analogies = state_dict["analogies"]
            self.word_voc = set(state_dict["word_voc"])
            self.features = set(state_dict["features"])
            self.features_with_analogies = set(state_dict["features_with_analogies"])
            self.words_with_analogies = set(state_dict["words_with_analogies"])

    def set_analogy_classes(self):
        self.analogies = []
        self.word_voc = set()
        self.features = set()
        self.features_with_analogies = set()
        self.words_with_analogies = set()
        for i, (word_a_i, feature_b_i, word_b_i) in enumerate(self.raw_data):
            self.word_voc.add(word_a_i)
            self.word_voc.add(word_b_i)
            self.features.add(feature_b_i)
            self.analogies.append((i, i)) # add the identity
            for j, (word_a_j, feature_b_j, word_b_j) in enumerate(self.raw_data[i+1:]):
                if feature_b_i == feature_b_j:
                    self.analogies.append((i, i+1+j))
                    self.features_with_analogies.add(feature_b_i)
                    self.words_with_analogies.add(word_a_i)
                    self.words_with_analogies.add(word_b_i)
                    self.words_with_analogies.add(word_a_j)
                    self.words_with_analogies.add(word_b_j)
                    assert i+1+j<len(self.raw_data), f"{i+1+j=}<{len(self.raw_data)=}"
                    assert i<len(self.raw_data), f"{i=}<{len(self.raw_data)=}"

    def __getitem__(self, index: int) -> t.Tuple[EncodedWord, EncodedWord, EncodedWord, EncodedWord]:
        ab_index, cd_index = self.analogies[index]
        assert cd_index<len(self.raw_data), f"{cd_index=}<{len(self.raw_data)=}"
        assert ab_index<len(self.raw_data), f"{cd_index=}<{len(self.raw_data)=}"
        a, feature_b, b = self.raw_data[ab_index]
        c, feature_d, d = self.raw_data[cd_index]
        return self.encode(a, b, c, d)

def dataset_factory(language="german", mode="train", word_encoder="none", dataset_pkl_folder=SIG2016_SERIALIZATION_PATH,
        dataset_folder=SIG2016_DATASET_PATH, force_rebuild=False, serialization=SERIALIZATION, download=DOWNLOAD,
        load_data=True) -> Sig2016Dataset:
    file_name = f"{language}-{mode}-{encoder_as_string(word_encoder)}.pkl"
    filepath = join(dataset_pkl_folder, file_name)


    # pickle does not exist, if allowed, attempt to fetch it online
    if not force_rebuild and not exists(filepath) and download:
        import urllib.request as r
        import urllib.error as er
        try:
            module_logger.info(f"Downloading remote pickle to {filepath}...")
            url = dorel_pkl_url(dataset=2016, language=language, mode=mode, word_encoder=word_encoder)
            
            if serialization: # download to file
                makedirs(dirname(filepath), exist_ok=True)
                r.urlretrieve(url, filepath)
                module_logger.info(f"Dataset {file_name} downloaded from remote file to {filepath}.")
            else: # use on the fly, return the file directly
                state_dict = r.urlretrieve(url)
                dataset = Sig2016Dataset.from_state_dict(state_dict=state_dict, dataset_folder=dataset_folder,
                        load_data=load_data)
                module_logger.info(f"Dataset {file_name} loaded from remote file.")
                return dataset
        except er.HTTPError:
            module_logger.error(f"HTTPError while attempting to get {file_name} from {url}. Possibly, the file does not exist remotely.")
        except er.URLError:
            module_logger.error(f"URLError while attempting to get {file_name} from {url}.")
        except Exception as e:
            module_logger.error(f"Unknown {e} exception while attempting to get {file_name} from {url}.")
                

    # pickle still does not exist (error above or download is False), create it
    if force_rebuild or not exists(filepath):

        module_logger.info(f"Starting building the dataset {filepath}...")
        if mode != "train":
            module_logger.info(f"Using the corresponding training dataset for  {filepath}...")
            train_dataset = dataset_factory(language=language, mode="train", word_encoder=word_encoder, 
                    dataset_pkl_folder=dataset_pkl_folder, dataset_folder=dataset_folder, force_rebuild=force_rebuild,
                    load_data=load_data)
            state_dict = train_dataset.state_dict()
            state_dict["mode"] = mode
            dataset = Sig2016Dataset.from_state_dict(state_dict, dataset_folder=dataset_folder, load_data=load_data)
            module_logger.info(f"Computing the analogies for {filepath}...")
            dataset.set_analogy_classes()
        else:
            dataset = Sig2016Dataset(language=language, mode=mode, word_encoder=word_encoder,
                    dataset_folder=dataset_folder, load_data=load_data)
        module_logger.info(f"Dataset {filepath} built.")
        
        if serialization:
            module_logger.info(f"Saving the dataset to {filepath}...")
            state_dict = dataset.state_dict()
            save_state_dict(state_dict, filepath)
            module_logger.info(f"Dataset saved to {filepath}.")
        return dataset
    
    # pickle exists
    else:
        state_dict = load_state_dict(filepath)
        dataset = Sig2016Dataset.from_state_dict(state_dict=state_dict, dataset_folder=dataset_folder, 
                load_data=load_data)
        module_logger.info(f"Dataset {filepath} loaded.")
        return dataset

if __name__ == "__main__":
    dataset = dataset_factory(word_encoder="char")
    print(len(dataset.analogies))
    print(dataset[1])
    print(dataset.analogies[1])
    print(dataset.raw_data[dataset.analogies[1][0]])
    print(dataset.raw_data[dataset.analogies[1][1]])

    print()

    dataset = dataset_factory(word_encoder=id)
    print(len(dataset.analogies))
    print(dataset[1])
    print(dataset.analogies[1])
    print(dataset.raw_data[dataset.analogies[1][0]])
    print(dataset.raw_data[dataset.analogies[1][1]])
    #print(len(Task2Dataset()))
    #print(Task2Dataset()[2500])
    #print(Task2Dataset().raw_data[2500])