from torch.utils.data import Dataset
from abc import ABC, abstractstaticmethod, abstractmethod
import collections as c
import collections.abc as c_abc
import typing as t
from pickle import dump, load
from json import dump as dump_json, load as load_json, JSONEncoder, JSONDecoder

from .encoders import Encoder, CharEncoder
EncodedWord = t.TypeVar('EncodedWord')

class WordEncoderJSONEncoder(JSONEncoder):
    def default(self, o: Encoder):
        return o.__dict__
class WordDecoderJSONDecoder(JSONDecoder):
    def default(self, o) -> Encoder:
        return o.__dict__

StateDict = t.Dict[str, t.Any]
def save_state_dict(state_dict: StateDict, filepath: str, format: t.Literal["auto", "json", "pkl"]="auto") -> None:
    if format == "auto": format = "json" if filepath.endswith(".json") else "pkl"
    if format == "json":
        with open(filepath, "w", encoding="utf8") as f:
            state_dict = {**state_dict}
            if "timestamp" in state_dict.keys():
                state_dict["timestamp"] = state_dict["timestamp"].isoformat()
            def default(o):
                if isinstance(o, Encoder):
                    return o.state_dict()
                elif isinstance(o, set):
                    return list(o)
            dump_json(state_dict, f, default=default)
    else:
        with open(filepath, "wb") as f:
            dump(state_dict, f)
def load_state_dict(filepath: str, format: t.Literal["auto", "json", "pkl"]="auto") -> StateDict:
    if format == "auto": format = "json" if filepath.endswith(".json") else "pkl"
    if format == "json":
        with open(filepath, "r", encoding="utf8") as f:
            state_dict = load_json(f)
    else:
        with open(filepath, "rb") as f:
            state_dict = load(f)
    return state_dict

class AbstractAnalogyDataset(Dataset, ABC):
    #analogies: c_abc.Sized

    @abstractstaticmethod
    def from_state_dict(state_dict, dataset_folder=None):
        """Create a dataset from saved data."""
        raise NotImplementedError("Should be implemented in subclass.")

    @abstractmethod
    def state_dict(self):
        "Return a data dictionary, loadable for future use of the dataset."
        raise NotImplementedError("Should be implemented in subclass.")

    def __init__(self, word_encoder: t.Union[t.Type, Encoder, str, None]=None, **kwargs):
        """A dataset class for manipultating files of task 1 of Sigmorphon2019."""
        super(AbstractAnalogyDataset).__init__()

        # if a state dict is provided, load the encoder from the state dict
        if isinstance(word_encoder, dict) and "encoder_type" in word_encoder.keys():
            encoder_type = word_encoder["encoder_type"]
            if encoder_type is None or encoder_type == "none" or encoder_type == id:
                self.word_encoder = Encoder.from_state_dict(word_encoder)
            elif encoder_type == "char":
                self.word_encoder = CharEncoder.from_state_dict(word_encoder)
            else:
                raise ValueError(f"Unsupported word encoding from state dict: {encoder_type}")
        
        # if an encoder is provided, use it directly
        elif isinstance(word_encoder, Encoder):
            self.word_encoder = word_encoder

        # if an encoder class or an encoder type is provided, create the corresponding object
        elif isinstance(word_encoder, type) and issubclass(word_encoder, Encoder):
            self.word_encoder = word_encoder()
        elif word_encoder == "char":
            self.word_encoder = CharEncoder()
        elif word_encoder is None or word_encoder == "none" or word_encoder == id:
            self.word_encoder = Encoder()
        
        # otherwise, throw an error
        else:
            raise ValueError(f"Unsupported word encoding: {word_encoder}")

    def prepare_encoder(self):
        """Initialize the encoder."""
        assert self.word_voc, "Word vocabulary not initialized yet."
        self.word_encoder.init(self.word_voc)

    @abstractmethod
    def set_analogy_classes(self):
        """Go through the data to extract the vocabulary, the available features, and build analogies."""
        raise NotImplementedError("Should be implemented in subclass.")

    def encode(self, a: str, b: str, c: str, d: str) -> t.Tuple[EncodedWord, EncodedWord, EncodedWord, EncodedWord]:
        """Encode 4 words using the selected encoding process.
        
        A wraper around self.encode_word applied to each of the arguments."""
        return self.word_encoder.encode(a), self.word_encoder.encode(b), self.word_encoder.encode(c), self.word_encoder.encode(d)

    def decode(self, a: EncodedWord, b: EncodedWord, c: EncodedWord, d: EncodedWord) -> t.Tuple[str, str, str, str]:
        """Decode 4 words using the selected encoding process.
        
        A wraper around self.decode_word applied to each of the arguments."""
        return self.word_encoder.decode(a), self.word_encoder.decode(b), self.word_encoder.decode(c), self.word_encoder.decode(d)

    #@abstractmethod
    def __len__(self) -> int:
        """Returns the number of analogies built in self.set_analogy_classes."""
        return len(self.analogies)
        #raise NotImplementedError("Should be implemented in subclass.")

    @abstractmethod
    def __getitem__(self, index: int):
        """Returns the index-th analogy built in self.set_analogy_classes as a quadruple A, B, C, D for a quadruple
        A:B::C:D."""
        raise NotImplementedError("Should be implemented in subclass.")
