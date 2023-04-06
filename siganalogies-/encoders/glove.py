"""Not supported anymore."""

from torch import LongTensor, Tensor
import typing

import torch

from .none import Encoder, StateDict

class GloveEncoder(Encoder):
    type = "glove"
    @staticmethod
    def from_state_dict(state_dict: StateDict):
        raise ValueError("State dict handling not supported with GloVe.")

    def state_dict(self) -> StateDict:
        "Return a data dictionary, loadable for future use of the encoder."
        state_dict = {
            "encoder_type": "glove",
        }
        return state_dict

    def decode(self, tensor: Tensor):
        """Decode a tensor as a word (or list of words if multiple dimensions are detected)."""
        raise ValueError("Word decoding not supported with GloVe.")

    def encode(self, word: str):
        """Encode a word."""
        assert self.glove, "Encoder not initialized."

        return self.glove.embeddings.get(word, torch.zeros(300))

    def init(self, vocabulary: typing.Iterable[str]):
        pass
    
    def __hash__(self) -> int:
        return -1
