from torch import Tensor
import typing as t

StateDict = t.Dict[str, t.Any]
class Encoder:
    type = None

    @staticmethod
    def from_state_dict(state_dict: StateDict):
        return Encoder()

    def state_dict(self) -> StateDict:
        "Return a data dictionary, loadable for future use of the encoder."
        state_dict = {
            "encoder_type": "none"
        }
        return state_dict

    def decode(self, tensor: Tensor):
        """Decode a tensor as a word (or list of words if multiple dimensions are detected)."""
        raise ValueError("Basic Encoder does not support decoding tensors, as no encoding was performed.")

    def encode(self, word: str):
        """Return the word itself, as no encoding is performed."""
        return word

    def init(self, *args, **kwargs):
        pass
    