from torch import LongTensor, Tensor
import typing

from .none import Encoder, StateDict

class CharEncoder(Encoder):
    id_to_char: typing.Tuple[str]
    char_to_id: typing.Dict[str, int]
    type="char"
    UNK_ID = -1
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2

    @staticmethod
    def from_state_dict(state_dict: StateDict):
        encoder = CharEncoder()
        encoder.id_to_char = state_dict["id_to_char"]
        encoder.char_to_id = {character: i for i, character in enumerate(encoder.id_to_char)}
        return encoder

    def state_dict(self) -> StateDict:
        "Return a data dictionary, loadable for future use of the encoder."
        state_dict = {
            "encoder_type": "char",
            "id_to_char": self.id_to_char
        }
        return state_dict

    def decode(self, tensor: Tensor, unk_char="#", pad_char="_", bos_char="<", eos_char=">"):
        """Decode a tensor as a word (or list of words if multiple dimensions are detected)."""
        assert hasattr(self,"id_to_char"), "Encoder not initialized."
        
        def id_to_char(index):
            if index == CharEncoder.UNK_ID:
                return unk_char
            elif index == CharEncoder.PAD_ID:
                return pad_char
            elif index == CharEncoder.BOS_ID:
                return bos_char
            elif index == CharEncoder.EOS_ID:
                return eos_char
            else:
                return self.id_to_char[index]

        if tensor.dim() > 1:
            return [self.decode(part, unk_char=unk_char, pad_char=pad_char, bos_char=bos_char, eos_char=eos_char) for part in tensor]
        else:
            return "".join([id_to_char(char.int().item()) for char in tensor])

    def encode(self, word: str):
        """Encode a word."""
        assert hasattr(self,"char_to_id"), "Encoder not initialized."

        return LongTensor([self.char_to_id.get(c, CharEncoder.UNK_ID) for c in word])

    def init(self, vocabulary: typing.Iterable[str]):
        chars = set()
        for word in vocabulary:
            chars.update(word)

        self.id_to_char = tuple(["PAD", "BOS", "EOS"] + sorted(chars))
        self.char_to_id = {character: i for i, character in enumerate(self.id_to_char)}
    
    def __hash__(self) -> int:
        return hash(self.id_to_char)
