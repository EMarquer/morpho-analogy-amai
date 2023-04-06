"""Package containing character & word encoders."""
from .none import Encoder
from .char import CharEncoder

NO_ENCODER=Encoder()

def encoder_as_string(encoder):
    if isinstance(encoder, dict) and "encoder_type" in encoder.keys():
        return encoder["encoder_type"]
    elif (isinstance(encoder, type) and issubclass(encoder, Encoder)) or isinstance(encoder, Encoder):
        return encoder.type
    elif encoder == "char":
        return "char"
    elif encoder == "none" or encoder is None or encoder == id:
        return "none"
    else:
        raise ValueError(f"Unsupported word encoding: {encoder}")
