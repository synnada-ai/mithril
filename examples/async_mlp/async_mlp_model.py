from mithril import IOKey
from mithril.models import (
    Gelu,
    Linear,
    Model
)

def mlp_async(input_dim: int):
    block = Model(name="mlp_async")
    block += Linear(dimension=input_dim * 4, name="fc")(input=IOKey("input"))
    block += Gelu()
    block += Linear(dimension=input_dim, name="proj")(output=IOKey("output"))
    return block

def create_mlp_async(input_dim: int = 1024):
    model = Model(name="async_mlp")
    model += mlp_async(input_dim)(input="input")
    return model