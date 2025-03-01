import torch
from einops import rearrange
from copy import deepcopy
import mithril as ml
from mithril import IOKey
from mithril.models import *
from torch import nn

backend_torch = ml.TorchBackend(device="cuda")
from collections import OrderedDict
from typing import Tuple, Union
import sys


def quick_gelu(name: str | None = None):
    block = Model(name=name)
    input = IOKey("input")
    block |= Sigmoid()((1.702 * input), output="sigmoid")
    block |= Buffer()(input * block.sigmoid, output=IOKey("output"))
    return block


def multi_head_attention(
    d_model: int, n_head: int, use_attn_mask: bool = False, name: str | None = None
):
    block = Model(name=name)
    assert d_model % n_head == 0, "d_model is not divisible by h"
    queries = IOKey("queries",shape = (1,77,768))
    head_dim = d_model // n_head
    B, L = queries.shape[0], queries.shape[1]
    block |= Linear(3 * d_model, name="in_proj")(queries, output="in_proj")
    
    in_proj: ml.Connection = block.in_proj
    # block |= (buffer_pre:=Buffer())(input=in_proj.reshape((B, L, n_head,3*head_dim)).transpose((0, 2, 1, 3)), output="in_proj_buffer")
    in_proj = in_proj.reshape((B, L, n_head,3*head_dim)).transpose((0, 2, 1, 3)).reshape((B,  n_head,L,3,-1))
    
    # block |= Buffer()(input=in_proj, output="in_proj_buffer")

    queries = (
        in_proj[:, :, :,0,:]
        .reshape((B,  n_head,L, -1))
    ) 
    # queries = in_proj[:,:,0].reshape((B, L,-1))

    keys = (
        in_proj[:, :, :,1,:]
        .reshape((B,  n_head, L,-1))
    )
    values = (
        in_proj[:, :, :,2,:]
        .reshape((B,n_head,  L, -1))
    )
    block |= (buf_q:=Buffer())(input=queries, output="query_buffer")
    block |= (buf_k:=Buffer())(input=keys, output="key_buffer")
    block |= (buf_v:=Buffer())(input=values, output="value_buffer")

    if use_attn_mask:
        block |= ScaledDotProduct(is_causal=False,use_attn_mask=True)(
            query = queries,
            key = keys,
            value = values,
            attn_mask=IOKey("mask"),
            output="attention",
        )
    else:
        block |= ScaledDotProduct(is_causal=False,use_attn_mask = False)(
            query = queries, key = keys, value = values, output="attention"
        )
    values_hat = block.attention.transpose((0, 2, 1, 3)).reshape((B, L, -1))
    block |= Linear(d_model, name="out_proj")(values_hat, output=IOKey("output"))
    return block


m_model = multi_head_attention(768,12,True)
