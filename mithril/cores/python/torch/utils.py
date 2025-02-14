# Copyright 2022 Synnada, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
from functools import partial

import torch

from ....common import BiMap
from ...utils import binary_search

dtype_map: BiMap[str, torch.dtype] = BiMap(
    {
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bool": torch.bool,
    }
)


def tsne_softmax(
    input_tensor: torch.Tensor,
    diag_zero: bool = False,
    zero_index: int | None = None,
) -> torch.Tensor:
    input_tensor = input_tensor - torch.max(input_tensor, dim=1, keepdim=True)[0]
    e = torch.exp(input_tensor)
    if zero_index is None:
        if diag_zero:
            e.fill_diagonal_(0.0)
    else:
        e[:, zero_index] = 0.0
    s = torch.sum(e, dim=1, keepdim=True)
    return e / s


def calc_prob_matrix(
    negative_dist_sq: torch.Tensor,
    sigmas: torch.Tensor,
    zero_index: int | None = None,
) -> torch.Tensor:
    """Convert a distances matrix to a matrix of probabilities.
    Parameters
    ----------
    negative_dist_sq : jax.Array
        Square of distance matrix multiplied by (-1).
    sigmas : jax.Array
        Sigma values according to desired perplexity
        Sigmas calculated using binary search.
    zero_index : int, optional
        The index to be set 0, by default None.
    Returns
    -------
    jax.Array
        Returns conditional probabilities using distance matrix.
    """
    two_sig_sq = 2.0 * torch.square(sigmas.reshape((-1, 1)))
    if two_sig_sq.shape[0] == 1:
        dist_sig = [negative_dist_sq / two_sig_sq, 0][torch.squeeze(two_sig_sq) == 0.0]
        assert isinstance(dist_sig, torch.Tensor)

    else:
        mask = two_sig_sq == 0.0
        dist_sig = torch.zeros_like(negative_dist_sq)
        dist_sig[~mask[:, 0], :] = negative_dist_sq[~mask[:, 0], :] / two_sig_sq[~mask]

    return tsne_softmax(input_tensor=dist_sig, diag_zero=True, zero_index=zero_index)


def perplexity_fn(
    negative_dist_sq: torch.Tensor,
    sigmas: torch.Tensor,
    zero_index: int,
    threshold: torch.Tensor,
) -> torch.Tensor:
    """Wrapper function for quick calculation of
        perplexity over a distance matrix.
    Parameters
    ----------
    negative_dist_sq : np.ndarray
        Square of distance matrix multiplied by (-1).
    sigmas : np.ndarray, optional
        Sigma values according to desired perplexity
        Sigmas calculated using binary search, by default None.
    zero_index : int, optional
        The index to be set 0, by default None.
    Returns
    -------
    float
        Returns current perplexity result.
    """
    prob_matrix = calc_prob_matrix(negative_dist_sq, sigmas, zero_index)
    prob_matrix = torch.clip(prob_matrix, threshold, (1 - threshold))
    entropy = -torch.sum(prob_matrix * torch.log2(prob_matrix), 1)
    perplexity = 2**entropy
    return perplexity


def find_optimal_sigmas(
    negative_dist_sq: torch.Tensor,
    target_perplexity: torch.Tensor,
    threshold: torch.Tensor,
) -> torch.Tensor:
    """For each row of distances matrix, find sigma that results
    in target perplexity for that role.
    Parameters
    ----------
    negative_dist_sq : np.ndarray
        Square of distance matrix multiplied by (-1).
    target_perplexity : torch.Tensor
        Desired perplexity value.
    Returns
    -------
    np.ndarray
        Returns optimal sigma values.
    """
    sigmas: list[float] = []

    # Make fn that returns perplexity of this row given sigma
    def eval_fn(sigma: float, i: int) -> torch.Tensor:
        return perplexity_fn(negative_dist_sq[i, :], torch.tensor(sigma), i, threshold)

    # For each row of the matrix (each point in our dataset)
    for i in range(negative_dist_sq.shape[0]):
        eval_fn_p = partial(eval_fn, i=i)

        # Binary search over sigmas to achieve target perplexity
        # TODO: fix types!
        low, high = binary_search(eval_fn_p, target_perplexity, lower=0.0)
        correct_sigma = (low + high) / 2
        # Append the resulting sigma to our output array
        sigmas.append(correct_sigma)
    return torch.tensor(sigmas)


def log_sigmoid(
    input: torch.Tensor, log: Callable[..., torch.Tensor], robust: bool
) -> torch.Tensor:
    min = torch.minimum(torch.tensor(0, device=input.device, dtype=input.dtype), input)
    input = torch.exp(-torch.abs(input))
    if not robust:
        return min - torch.log1p(input)
    return min - log(1 + input)


def log_softmax(
    input: torch.Tensor, log: Callable[..., torch.Tensor], robust: bool, axis: int = -1
) -> torch.Tensor:
    if not robust:
        return torch.log_softmax(input, dim=None)
    return input - log(torch.exp(input).sum(dim=axis, keepdim=True))


def calculate_binary_class_weight(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.double()
    return (1 - labels.mean()) / labels.mean()


def calculate_categorical_class_weight(
    labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    one_hot = torch.eye(num_classes)[labels]
    return calculate_class_weight(one_hot)


def calculate_class_weight(labels: torch.Tensor) -> torch.Tensor:
    return (
        (1 / labels.sum(dim=tuple(i for i in range(labels.ndim) if i != 1)))
        * labels.sum()
        / labels.shape[1]
    )


def calculate_cross_entropy_class_weights(
    input: torch.Tensor,
    labels: torch.Tensor,
    is_categorical: bool,
    weights: bool | list[float],
) -> torch.Tensor:
    _weights = None
    if isinstance(weights, bool):
        if is_categorical:
            _weights = (
                calculate_categorical_class_weight(labels, input.size(1))
                .type(input.dtype)
                .to(input.device)
                if weights
                else torch.ones(input.size(1), dtype=input.dtype, device=input.device)
            )
        else:
            _weights = (
                calculate_class_weight(labels)
                if weights
                else torch.ones(input.size(1), dtype=input.dtype, device=input.device)
            )
    else:
        _weights = (
            torch.tensor(weights, dtype=input.dtype, device=input.device)
            .type(input.dtype)
            .to(input.device)
        )
        if _weights.ndim > 1:
            raise ValueError(f"Provided weights: '{weights}' must be 1D list.")
    if not is_categorical:
        shape = [1 for _ in range(input.ndim)]
        shape[1] = input.shape[1]
        _weights = _weights.reshape(shape)
    return _weights


def calculate_tpr_fpr(
    threshold: torch.Tensor, input: torch.Tensor, label: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    input_c = input.clone()

    n_positive = (label == 1).sum()
    n_negative = len(label) - n_positive

    input_c = torch.where(input_c >= threshold, 1, 0)
    true_positives = torch.sum((input_c == 1) & (label == 1))
    false_positives = torch.sum((input_c == 1) & (label == 0))

    fpr = false_positives / n_negative
    tpr = true_positives / n_positive
    return tpr, fpr
