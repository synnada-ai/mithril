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

import random

import numpy as np
import pytest
import torch

import mithril
from mithril import TorchBackend
from mithril.framework.common import NOT_GIVEN, ConnectionType
from mithril.models import (
    TBD,
    AbsoluteError,
    Add,
    Buffer,
    Cell,
    EncoderDecoder,
    ExtendInfo,
    IOKey,
    LSTMCell,
    ManyToOne,
    MatrixMultiply,
    OneToMany,
    ScalarItem,
    Shape,
    Sum,
    Tanh,
    TensorSlice,
    TrainModel,
)
from mithril.utils.utils import pack_data_into_time_slots

torch_to_mithril_map_without_bias = {
    "weight_ih_l0": "w_ih",
    "weight_hh_l0": "w_hh",
    "bias_ih_l0": "bias_ih",
    "bias_hh_l0": "bias_hh",
}

torch_to_mithril_map = {
    "rnn.weight_ih_l0": "w_ih",
    "rnn.weight_hh_l0": "w_hh",
    "rnn.bias_ih_l0": "bias_ih",
    "rnn.bias_hh_l0": "bias_hh",
    "fc.weight": "w_ho",
    "fc.bias": "bias_o",
}

torch_to_encoder_decoder = {
    "rnn.weight_ih_l0": "w_ih",
    "rnn.weight_hh_l0": "w_hh",
    "rnn.bias_ih_l0": "bias_ih",
    "rnn.bias_hh_l0": "bias_hh",
    "fc.weight": "w_ho",
    "fc.bias": "bias_o",
}


# modified version of
# https://www.reddit.com/r/learnpython/comments/cpwxpe/comment/ewsiwcs/ solution
def create_random_array_with_fixed_sum(n: int, total_sum: int) -> list[int]:
    """creates a random list of integers with a fixed sum and with a specified length

    Examples:
    >>> create_random_array_with_fixed_sum(4, 30)
    [3, 4, 10, 13]
    >>> create_random_array_with_fixed_sum(4, 30)
    [4, 11, 11, 4]
    >>> create_random_array_with_fixed_sum(5, 30)
    [1, 3, 7, 10, 9]

    Args:
        n (int): number of elements in the list
        total_sum (int): total sum of the list

    Returns:
        list[int]: list of randomized integers with a fixed sum
    """
    random_n_numbers = [random.random() for _ in range(n)]
    sum_n_numbers = sum(random_n_numbers)
    result = [int(num * total_sum / sum_n_numbers) for num in random_n_numbers]
    for _ in range(total_sum - sum(result)):
        result[random.randint(0, n - 1)] += 1

    # In the result, If one of the numbers is zero, increment it by 1 and decrement max
    # value by 1
    for idx, num in enumerate(result):
        if num == 0:
            max_idx = result.index(max(result))
            result[idx] += 1
            result[max_idx] -= 1

    return result


def create_random_sequence_with_variable_lengths(
    max_batch_size: int, max_seq_len: int, num_of_features: int
) -> tuple[np.ndarray, ...]:
    """Creates random sequence batches with variable lengths. returns tuple of
    numpy arrays with randomized batches and up to maximum sequence length.

    Examples:
    >>> in_1, in_2, in_3 =  create_random_sequence_with_variable_lengths(20, 3, 10)
    >>> in_1.shape
    (9, 1, 10)
    >>> in_2.shape
    (4, 2, 10)
    >>> in_3.shape
    (7, 3, 10)


    Args:
        max_batch_size (int): total batch size. function will generate max_batch_size
            batches in total.
        max_seq_len (int): maximum sequence length that function will produce. Note
            that function will generatenumpy arrays as the same as the number of
            max_seq_len
        num_of_features (int): number of features

    Returns:
        tuple[np.ndarray]: tuple of numpy arrays with randomized batch sizes and
        different sequence lengths
    """

    # randomly generates batch size for every sequence length
    sequence_batch_sizes = create_random_array_with_fixed_sum(
        max_seq_len, max_batch_size
    )
    return tuple(
        np.random.randn(*shape)
        for shape in map(
            lambda x: x[::-1] + (num_of_features,),
            enumerate(sequence_batch_sizes, start=1),
        )
    )


def unpack_inputs(inputs: tuple) -> list:
    """Unpacks the batched inputs

    Examples:
    >>> inputs = (np.random.randn(2,3,1), np.random.randn(3, 2, 1),
            np.random.randn(4,1,1))
    >>> unpacked_inputs = unpack_inputs(inputs)
    >>> [input.shape for input in unpacked_inputs]
    [(3, 1), (3, 1), (2, 1), (2, 1), (2, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

    Args:
        inputs (tuple): tuple of numpy arrays (with shape of (ni, li, ki) for each
        numpy array)

    Returns:
        list: list of unpacked numpy arrays (with shape of (li, ki)) whose lengths
          equal to sum of batches
    """

    unpacked_inputs = []
    for input in inputs:
        splitted_input = [
            np.squeeze(instance, axis=0) for instance in np.split(input, input.shape[0])
        ]
        unpacked_inputs.extend(splitted_input)
    return unpacked_inputs


class MySimpleRNNCellWithLinear(Cell):
    # Slightly modified version of RNNCell for making identical to torch's RNNCell(),
    # since Torch's RNN model has two biases
    # (see https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html) in order to
    # make it identical to Torch's model. one additional parameter of bias is added.

    shared_keys = {"w_ih", "w_hh", "w_ho", "bias_hh", "bias_ih", "bias_o"}
    state_keys = {"hidden"}
    out_key = "output"
    # output_keys = {out, hidden_compl}

    def __init__(
        self,
    ) -> None:
        super().__init__()

        shp_model = Shape()
        scalar_item = ScalarItem()
        slice_model_1 = TensorSlice(start=TBD)
        slice_model_2 = TensorSlice(stop=TBD)
        mult_model_1 = MatrixMultiply()
        mult_model_2 = MatrixMultiply()
        sum_model_1 = Add()
        sum_model_2 = Add()
        sum_model_3 = Add()
        tanh = Tanh()
        mult_model_3 = MatrixMultiply()
        sum_model_4 = Add()

        self += shp_model(input="input")
        self += scalar_item(input=shp_model.output, index=0)
        self += slice_model_1(
            input="prev_hidden", start=scalar_item.output, output=IOKey("hidden_compl")
        )
        self += slice_model_2(input="prev_hidden", stop=scalar_item.output)
        self += mult_model_1(left="input", right="w_ih")
        self += mult_model_2(left=slice_model_2.output, right="w_hh")
        self += sum_model_1(left=mult_model_1.output, right=mult_model_2.output)
        self += sum_model_2(left=sum_model_1.output, right="bias_hh")
        self += sum_model_3(
            left=sum_model_2.output,
            right="bias_ih",
        )
        self += tanh(input=sum_model_3.output, output=IOKey("hidden"))
        self += mult_model_3(left="hidden", right="w_ho")
        self += sum_model_4(
            left=mult_model_3.output, right="bias_o", output=IOKey("output")
        )

        # TODO: Commented code below does not work while above code does.
        # There may be a bug. Investigate in detail
        # self += Shape()(input = "input", output = "shp_output")
        # self += ScalarItem(index = 0)(input = "shp_output", output = "scalar_out")
        # self += TensorSlice(start = ...)(input = "prev_hidden", start = "scalar_out",
        #           output = IOKey("hidden_compl"))
        # self += TensorSlice(stop = ...)(input = "prev_hidden", stop = "scalar_out",
        #               output = "slice_out")
        # self += MatrixMultiply()(left = "input", right = "w_ih",
        #           output = "matmul1_out")
        # self += MatrixMultiply()(left = "slice_out", right = "w_hh",
        #           output = "matmul2_out")
        # self += Add()(left = "matmul2_out", right = "matmul1_out",
        #           output = "add1_out")
        # self += Add()(left = "add1_out", right = "bias_hh",
        #           output = "add2_out")
        # self += Add()(left = "add2_out", right = "bias_ih",
        #           output = "add3_out")
        # self += Tanh()(input = "add3_out", output = IOKey("hidden"))
        # self += MatrixMultiply()(left = "hidden", right = "w_ho",
        #           output = "matmul3_out")
        # self += Add()(left = "matmul3_out", right = "bias_o",
        #           output = IOKey("output"))

        ...
        shapes: dict[str, list[str | int]] = {
            "input": ["N", 1, "d_in"],
            "prev_hidden": ["M", 1, "d_hid"],
            "w_ih": ["d_in", "d_hid"],
            "w_hh": ["d_hid", "d_hid"],
            "w_ho": ["d_hid", "d_out"],
            "bias_hh": ["d_hid"],
            "bias_ih": ["d_hid"],
            "bias_o": ["d_out"],
        }

        self._set_shapes(shapes)
        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        prev_hidden: ConnectionType = NOT_GIVEN,
        w_ih: ConnectionType = NOT_GIVEN,
        w_hh: ConnectionType = NOT_GIVEN,
        w_ho: ConnectionType = NOT_GIVEN,
        bias_hh: ConnectionType = NOT_GIVEN,
        bias_ih: ConnectionType = NOT_GIVEN,
        bias_o: ConnectionType = NOT_GIVEN,
        hidden: ConnectionType = NOT_GIVEN,
        hidden_compl=NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {
            "input": input,
            "prev_hidden": prev_hidden,
            "w_ih": w_ih,
            "w_hh": w_hh,
            "w_ho": w_ho,
            "bias_hh": bias_hh,
            "bias_ih": bias_ih,
            "bias_o": bias_o,
            "hidden": hidden,
            "hidden_compl": hidden_compl,
            "output": output,
        }
        return ExtendInfo(self, kwargs)


class MyRNNCell(Cell):
    # RNN Cell Created for test purposes. Original RNNCell modified to make identical
    # to Torch's RNN model. It is similar to MySimpleRNNCellWithLinear. Only
    # difference is MyRNNCell() does not have a linear layer
    shared_keys = {"w_ih", "w_hh", "bias_hh", "bias_ih"}
    state_keys = {"hidden"}
    out_key = "hidden"
    # output_keys = {out, hidden_compl}

    def __init__(
        self,
    ) -> None:
        super().__init__()

        shp_model = Shape()
        scalar_item = ScalarItem()
        slice_model_1 = TensorSlice(start=TBD)
        slice_model_2 = TensorSlice(stop=TBD)
        mult_model_1 = MatrixMultiply()
        mult_model_2 = MatrixMultiply()
        sum_model_1 = Add()
        sum_model_2 = Add()
        sum_model_3 = Add()
        tanh = Tanh()

        self += shp_model(input="input")
        self += scalar_item(input=shp_model.output, index=0)
        self += slice_model_1(
            input="prev_hidden", start=scalar_item.output, output=IOKey("hidden_compl")
        )
        self += slice_model_2(input="prev_hidden", stop=scalar_item.output)
        self += mult_model_1(left="input", right="w_ih")
        self += mult_model_2(left=slice_model_2.output, right="w_hh")
        self += sum_model_1(left=mult_model_1.output, right=mult_model_2.output)
        self += sum_model_2(left=sum_model_1.output, right="bias_hh")
        self += sum_model_3(
            left=sum_model_2.output,
            right="bias_ih",
        )
        self += tanh(input=sum_model_3.output, output=IOKey("hidden"))

        shapes: dict[str, list[str | int]] = {
            "input": ["N", 1, "d_in"],
            "prev_hidden": ["N", 1, "d_hid"],
            "w_ih": ["d_in", "d_hid"],
            "w_hh": ["d_hid", "d_hid"],
            "bias_hh": ["d_hid"],
            "bias_ih": ["d_hid"],
        }
        self._set_shapes(shapes)
        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        prev_hidden: ConnectionType = NOT_GIVEN,
        w_ih: ConnectionType = NOT_GIVEN,
        w_hh: ConnectionType = NOT_GIVEN,
        bias_hh: ConnectionType = NOT_GIVEN,
        bias_ih: ConnectionType = NOT_GIVEN,
        hidden: ConnectionType = NOT_GIVEN,
        hidden_compl=NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {
            "input": input,
            "prev_hidden": prev_hidden,
            "w_ih": w_ih,
            "w_hh": w_hh,
            "bias_hh": bias_hh,
            "bias_ih": bias_ih,
            "hidden": hidden,
            "hidden_compl": hidden_compl,
        }
        return ExtendInfo(self, kwargs)


class ManyToOneWithLinear(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ManyToOneWithLinear, self).__init__()  # noqa UP008
        self.hidden_dim = hidden_dim
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, init_h):
        out, hn = self.rnn(x, init_h)
        out = self.fc(out)
        return out, hn


class LSTMLinear(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMLinear, self).__init__()  # noqa UP008
        self.hidden_dim = hidden_dim
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, init_h, init_c):
        out, h = self.rnn(x, (init_h, init_c))
        out = self.fc(out)
        return out, h


class TorchOneToMany(torch.nn.Module):
    """
    One to Many model implemented in pytorch.
    """

    def __init__(self, input_features, hidden_features, sequence_length):
        super(TorchOneToMany, self).__init__()  # noqa UP008
        self.input_size = input_features
        self.hidden_size = hidden_features
        self.seq_len = sequence_length
        self.rnn = torch.nn.RNN(input_features, hidden_features, batch_first=True)
        self.fc = torch.nn.Linear(hidden_features, input_features)

    def forward(self, x, init_h, lengths=None):
        outputs = []
        out, hn = self.rnn(x, init_h)
        out = self.fc(out)
        outputs.append(out)
        for _ in range(self.seq_len - 1):
            out, hn = self.rnn(out, hn)
            out = self.fc(out)
            outputs.append(out)
        outputs = torch.cat(outputs, dim=1)  # type: ignore
        return outputs


class TorchEncoderDecoder(torch.nn.Module):
    """Encoder Decoder implementation in pytorch. Many to one model is used as
    encoder and one to many model is used as a decoder.
    """

    def __init__(self, input_features, hidden_features, decoder_seq_len):
        super(TorchEncoderDecoder, self).__init__()  # noqa UP008
        self.encoder = torch.nn.RNN(input_features, hidden_features, batch_first=True)
        self.decoder = TorchOneToMany(input_features, hidden_features, decoder_seq_len)

    def forward(self, x, init_h, decoder_input):
        output, hn = self.encoder(x, init_h)
        outputs = self.decoder(decoder_input, hn)
        return outputs


def test_rnn_many_to_one():
    # NOTE: Does not work when max_seq_length = 1.
    """
    This test aims to test most simple and primitive building block of RNN's.
    This test runs 15 times in single run. In all runs, parameters of rnn's
    (batch size, input features, hidden features, max sequence length) is randomized.
    These randomized parameters feed into pure torch model and Mithril model.
    Then both models calculates the grads and outputs, It is expected that all outputs
    and grads of both models are the same.
    """
    for _ in range(15):
        batch_size = np.random.randint(1, 11)
        input_features = np.random.randint(1, 15)
        hidden_features = np.random.randint(1, 20)
        max_seq_length = np.random.randint(2, 8)

        torch_model = torch.nn.RNN(input_features, hidden_features, batch_first=True)
        model_trainable_params = torch_model.state_dict()

        mithril_model = ManyToOne(
            cell_type=MyRNNCell(), max_sequence_length=max_seq_length
        )
        ctx = TrainModel(mithril_model)
        ctx.add_loss(
            Buffer(), input=f"output{max_seq_length - 1}", reduce_steps=[Sum()]
        )

        input = torch.randn(batch_size, max_seq_length, input_features)
        h0 = torch.randn(1, batch_size, hidden_features)

        input_dict = {
            f"input{idx}": input[:, idx : idx + 1, :] for idx in range(max_seq_length)
        }
        input_dict["initial_hidden"] = torch.swapaxes(h0, 0, 1)

        comp_model = mithril.compile(
            model=ctx, backend=TorchBackend(), static_keys=input_dict
        )

        randomized_params = {
            torch_to_mithril_map_without_bias[key]: value.permute(
                *torch.arange(value.ndim - 1, -1, -1)
            )
            for key, value in model_trainable_params.items()
        }

        output_coml, grads_coml = comp_model.evaluate_all(randomized_params)
        output_torch, _ = torch_model(input, h0)
        output_torch_dict = {
            f"output{idx}": output_torch[:, idx : idx + 1, :]
            for idx in range(max_seq_length)
        }
        loss = output_torch_dict[f"output{max_seq_length - 1}"].sum()
        loss.backward()

        for key in output_torch_dict:
            torch.testing.assert_close(output_torch_dict[key], output_coml[key])

        composite_ml_grad_order = ["w_ih", "w_hh", "bias_ih", "bias_hh"]
        for idx, value in enumerate(torch_model.parameters()):
            grad = grads_coml[composite_ml_grad_order[idx]]
            grad = grad.permute(*torch.arange(grad.ndim - 1, -1, -1))
            torch.testing.assert_close(grad, value.grad)


def test_rnn_many_to_one_with_linear():
    # NOTE: Does not work when max_seq_length = 1.
    """
    This test is very similar to test_rnn_many_to_one. This test is tests RNN model
    with linear added output
    """
    for _ in range(15):
        batch_size = np.random.randint(1, 11)
        input_features = np.random.randint(1, 15)
        hidden_features = np.random.randint(1, 20)
        max_seq_length = np.random.randint(2, 8)
        output_features = np.random.randint(5, 20)

        torch_model = ManyToOneWithLinear(
            input_features, hidden_features, output_features
        )
        model_trainable_params = torch_model.state_dict()

        mithril_model = ManyToOne(
            cell_type=MySimpleRNNCellWithLinear(), max_sequence_length=max_seq_length
        )
        ctx = TrainModel(mithril_model)
        ctx.add_loss(
            Buffer(), input=f"output{max_seq_length - 1}", reduce_steps=[Sum()]
        )

        input = torch.randn(batch_size, max_seq_length, input_features)
        h0 = torch.randn(1, batch_size, hidden_features)

        input_dict = {
            f"input{idx}": input[:, idx : idx + 1, :] for idx in range(max_seq_length)
        }
        input_dict["initial_hidden"] = torch.swapaxes(h0, 0, 1)

        comp_model = mithril.compile(
            model=ctx, backend=TorchBackend(), static_keys=input_dict
        )

        randomized_params = {
            torch_to_mithril_map[key]: value.permute(
                *torch.arange(value.ndim - 1, -1, -1)
            )
            for key, value in model_trainable_params.items()
        }

        output_coml, grads_coml = comp_model.evaluate_all(randomized_params)
        output_torch, _ = torch_model(input, h0)
        output_torch_dict = {
            f"output{idx}": output_torch[:, idx : idx + 1, :]
            for idx in range(max_seq_length)
        }
        loss = output_torch_dict[f"output{max_seq_length - 1}"].sum()
        loss.backward()

        for key in output_torch_dict:
            torch.testing.assert_close(output_torch_dict[key], output_coml[key])

        composite_ml_grad_order = [
            "w_ih",
            "w_hh",
            "bias_ih",
            "bias_hh",
            "w_ho",
            "bias_o",
        ]
        for idx, value in enumerate(torch_model.parameters()):
            grad = grads_coml[composite_ml_grad_order[idx]]
            grad = grad.permute(*torch.arange(grad.ndim - 1, -1, -1))
            torch.testing.assert_close(grad, value.grad)


def test_rnn_one_to_many():
    for _ in range(15):
        batch_size = np.random.randint(1, 11)
        input_features = np.random.randint(1, 15)
        hidden_features = np.random.randint(1, 20)
        max_seq_length = np.random.randint(2, 8)
        model = OneToMany(
            cell_type=MySimpleRNNCellWithLinear(),
            max_sequence_length=max_seq_length,
            teacher_forcing=False,
        )
        ctx = TrainModel(model)
        for idx in range(max_seq_length):
            ctx.add_loss(
                AbsoluteError(),
                input=f"output{idx}",
                target=f"target{idx}",
                reduce_steps=[Sum()],
            )

        torch_model = TorchOneToMany(input_features, hidden_features, max_seq_length)

        model_trainable_params = torch_model.state_dict()
        input = torch.randn(batch_size, 1, input_features)
        h0 = torch.randn(1, batch_size, hidden_features)
        targets = torch.randn(batch_size, max_seq_length, input_features)

        input_dict = {
            f"target{idx}": targets[:, idx : idx + 1, :]
            for idx in range(max_seq_length)
        }
        input_dict["initial_hidden"] = torch.swapaxes(h0, 0, 1)
        input_dict["input"] = input

        comp_model = mithril.compile(
            model=ctx, backend=TorchBackend(), static_keys=input_dict
        )
        randomized_params = {
            torch_to_mithril_map[key]: value.permute(
                *torch.arange(value.ndim - 1, -1, -1)
            )
            for key, value in model_trainable_params.items()
        }
        output_coml, grads_coml = comp_model.evaluate_all(randomized_params)
        output_torch = torch_model(input, h0)
        output_torch_dict = {
            f"output{idx}": output_torch[:, idx : idx + 1, :]
            for idx in range(max_seq_length)
        }
        loss = torch.tensor(0.0)
        for idx, key in enumerate(output_torch_dict):
            loss += torch.abs(output_torch_dict[key] - input_dict[f"target{idx}"]).sum()
        loss.backward()

        for key in output_torch_dict:
            torch.testing.assert_close(output_torch_dict[key], output_coml[key])

        composite_ml_grad_order = [
            "w_ih",
            "w_hh",
            "bias_ih",
            "bias_hh",
            "w_ho",
            "bias_o",
        ]
        for idx, value in enumerate(torch_model.parameters()):
            grad = grads_coml[composite_ml_grad_order[idx]]
            grad = grad.permute(*torch.arange(grad.ndim - 1, -1, -1))
            torch.testing.assert_close(grad, value.grad)


@pytest.mark.skip(reason="This test will be enabled after fixing the tolerance issue.")
def test_torch_encoder_decoder_var_seq_len():
    """This test aims to test encoder decoder architecture of Mithril
    library when input's and targets's length is not fixed. This test ramdomizes
    the inputs and targets with variable lengths, feed it to Mithril's
    encoder decoder model, and then evaluate outputs and gradients. Having evaluated
    outputs and gradients of the Mithril's encoder decoder, then evaluate outputs
    and gradients with pure torch model and compare the results.
    """
    # set conffigurations
    batch_size = 2000
    max_seq_len = 5
    input_dim = 2
    hidden_dim = 4

    # define backend
    backend = TorchBackend()
    # import sys
    # sys.setrecursionlimit(1500)

    # Create random input and target with variable lengths
    inputs = create_random_sequence_with_variable_lengths(
        batch_size, max_seq_len, input_dim
    )
    targets = create_random_sequence_with_variable_lengths(
        batch_size, max_seq_len, input_dim
    )

    # Unpack the inputs and targets and cast them to defined backend
    unpacked_inputs = [backend.array(input) for input in unpack_inputs(inputs)]
    unpacked_targets = [backend.array(target) for target in unpack_inputs(targets)]

    # Shuffle the inputs and targets seperately
    random.shuffle(unpacked_inputs)
    random.shuffle(unpacked_targets)

    # zip inputs and targets into a list
    data = list(zip(unpacked_inputs, unpacked_targets, strict=False))

    # Pack the inputs into their corresponding time slots
    train_time_inputs, data_sorted_wrt_input = pack_data_into_time_slots(
        backend=backend, data=data, key=("input",), index=0
    )

    # Pack the targets into their corresponding time slots and also return indices
    train_indices, train_time_targets, _ = pack_data_into_time_slots(
        backend=backend,
        data=data_sorted_wrt_input,
        key=("target",),
        index=1,
        return_indices=True,
    )

    # define static inputs to be used in compile
    static_inputs = (
        {
            "initial_hidden": torch.randn(batch_size, 1, hidden_dim),
            "decoder_input": torch.randn(batch_size, 1, input_dim),
            "indices": backend.array(train_indices),
        }
        | train_time_inputs
        | train_time_targets
    )

    # define Encoder Decoder model and wrap it with encoderdecoder model
    model = EncoderDecoder(
        cell_type=MySimpleRNNCellWithLinear(),
        max_input_sequence_length=max_seq_len,
        max_target_sequence_length=max_seq_len,
    )
    ctx = TrainModel(model)

    # attach losses to the outputs of the encoder decoder model
    for idx in range(max_seq_len):
        ctx.add_loss(
            AbsoluteError(),
            input=f"output{idx}",
            target=f"target{idx}",
            reduce_steps=[Sum()],
        )

    # compile the model
    comp_model = mithril.compile(model=ctx, backend=backend, static_keys=static_inputs)

    # generate trainable inputs and get outputs and gradients of Mithril
    trainable_inputs = comp_model.randomize_params()

    # TODO: After implemetation of automatic weight initialization
    # algorithm, this part will be removed.
    # For now, we need to lower the scale of the weights.
    for key, value in trainable_inputs.items():
        trainable_inputs[key] = value * 0.1

    outputs, gradients_mithril = comp_model.evaluate_all(trainable_inputs)

    ############# Torch Model #############

    # unpack the zipped data into inputs and targets
    torch_inputs, torch_targets = zip(*data, strict=False)

    # pad them to feed torch's RNN model
    torch_inputs = torch.nn.utils.rnn.pad_sequence(torch_inputs, batch_first=True)  # type: ignore
    torch_targets = torch.nn.utils.rnn.pad_sequence(torch_targets, batch_first=True)  # type: ignore

    # Define the torch model
    torch_model = TorchEncoderDecoder(input_dim, hidden_dim, max_seq_len)
    state_dict = torch_model.state_dict()
    torch_randomized_inputs = {
        key: value.permute(*reversed(range(value.ndim)))
        for key, value in trainable_inputs.items()
    }

    key_map = {
        "encoder.weight_ih_l0": "w_ih",
        "encoder.weight_hh_l0": "w_hh",
        "encoder.bias_ih_l0": "bias_ih",
        "encoder.bias_hh_l0": "bias_hh",
        "decoder.rnn.weight_ih_l0": "decoder_w_ih",
        "decoder.rnn.weight_hh_l0": "decoder_w_hh",
        "decoder.rnn.bias_ih_l0": "decoder_bias_ih",
        "decoder.rnn.bias_hh_l0": "decoder_bias_hh",
        "decoder.fc.weight": "decoder_w_ho",
        "decoder.fc.bias": "decoder_bias_o",
    }

    for key in state_dict:
        state_dict[key] = torch_randomized_inputs[key_map[key]]

    torch_model.load_state_dict(state_dict)

    output = torch_model(
        torch_inputs,
        torch.swapaxes(static_inputs["initial_hidden"], 0, 1),
        static_inputs["decoder_input"],
    )
    for idx in range(max_seq_len):
        mithril_output = outputs[f"output{idx}"]
        torch.testing.assert_close(
            mithril_output,
            output[: mithril_output.shape[0], idx : idx + 1, :],
        )

    loss = torch.tensor(0.0)
    for idx in range(max_seq_len):
        target = train_time_targets[f"target{idx}"]
        loss += torch.abs(output[: target.shape[0], idx : idx + 1, :] - target).sum()
    loss.backward()

    for key, value in torch_model.named_parameters():
        grad = gradients_mithril[key_map.get(key)]
        grad = grad.permute(*torch.arange(grad.ndim - 1, -1, -1))
        torch.testing.assert_close(grad, value.grad)


@pytest.mark.skip(reason="This test will be enabled after fixing the tolerance issue.")
def test_lstm_many_to_one():
    backend = TorchBackend()
    for _ in range(15):
        # select random feature sizes
        batch_size = np.random.randint(40, 50)
        input_features = np.random.randint(20, 25)
        hidden_features = np.random.randint(30, 40)
        max_seq_length = np.random.randint(2, 8)
        output_features = np.random.randint(1, 20)

        # initialize Torch model
        torch_model = LSTMLinear(input_features, hidden_features, output_features)

        # initialize Mithril model
        mithril_model = ManyToOne(
            cell_type=LSTMCell(), max_sequence_length=max_seq_length
        )

        # create TrainModel that only sums the last oputput
        ctx = TrainModel(mithril_model)
        ctx.add_loss(
            Buffer(), input=f"output{max_seq_length - 1}", reduce_steps=[Sum()]
        )

        # set shapes of the Mithril model woth given fetures
        for idx in range(max_seq_length):
            ctx.set_shapes({f"input{idx}": [batch_size, 1, input_features]})
        ctx.set_shapes({"initial_hidden": [batch_size, 1, hidden_features]})
        ctx.set_shapes({"output1": [batch_size, 1, output_features]})

        # initialize static torch inputs
        inputs = torch.randn(batch_size, max_seq_length, input_features)
        h0 = torch.randn(1, batch_size, hidden_features)
        c0 = torch.randn(1, batch_size, hidden_features)

        # initialize static keys dict
        static_keys = {
            f"input{idx}": inputs[:, idx : idx + 1, :] for idx in range(max_seq_length)
        }
        static_keys["initial_hidden"] = torch.swapaxes(h0, 0, 1)
        static_keys["initial_cell"] = torch.swapaxes(c0, 0, 1)

        # compile the given model
        pm_lstm = mithril.compile(
            model=ctx, backend=backend, static_keys=static_keys, jit=True
        )

        # randomize the parameters
        randomized_params = pm_lstm.randomize_params()

        # TODO: After implemetation of automatic weight initialization
        # algorithm, this part will be removed.
        # For now, we need to lower the scale of the weights.
        for key, value in randomized_params.items():
            randomized_params[key] = value

        # In Standard LSTM model, there are 8 weights.
        # However, LSTM model of Mithril concats the weights
        # of hidden and input vectors.
        # Therefore, 8 standard weights are reduced to 4.
        # Following step extracts 8 standard weights
        w_ig = randomized_params["w_c"][:input_features].T
        w_hg = randomized_params["w_c"][input_features:].T

        w_if = randomized_params["w_f"][:input_features].T
        w_hf = randomized_params["w_f"][input_features:].T

        w_ii = randomized_params["w_i"][:input_features].T
        w_hi = randomized_params["w_i"][input_features:].T

        w_io = randomized_params["w_o"][:input_features].T
        w_ho = randomized_params["w_o"][input_features:].T

        # Torch LSTM Model concats the given weights in this way
        # see https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        weight_ih_l0 = torch.concat((w_ii, w_if, w_ig, w_io), dim=0)
        weight_hh_l0 = torch.concat((w_hi, w_hf, w_hg, w_ho), dim=0)

        # do the same with biases
        b_ig = randomized_params["bias_c"][:]
        b_if = randomized_params["bias_f"][:]
        b_ii = randomized_params["bias_i"][:]
        b_io = randomized_params["bias_o"][:]

        bias_ih_l0 = torch.concat((b_ii, b_if, b_ig, b_io), dim=0)

        # Contrary to standard definition, Torch uses two
        # biases in their rnn models. Set to zeros one
        # of the biases
        bias_hh_l0 = torch.zeros(4 * hidden_features)

        # take linear weights and linear biases
        weight = randomized_params["w_out"].T
        bias = randomized_params["bias_out"][:]

        state_dict = {
            "rnn.weight_ih_l0": weight_ih_l0,
            "rnn.weight_hh_l0": weight_hh_l0,
            "rnn.bias_ih_l0": bias_ih_l0,
            "rnn.bias_hh_l0": bias_hh_l0,
            "fc.weight": weight,
            "fc.bias": bias,
        }
        # load the torch parameters
        torch_model.load_state_dict(state_dict)

        outputs_mithril, grads_mithril = pm_lstm.evaluate_all(randomized_params)
        output_torch, _ = torch_model(inputs, h0, c0)
        loss = output_torch[:, -1, :].sum()
        loss.backward()

        # do the same with grad
        w_ig_grad = grads_mithril["w_c"][:input_features].T
        w_hg_grad = grads_mithril["w_c"][input_features:].T

        w_if_grad = grads_mithril["w_f"][:input_features].T
        w_hf_grad = grads_mithril["w_f"][input_features:].T

        w_ii_grad = grads_mithril["w_i"][:input_features].T
        w_hi_grad = grads_mithril["w_i"][input_features:].T

        w_io_grad = grads_mithril["w_o"][:input_features].T
        w_ho_grad = grads_mithril["w_o"][input_features:].T

        weight_ih_l0_grad = torch.concat(
            (w_ii_grad, w_if_grad, w_ig_grad, w_io_grad), dim=0
        )
        weight_hh_l0_grad = torch.concat(
            (w_hi_grad, w_hf_grad, w_hg_grad, w_ho_grad), dim=0
        )

        b_ig_grad = grads_mithril["bias_c"][:]
        b_if_grad = grads_mithril["bias_f"][:]
        b_ii_grad = grads_mithril["bias_i"][:]
        b_io_grad = grads_mithril["bias_o"][:]

        bias_ih_l0_grad = torch.concat(
            (b_ii_grad, b_if_grad, b_ig_grad, b_io_grad), dim=0
        )

        weight_grad = grads_mithril["w_out"].T
        bias_grad = grads_mithril["bias_out"]

        grad_dict = {
            "rnn.weight_ih_l0": weight_ih_l0_grad,
            "rnn.weight_hh_l0": weight_hh_l0_grad,
            "rnn.bias_ih_l0": bias_ih_l0_grad,
            "fc.weight": weight_grad,
            "fc.bias": bias_grad,
        }

        # finally, assert both model's outputs
        for idx in range(max_seq_length):
            output_mithril = outputs_mithril[f"output{idx}"]
            torch_out = output_torch[:, idx : idx + 1, :]
            torch.testing.assert_close(torch_out, output_mithril)

        # assert gradients
        for name, param in torch_model.named_parameters():
            if name == "rnn.bias_hh_l0":
                continue
            grad_mithril = grad_dict[name]
            torch.testing.assert_close(param.grad, grad_mithril)
