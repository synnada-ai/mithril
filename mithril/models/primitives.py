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

from __future__ import annotations

from types import NoneType

from ..core import Constant
from ..framework.common import (
    NOT_GIVEN,
    TBD,
    Connection,
    ConnectionType,
    GenericTensorType,
    IOKey,
    MyTensor,
    TensorValueType,
    ToBeDetermined,
)
from ..framework.constraints import (
    arange_constraints,
    bcast,
    broadcast_to_constraints,
    concat_constraints,
    conv_1d_constraints,
    conv_2d_constraints,
    cross_entropy_constraint,
    eye_constraints,
    flatten_constrains,
    general_tensor_type_constraint,
    pad_constraints,
    padding_1d_constraint,
    padding_2d_constraint,
    polynomial_features_constraints,
    sliding_window_1d_constraints,
    sliding_window_2d_constraints,
    squeeze_constraints,
    stride_constraint,
    swap_axes_constraints,
    tuple_converter_constraint,
    where_constrains,
)
from ..framework.logical.base import BaseModel
from ..framework.logical.essential_primitives import SingleInputOperation
from ..models import ExtendInfo, PrimitiveModel
from ..utils.utils import PaddingType

__all__ = [
    "CustomPrimitiveModel",
    "Activation",
    "SquaredError",
    "AbsoluteError",
    "QuantileLoss",
    "CrossEntropy",
    "KLDivergence",
    "BinaryCrossEntropy",
    "HingeLoss",
    "QuadHingeLoss",
    "Sine",
    "Cosine",
    "Sign",
    "Square",
    "Log",
    "StableReciprocal",
    "Relu",
    "LeakyRelu",
    "Sigmoid",
    "Softmax",
    "Softplus",
    "Tanh",
    "CartesianDifference",
    "Concat",
    "PermuteTensor",
    "PrimitiveConvolution1D",
    "PrimitiveConvolution2D",
    "Flatten",
    "PrimitiveMaxPool2D",
    "PrimitiveMaxPool1D",
    "NormModifier",
    "DistanceMatrix",
    "PolynomialFeatures",
    "TsnePJoint",
    "EyeComplement",
    "Eye",
    "ZerosLike",
    "Cholesky",
    "GPRAlpha",
    "GPRVOuter",
    "TransposedDiagonal",
    "Eigvalsh",
    "Squeeze",
    "AUCCore",
    "Embedding",
    "PositionalEncoding",
    "SwapAxes",
    "Where",
    "Arange",
    "BroadcastTo",
    "Gelu",
    "PrimitiveUnion",
    "ScaledDotProduct",
    "IsNan",
    "NanToNum",
    "PaddingConverter1D",
    "PaddingConverter2D",
    "StrideConverter",
    "TupleConverter",
    "Unique",
    "Trapezoid",
    "Pad",
]

# Define types used to define keys:
ConstantType = float | int | Constant


class CustomPrimitiveModel(PrimitiveModel):
    def __init__(
        self, formula_key: str, name: str | None = None, **kwargs: IOKey
    ) -> None:
        self.factory_args = {"formula_key": formula_key} | kwargs
        super().__init__(formula_key=formula_key, name=name, **kwargs)


########################## Supervised Loss Types ##########################
class SupervisedLoss(PrimitiveModel):
    """Base class for supervised losses with one input and a target.
    Takes N-dimensional input and target and produces N-dimensional output.

    Parameters
    ----------
    PrimitiveModel : _type_
        _description_
    """

    input: Connection
    target: Connection
    output: Connection

    def __init__(
        self,
        formula_key: str,
        polymorphic_constraint: bool = True,
        name: str | None = None,
        **kwargs: IOKey,
    ) -> None:
        default_kwargs: dict[str, IOKey] = {
            "output": IOKey(shape=[("Var_1", ...)], type=GenericTensorType),
            "input": IOKey(shape=[("Var_2", ...)], type=GenericTensorType),
            "target": IOKey(shape=[("Var_3", ...)], type=GenericTensorType),
        }
        # Finalize kwargs.
        kwargs = default_kwargs | kwargs
        super().__init__(formula_key=formula_key, name=name, **kwargs)

        # Set constraints.
        self._set_constraint(
            fn=bcast, keys=[PrimitiveModel.output_key, "input", "target"]
        )
        if polymorphic_constraint:
            self._set_constraint(
                fn=general_tensor_type_constraint,
                keys=[PrimitiveModel.output_key, "input", "target"],
            )

        self.safe_shapes = {
            "output": ["N", ("Var", ...)],
            "input": ["N", ("Var", ...)],
            "target": ["N", ("Var", ...)],
        }

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        target: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, target=target, output=output)


class SquaredError(SupervisedLoss):
    def __init__(
        self,
        name: str | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
        target: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(formula_key="squared_error", name=name)
        self.factory_inputs = {"input": input, "target": target}


class AbsoluteError(SupervisedLoss):
    def __init__(
        self,
        name: str | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
        target: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(formula_key="absolute_error", name=name)
        self.factory_inputs = {"input": input, "target": target}


class HingeLoss(SupervisedLoss):
    def __init__(
        self,
        name: str | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
        target: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            polymorphic_constraint=False,
            formula_key="hinge_loss",
            name=name,
            output=IOKey(shape=["N", ("Var", ...)], type=MyTensor[float]),
        )
        self.factory_inputs = {"input": input, "target": target}


class QuadHingeLoss(SupervisedLoss):
    def __init__(
        self,
        name: str | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
        target: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            polymorphic_constraint=False,
            formula_key="quad_hinge_loss",
            name=name,
            output=IOKey(shape=["N", ("Var", ...)], type=MyTensor[float]),
        )
        self.factory_inputs = {"input": input, "target": target}


class QuantileLoss(PrimitiveModel):
    """
    Takes N-dimensional input and target and produces N-dimensional output.
    """

    input: Connection
    target: Connection
    quantile: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        quantile: int | float | ToBeDetermined = TBD,
        input: TensorValueType | ToBeDetermined = TBD,
        target: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="quantile_loss",
            name=name,
            output=IOKey(shape=[("Var_1", ...)], type=GenericTensorType),
            input=IOKey(shape=[("Var_2", ...)], type=GenericTensorType),
            target=IOKey(shape=[("Var_3", ...)], type=GenericTensorType),
            quantile=IOKey(shape=[], type=MyTensor[int] | MyTensor[float]),
        )
        self.factory_inputs = {"input": input, "target": target, "quantile": quantile}

        self._set_constraint(
            fn=bcast, keys=[PrimitiveModel.output_key, "input", "target"]
        )
        self._set_constraint(
            fn=general_tensor_type_constraint,
            keys=[PrimitiveModel.output_key, "input", "target", "quantile"],
        )

        self.safe_shapes = {
            "output": ["N", ("Var", ...)],
            "input": ["N", ("Var", ...)],
            "target": ["N", ("Var", ...)],
        }

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        target: ConnectionType = NOT_GIVEN,
        quantile: int | float | ConnectionType = 0.5,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input, target=target, quantile=quantile, output=output
        )


class CrossEntropy(PrimitiveModel):
    """
    If categorical = True:
        Takes N-dimensional input and (N-1)-dimensional target and
        produces (N-1)-dimensional output.
    else:
        Takes N-dimensional input and target and produces (N-1)-dimensional output.
    """

    input: Connection
    target: Connection
    weights: Connection
    cutoff: Connection
    robust: Connection
    output: Connection

    def __init__(
        self,
        input_type: str = "logits",
        weights: list[float] | str = "",
        name: str | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
        target: TensorValueType | ToBeDetermined = TBD,
        robust: bool | ToBeDetermined = TBD,
        cutoff: ConstantType | ToBeDetermined = TBD,
    ) -> None:
        self.factory_args = {"input_type": input_type, "weights": weights}

        weights_type: type = list[float]
        if isinstance(weights, str):
            if weights not in ("", "auto"):
                raise ValueError(
                    f"weights can only be set to 'auto' in string format, got {weights}"
                )
            final_weights: list[float] | bool = weights.lower() == "auto"
            weights_type = bool
        else:
            final_weights = weights

        kwargs: dict[str, IOKey] = {
            "output": IOKey(shape=["N", ("Var", ...)], type=MyTensor[float]),
            "input": IOKey(shape=["N", "C", ("Var", ...)], type=GenericTensorType),
            "target": IOKey(shape=["N", ("VarTarget", ...)], type=GenericTensorType),
            "weights": IOKey(type=weights_type, value=final_weights),
            "categorical": IOKey(type=bool),
            "cutoff": IOKey(shape=[], type=GenericTensorType),
            "robust": IOKey(type=bool),
        }

        if input_type == "logits":
            formula_key = "cross_entropy_with_logits"
        elif input_type == "probs":
            formula_key = "cross_entropy"
        elif input_type == "log_probs":
            formula_key = "cross_entropy_with_log_probs"
            kwargs.pop("cutoff")
            kwargs.pop("robust")
        else:
            raise ValueError(
                f"Cross entropy does not support '{input_type}' input type. "
                " Available input types: 'logits', 'probs', and 'log_probs'."
            )

        super().__init__(formula_key=formula_key, name=name, **kwargs)
        self.factory_inputs = {
            "input": input,
            "target": target,
            "robust": robust,
            "cutoff": cutoff,
        }

        self._set_constraint(
            fn=cross_entropy_constraint, keys=["categorical", "input", "target"]
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        target: ConnectionType = NOT_GIVEN,
        weights: ConnectionType = NOT_GIVEN,
        categorical: bool | ConnectionType = True,
        cutoff: ConstantType | ConnectionType = Constant.MIN_POSITIVE_NORMAL,
        robust: bool | ConnectionType = False,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {
            "input": input,
            "target": target,
            "weights": weights,
            "categorical": categorical,
            "output": output,
        }
        # Check if the given argument set is valid.
        if self._formula_key == "cross_entropy_with_log_probs":
            args: list[str] = []
            if robust is not False:
                args.append("robust")
            if cutoff != Constant.MIN_POSITIVE_NORMAL:
                args.append("cutoff")
            if args:
                raise ValueError(
                    f"Cross entropy with log probs does not accept {args} arguments."
                )
        else:
            kwargs |= {"cutoff": cutoff, "robust": robust}

        return super().__call__(**kwargs)


class KLDivergence(PrimitiveModel):
    """
    Takes N-dimensional input and target and produces N-dimensional output.
    """

    input: Connection
    target: Connection
    cutoff: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
        target: TensorValueType | ToBeDetermined = TBD,
        cutoff: ConstantType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="kl_divergence",
            name=name,
            output=IOKey(shape=[("Var_1", ...)], type=MyTensor[float]),
            input=IOKey(shape=[("Var_2", ...)], type=GenericTensorType),
            target=IOKey(shape=[("Var_3", ...)], type=GenericTensorType),
            cutoff=IOKey(shape=[], type=GenericTensorType),
        )
        self.factory_inputs = {"input": input, "target": target, "cutoff": cutoff}

        self.safe_shapes = {
            "output": ["N", ("Var", ...)],
            "input": ["N", ("Var", ...)],
            "target": ["N", ("Var", ...)],
        }
        self._set_constraint(
            fn=bcast, keys=[PrimitiveModel.output_key, "input", "target"]
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        target: ConnectionType = NOT_GIVEN,
        cutoff: ConnectionType = Constant.MIN_POSITIVE_NORMAL,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input, target=target, cutoff=cutoff, output=output
        )


class BinaryCrossEntropy(PrimitiveModel):
    """
    Takes N-dimensional input and target and produces N-dimensional output.
    """

    input: Connection
    target: Connection
    pos_weight: Connection
    cutoff: Connection
    robust: Connection
    output: Connection

    def __init__(
        self,
        input_type: str = "logits",
        pos_weight: float | str | ToBeDetermined = 1.0,
        name: str | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
        target: TensorValueType | ToBeDetermined = TBD,
        cutoff: ConstantType | ToBeDetermined = TBD,
        robust: bool | ToBeDetermined = TBD,
    ) -> None:
        self.factory_args = {"input_type": input_type, "pos_weight": pos_weight}

        if isinstance(pos_weight, str):
            if pos_weight != "auto":
                raise ValueError(
                    "pos_weight can only be set to 'auto' in string format,"
                    " got {pos_weight}"
                )
            pos_weight = True

        pos_weight_type = (
            float | bool if pos_weight in (..., None) else type(pos_weight)
        )
        kwargs: dict[str, IOKey] = {
            "output": IOKey(shape=[("Var_out", ...)], type=MyTensor[float]),
            "input": IOKey(shape=[("Var_out", ...)], type=GenericTensorType),
            "target": IOKey(
                shape=[("Var_out", ...)], type=MyTensor[int] | MyTensor[float]
            ),
            "pos_weight": IOKey(type=pos_weight_type, value=pos_weight),
            "cutoff": IOKey(shape=[], type=GenericTensorType),
            "robust": IOKey(type=bool),
        }

        if input_type == "logits":
            formula_key = "binary_cross_entropy_with_logits"
        elif input_type == "probs":
            formula_key = "binary_cross_entropy"
        else:
            raise ValueError(f"Binary Cross Entropy does not support \
                             '{input_type}' input type. Available    \
                             input types: 'logits' and 'probs'.")

        super().__init__(formula_key=formula_key, name=name, **kwargs)
        self.factory_inputs = {
            "input": input,
            "target": target,
            "cutoff": cutoff,
            "robust": robust,
        }

        self._set_constraint(
            fn=bcast, keys=[PrimitiveModel.output_key, "input", "target"]
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        target: ConnectionType = NOT_GIVEN,
        pos_weight: ConnectionType = NOT_GIVEN,
        cutoff: ConstantType | ConnectionType = Constant.MIN_POSITIVE_NORMAL,
        robust: bool | ConnectionType = False,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input,
            target=target,
            pos_weight=pos_weight,
            cutoff=cutoff,
            robust=robust,
            output=output,
        )


class Log(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        robust: bool = False,
        name: str | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
        *,
        cutoff: ConstantType | ToBeDetermined = TBD,
    ) -> None:
        self.robust = robust
        self.factory_args = {"robust": robust}

        if robust:
            super().__init__(
                formula_key="robust_log",
                name=name,
                output=IOKey(shape=[("Var", ...)], type=MyTensor[float]),
                input=IOKey(shape=[("Var", ...)], type=GenericTensorType),
                cutoff=IOKey(shape=[], type=GenericTensorType),
            )
            self.factory_inputs = {"input": input, "cutoff": cutoff}
        else:
            super().__init__(
                formula_key="log",
                name=name,
                output=IOKey(shape=[("Var", ...)], type=MyTensor[float]),
                input=IOKey(shape=[("Var", ...)], type=GenericTensorType),
            )
            self.factory_inputs = {"input": input}

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        *,
        cutoff: ConnectionType = Constant.MIN_POSITIVE_NORMAL,
    ) -> ExtendInfo:
        kwargs = {"input": input, "output": output}

        is_constant = isinstance(cutoff, Constant)
        if self.robust:
            kwargs["cutoff"] = cutoff
        elif not (is_constant and cutoff == Constant.MIN_POSITIVE_NORMAL):
            raise ValueError("Cutoff cannot be specified when robust mode is off")

        return super().__call__(**kwargs)


class StableReciprocal(PrimitiveModel):
    input: Connection
    cutoff: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
        cutoff: ConstantType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="stable_reciprocal",
            name=name,
            output=IOKey(shape=[("Var", ...)], type=MyTensor[float]),
            input=IOKey(shape=[("Var", ...)], type=GenericTensorType),
            cutoff=IOKey(shape=[], type=GenericTensorType),
        )
        self.factory_inputs = {"input": input, "cutoff": cutoff}

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        cutoff: ConnectionType = Constant.STABLE_RECIPROCAL_THRESHOLD,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, cutoff=cutoff, output=output)


class Sine(SingleInputOperation):
    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="sin",
            name=name,
            polymorphic_constraint=False,
            output=IOKey(shape=[("Var", ...)], type=MyTensor[float]),
        )
        self.factory_inputs = {"input": input}


class Cosine(SingleInputOperation):
    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="cos",
            name=name,
            polymorphic_constraint=False,
            output=IOKey(shape=[("Var", ...)], type=MyTensor[float]),
        )
        self.factory_inputs = {"input": input}


class Sign(SingleInputOperation):
    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="sign",
            name=name,
            polymorphic_constraint=False,
            output=IOKey(shape=[("Var", ...)], type=MyTensor[int]),
        )
        self.factory_inputs = {"input": input}


class Square(SingleInputOperation):
    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(formula_key="square", name=name)
        self.factory_inputs = {"input": input}


############################# Activation Types ##############################
class Activation(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        formula_key: str,
        polymorphic_constraint: bool = False,
        name: str | None = None,
        **kwargs: IOKey,
    ) -> None:
        # NOTE: Torch and JAX behave different for some activation functions.
        # For example JAX handles int type inputs for GELU or LeakyRelu while
        # Torch assumes only float inputs for these activations. Since JAX handles
        # more general case, default types are written taking this into account.
        default_kwargs: dict[str, IOKey] = dict(
            input=IOKey(shape=[("Var", ...)], type=GenericTensorType),
            output=IOKey(shape=[("Var", ...)], type=MyTensor[float]),
        )
        # Finalize kwargs.
        kwargs = default_kwargs | kwargs
        super().__init__(formula_key, name=name, **kwargs)

        if polymorphic_constraint:
            self._set_constraint(
                fn=general_tensor_type_constraint,
                keys=[PrimitiveModel.output_key, "input"],
            )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Relu(Activation):
    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="relu",
            name=name,
            polymorphic_constraint=True,
            output=IOKey(shape=[("Var", ...)], type=GenericTensorType),
            input=IOKey(shape=[("Var", ...)], type=GenericTensorType),
        )
        self.factory_inputs = {"input": input}


class Gelu(Activation):
    def __init__(
        self,
        name: str | None = None,
        approximate: bool = False,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="gelu",
            approximate=IOKey(value=approximate, type=(bool)),
            name=name,
        )
        self.factory_inputs = {"input": input}

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        approximate: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return BaseModel.__call__(
            self, input=input, approximate=approximate, output=output
        )


class Sigmoid(Activation):
    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(formula_key="sigmoid", name=name)
        self.factory_inputs = {"input": input}


class Softmax(Activation):
    def __init__(
        self,
        name: str | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
        axis: int | None | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(formula_key="softmax", name=name, axis=IOKey(type=int | None))
        self.factory_inputs = {"input": input, "axis": axis}

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        axis: ConnectionType = -1,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return BaseModel.__call__(self, input=input, axis=axis, output=output)


class Softplus(Activation):
    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(formula_key="softplus", name=name)
        self.factory_inputs = {"input": input}


class Tanh(Activation):
    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(formula_key="tanh", name=name)
        self.factory_inputs = {"input": input}


class LeakyRelu(Activation):
    input: Connection
    output: Connection
    slope: Connection

    def __init__(
        self,
        name: str | None = None,
        slope: TensorValueType | ToBeDetermined = TBD,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="leaky_relu",
            name=name,
            slope=IOKey(shape=[], type=MyTensor[float]),
        )
        self.factory_inputs = {"input": input, "slope": slope}

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        slope: ConnectionType = 0.01,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return PrimitiveModel.__call__(self, input=input, slope=slope, output=output)


class StopGradient(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="stop_gradient",
            name=name,
            output=IOKey(shape=[("Var", ...)], type=GenericTensorType),
            input=IOKey(shape=[("Var", ...)], type=GenericTensorType),
        )
        self.factory_inputs = {"input": input}

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class CartesianDifference(PrimitiveModel):
    left: Connection
    right: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="cartesian_diff",
            name=name,
            output=IOKey(shape=["N", "M", "dim"], type=GenericTensorType),
            left=IOKey(shape=["N", "dim"], type=GenericTensorType),
            right=IOKey(shape=["M", "dim"], type=GenericTensorType),
        )
        self.factory_inputs = {"left": left, "right": right}
        self._set_constraint(
            fn=general_tensor_type_constraint,
            keys=[PrimitiveModel.output_key, "left", "right"],
        )

    def __call__(  # type: ignore[override]
        self,
        left: ConnectionType = NOT_GIVEN,
        right: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(left=left, right=right, output=output)


class Concat(PrimitiveModel):
    output: Connection
    axis: Connection

    def __init__(
        self,
        n: int,
        axis: int | None | ToBeDetermined = 0,
        name: str | None = None,
        **kwargs: TensorValueType | ToBeDetermined,
    ) -> None:
        self.factory_args = {"n": n, "axis": axis}

        key_definitions: dict[str, IOKey] = {}
        key_definitions["output"] = IOKey(
            shape=[("Var_out", ...)], type=GenericTensorType
        )
        key_definitions |= {
            f"input{idx+1}": IOKey(
                shape=[(f"Var_{idx + 1}", ...)], type=GenericTensorType
            )
            for idx in range(n)
        }
        key_definitions["axis"] = IOKey(type=int | None, value=axis)

        super().__init__(formula_key="concat", name=name, **key_definitions)
        # self.factory_inputs = {key: value for key, value in kwargs.items()}
        self.factory_inputs = kwargs  # type: ignore

        input_keys = [key for key in self._input_keys if key != "axis"]
        self._set_constraint(
            fn=concat_constraints, keys=["output"] + ["axis"] + input_keys
        )
        self._set_constraint(
            fn=general_tensor_type_constraint,
            keys=[PrimitiveModel.output_key] + input_keys,
        )


class PrimitiveUnion(PrimitiveModel):
    output: Connection

    def __init__(
        self,
        n: int = 1,
        name: str | None = None,
        **kwargs: TensorValueType | ToBeDetermined,
    ) -> None:
        self.factory_args = {"n": n}
        input_definitions = {
            f"input{idx + 1}": IOKey(type=int | float | tuple[int | float, ...])
            for idx in range(n)
        }

        super().__init__(
            formula_key="union",
            name=name,
            output=IOKey(type=tuple[int | float, ...]),
            **input_definitions,
        )
        self.factory_inputs = kwargs  # type: ignore


class PermuteTensor(PrimitiveModel):
    input: Connection
    indices: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        indices: TensorValueType | ToBeDetermined = TBD,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="permute_tensor",
            name=name,
            output=IOKey(shape=["N", ("Var", ...)], type=GenericTensorType),
            input=IOKey(shape=["N", ("Var", ...)], type=GenericTensorType),
            indices=IOKey(shape=["N"], type=GenericTensorType),
        )
        self.factory_inputs = {"input": input, "indices": indices}

        self._set_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )
        self.indices.set_differentiable(False)

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        indices: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, indices=indices, output=output)


class PrimitiveConvolution1D(PrimitiveModel):
    input: Connection
    weight: Connection
    stride: Connection
    padding: Connection
    dilation: Connection
    output: Connection
    bias: Connection

    def __init__(
        self,
        use_bias: bool = True,
        name: str | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
        weight: TensorValueType | ToBeDetermined = TBD,
        stride: int | ToBeDetermined = TBD,
        padding: int | tuple[int, int] | ToBeDetermined = TBD,
        dilation: int | ToBeDetermined = TBD,
        *,
        bias: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        self.factory_args = {"use_bias": use_bias}
        formula_key = "conv1d_bias"
        kwargs: dict[str, IOKey] = {
            "output": IOKey(
                shape=["N", "out_channels", "d_out"], type=GenericTensorType
            ),
            "input": IOKey(shape=["N", "C_in", "d_in"], type=GenericTensorType),
            "weight": IOKey(
                shape=["out_channels", "C_in", "kernel_size"], type=GenericTensorType
            ),
            "bias": IOKey(shape=[1, "out_channels", 1], type=GenericTensorType),
            "stride": IOKey(type=int),
            "padding": IOKey(type=int | tuple[int, int]),
            "dilation": IOKey(type=int),
        }
        self.factory_inputs = {
            "input": input,
            "weight": weight,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "bias": bias,
        }

        if not use_bias:
            formula_key = "conv1d"
            kwargs.pop("bias")

        super().__init__(formula_key=formula_key, name=name, **kwargs)

        self._set_constraint(
            fn=conv_1d_constraints,
            keys=["output", "input", "stride", "padding", "dilation", "weight"],
        )

        constraint_keys = ["input", "weight"]
        if use_bias:
            constraint_keys.append("bias")

        self._set_constraint(
            fn=general_tensor_type_constraint,
            keys=[PrimitiveModel.output_key] + constraint_keys,
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        weight: ConnectionType = NOT_GIVEN,
        stride: ConnectionType = NOT_GIVEN,
        padding: ConnectionType = NOT_GIVEN,
        dilation: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        *,
        bias: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {
            "input": input,
            "weight": weight,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "output": output,
        }

        if "bias" not in self._input_keys and bias != NOT_GIVEN:
            raise ValueError(f"Model does not have 'bias' input. \
                             Got {bias} as bias argument!")
        elif "bias" in self._input_keys:
            kwargs |= {"bias": bias}

        return super().__call__(**kwargs)


class PrimitiveConvolution2D(PrimitiveModel):
    input: Connection
    weight: Connection
    stride: Connection
    padding: Connection
    dilation: Connection
    output: Connection
    bias: Connection

    def __init__(
        self,
        use_bias: bool = True,
        name: str | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
        weight: TensorValueType | ToBeDetermined = TBD,
        stride: int | tuple[int, int] | ToBeDetermined = TBD,
        padding: int
        | tuple[int, int]
        | tuple[tuple[int, int], tuple[int, int]]
        | ToBeDetermined = TBD,
        dilation: int | tuple[int, int] | ToBeDetermined = TBD,
        *,
        bias: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        self.factory_args = {"use_bias": use_bias}
        formula_key = "conv2d_bias"
        kwargs: dict[str, IOKey] = {
            "output": IOKey(
                shape=["N", "out_channels", "H_out", "W_out"], type=GenericTensorType
            ),
            "input": IOKey(shape=["N", "C_in", "H", "W"], type=GenericTensorType),
            "weight": IOKey(
                shape=["out_channels", "C_in", "kernel_size_0", "kernel_size_1"],
                type=GenericTensorType,
            ),
            "bias": IOKey(shape=[1, "out_channels", 1, 1], type=GenericTensorType),
            "stride": IOKey(type=int | tuple[int, int]),
            "padding": IOKey(
                type=int | tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]
            ),
            "dilation": IOKey(type=int | tuple[int, int]),
        }

        if not use_bias:
            formula_key = "conv2d"
            kwargs.pop("bias")

        super().__init__(formula_key, name=name, **kwargs)
        self.factory_inputs = {
            "input": input,
            "weight": weight,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "bias": bias,
        }

        self._set_constraint(
            fn=conv_2d_constraints,
            keys=["output", "input", "stride", "padding", "dilation", "weight"],
        )

        constraint_keys = ["input", "weight"]
        if use_bias:
            constraint_keys.append("bias")
        self._set_constraint(
            fn=general_tensor_type_constraint,
            keys=[PrimitiveModel.output_key] + constraint_keys,
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        weight: ConnectionType = NOT_GIVEN,
        stride: ConnectionType = NOT_GIVEN,
        padding: ConnectionType = NOT_GIVEN,
        dilation: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        *,
        bias: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {
            "input": input,
            "weight": weight,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "output": output,
        }

        if "bias" not in self._input_keys and bias != NOT_GIVEN:
            raise ValueError(
                f"Model does not have 'bias' input. Got {bias} as bias argument!"
            )
        elif "bias" in self._input_keys:
            kwargs |= {"bias": bias}
        return super().__call__(**kwargs)


class Flatten(PrimitiveModel):
    input: Connection
    start_dim: Connection
    end_dim: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        start_dim: int | ToBeDetermined = 0,
        end_dim: int | ToBeDetermined = -1,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        self.factory_args = {"start_dim": start_dim, "end_dim": end_dim}

        key_definitions: dict[str, IOKey] = {
            "output": IOKey(shape=[("C_out", ...)], type=GenericTensorType),
            "input": IOKey(shape=[("C_in", ...)], type=GenericTensorType),
            "start_dim": IOKey(type=int, value=start_dim),
            "end_dim": IOKey(type=int, value=end_dim),
        }
        super().__init__(formula_key="flatten", name=name, **key_definitions)
        # self.factory_inputs = {"input": input}
        self.factory_inputs = {
            "input": input,
            "start_dim": start_dim,
            "end_dim": end_dim,
        }

        self._set_constraint(
            fn=flatten_constrains,
            keys=[PrimitiveModel.output_key, "input", "start_dim", "end_dim"],
        )
        self._set_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        start_dim: ConnectionType = NOT_GIVEN,
        end_dim: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input, start_dim=start_dim, end_dim=end_dim, output=output
        )


class PrimitiveMaxPool1D(PrimitiveModel):
    input: Connection
    kernel_size: Connection
    stride: Connection
    padding: Connection
    dilation: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
        kernel_size: int | ToBeDetermined = TBD,
        stride: int | ToBeDetermined = TBD,
        padding: int | tuple[int, int] | ToBeDetermined = TBD,
        dilation: int | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="max_pool1d",
            name=name,
            output=IOKey(shape=["N", ("C_in", ...), "W_out"], type=GenericTensorType),
            input=IOKey(shape=["N", ("C_in", ...), "W"], type=GenericTensorType),
            kernel_size=IOKey(type=int),
            stride=IOKey(type=int),
            padding=IOKey(type=tuple[int, int]),
            dilation=IOKey(type=int),
        )
        self.factory_inputs = {
            "input": input,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
        }
        self._set_constraint(
            fn=sliding_window_1d_constraints,
            keys=["output", "input", "stride", "padding", "dilation", "kernel_size"],
        )
        # TODO: Torch does not accept any int type inputs but JAX implementation does.
        self._set_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        kernel_size: ConnectionType = NOT_GIVEN,
        stride: ConnectionType = NOT_GIVEN,
        padding: ConnectionType = NOT_GIVEN,
        dilation: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output=output,
        )


class PaddingConverter1D(PrimitiveModel):
    input: Connection
    kernel_size: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        kernel_size: int | ToBeDetermined = TBD,
        input: int | PaddingType | tuple[int, int] | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="padding_converter_1d",
            name=name,
            output=IOKey(type=tuple[int, int]),
            input=IOKey(type=int | PaddingType | tuple[int, int]),
            kernel_size=IOKey(type=int),
        )
        self.factory_inputs = {"input": input, "kernel_size": kernel_size}

        self._set_constraint(
            fn=padding_1d_constraint,
            keys=[PrimitiveModel.output_key, "input", "kernel_size"],
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        kernel_size: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, kernel_size=kernel_size, output=output)


class PaddingConverter2D(PrimitiveModel):
    input: Connection
    kernel_size: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        kernel_size: int | tuple[int, int] | ToBeDetermined = TBD,
        input: int
        | PaddingType
        | tuple[int, int]
        | tuple[tuple[int, int], tuple[int, int]]
        | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="padding_converter_2d",
            name=name,
            output=IOKey(
                type=tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]
            ),
            input=IOKey(
                type=int
                | PaddingType
                | tuple[int, int]
                | tuple[tuple[int, int], tuple[int, int]]
            ),
            kernel_size=IOKey(type=tuple[int, int]),
        )
        self.factory_inputs = {"input": input, "kernel_size": kernel_size}

        self._set_constraint(
            fn=padding_2d_constraint,
            keys=[PrimitiveModel.output_key, "input", "kernel_size"],
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        kernel_size: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, kernel_size=kernel_size, output=output)


class StrideConverter(PrimitiveModel):
    input: Connection
    kernel_size: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        kernel_size: int | tuple[int, int] | ToBeDetermined = TBD,
        input: int | PaddingType | tuple[int, int] | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="stride_converter",
            name=name,
            output=IOKey(type=int | tuple[int, int]),
            input=IOKey(type=int | PaddingType | tuple[int, int] | None),
            kernel_size=IOKey(type=int | tuple[int, int]),
        )
        self.factory_inputs = {"input": input, "kernel_size": kernel_size}
        self._set_constraint(
            fn=stride_constraint,
            keys=[PrimitiveModel.output_key, "input", "kernel_size"],
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        kernel_size: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, kernel_size=kernel_size, output=output)


class TupleConverter(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        input: int
        | PaddingType
        | tuple[int, int]
        | tuple[tuple[int, int], tuple[int, int]]
        | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="tuple_converter",
            name=name,
            output=IOKey(
                type=tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]
            ),
            input=IOKey(
                type=int
                | PaddingType
                | tuple[int, int]
                | tuple[tuple[int, int], tuple[int, int]]
                | None
            ),
        )
        self._set_constraint(
            fn=tuple_converter_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class PrimitiveMaxPool2D(PrimitiveModel):
    input: Connection
    kernel_size: Connection
    stride: Connection
    padding: Connection
    dilation: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
        kernel_size: int | tuple[int, int] | ToBeDetermined = TBD,
        stride: int | tuple[int, int] | ToBeDetermined = TBD,
        padding: int
        | tuple[int, int]
        | tuple[tuple[int, int], tuple[int, int]]
        | ToBeDetermined = TBD,
        dilation: int | tuple[int, int] | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="max_pool2d",
            name=name,
            output=IOKey(
                shape=["N", ("C_in", ...), "H_out", "W_out"], type=GenericTensorType
            ),
            input=IOKey(shape=["N", ("C_in", ...), "H", "W"], type=GenericTensorType),
            kernel_size=IOKey(type=tuple[int, int]),
            stride=IOKey(type=tuple[int, int]),
            padding=IOKey(
                type=tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]
            ),
            dilation=IOKey(type=tuple[int, int]),
        )
        self.factory_inputs = {
            "input": input,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
        }

        self._set_constraint(
            fn=sliding_window_2d_constraints,
            keys=["output", "input", "stride", "padding", "dilation", "kernel_size"],
        )
        self._set_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        kernel_size: ConnectionType = NOT_GIVEN,
        stride: ConnectionType = NOT_GIVEN,
        padding: ConnectionType = NOT_GIVEN,
        dilation: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output=output,
        )


class NormModifier(PrimitiveModel):
    """A helper model that modifies norm input. It is used for mapping
    norm values from (`-inf`, `inf`) to the interval (`1.0`, `5.0`) using a
    periodic triangular function with period 8 as shown on the figure below.
    This helper model guarantees norm values to be in an acceptable and
    meaningful range.

    ```
    5 _ _ _ _ _ _ _ _ _ _
         \\      /\\      /
          \\    /  \\    /
           \\  /    \\  /
    1 _ _ _ \\/_ _ _ \\/_ _
        |   |   |   |
        |   |   |   |
        -3   1   5   9
    ```
    """

    input: Connection
    output: Connection

    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="norm_modifier",
            name=name,
            output=IOKey(shape=[], type=GenericTensorType),
            input=IOKey(shape=[], type=GenericTensorType),
        )
        self.factory_inputs = {"input": input}

        self._set_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class DistanceMatrix(PrimitiveModel):
    left: Connection
    right: Connection
    norm: Connection
    output: Connection

    # TODO: torch.cdist handles batches of matrices, for now we don't.
    def __init__(
        self,
        name: str | None = None,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="distance_matrix",
            name=name,
            output=IOKey(shape=["N", "M"], type=GenericTensorType),
            left=IOKey(shape=["N", "d"], type=GenericTensorType),
            right=IOKey(shape=["M", "d"], type=GenericTensorType),
            norm=IOKey(shape=[], type=GenericTensorType),
        )
        self.factory_inputs = {"left": left, "right": right}

        self._set_constraint(
            fn=general_tensor_type_constraint,
            keys=[PrimitiveModel.output_key, "left", "right", "norm"],
        )

    def __call__(  # type: ignore[override]
        self,
        left: ConnectionType = NOT_GIVEN,
        right: ConnectionType = NOT_GIVEN,
        norm: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(left=left, right=right, norm=norm, output=output)


class PolynomialFeatures(PrimitiveModel):
    input: Connection
    degree: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        degree: int | ToBeDetermined = TBD,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="polynomial_features",
            name=name,
            output=IOKey(shape=["N", "d_out"], type=GenericTensorType),
            input=IOKey(shape=["N", "d_in"], type=GenericTensorType),
            degree=IOKey(type=int, value=degree),
        )
        self.factory_inputs = {"input": input, "degree": degree}

        self._set_constraint(
            fn=polynomial_features_constraints, keys=["output", "input", "degree"]
        )
        self._set_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        degree: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, degree=degree, output=output)


class TsnePJoint(PrimitiveModel):
    squared_distances: Connection
    target_perplexity: Connection
    threshold: Connection
    output: Connection

    disposable = True

    def __init__(
        self,
        name: str | None = None,
        squared_distances: TensorValueType | ToBeDetermined = TBD,
        target_perplexity: float | ToBeDetermined = TBD,
        threshold: ConstantType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="tsne_p_joint",
            name=name,
            output=IOKey(shape=["N", "M"], type=MyTensor[float]),
            squared_distances=IOKey(shape=["N", "M"], type=GenericTensorType),
            target_perplexity=IOKey(shape=[], type=MyTensor[float]),
            threshold=IOKey(shape=[], type=GenericTensorType),
        )
        self.factory_inputs = {
            "squared_distances": squared_distances,
            "target_perplexity": target_perplexity,
            "threshold": threshold,
        }

    def __call__(  # type: ignore[override]
        self,
        squared_distances: ConnectionType = NOT_GIVEN,
        target_perplexity: float | ConnectionType = NOT_GIVEN,
        threshold: ConnectionType = NOT_GIVEN,
        output: ConstantType | ConnectionType = Constant.EPSILON,
    ) -> ExtendInfo:
        return super().__call__(
            squared_distances=squared_distances,
            target_perplexity=target_perplexity,
            threshold=threshold,
            output=output,
        )


class EyeComplement(PrimitiveModel):
    N: Connection
    M: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        N: int | ToBeDetermined = TBD,
        M: int | ToBeDetermined | None = None,
    ) -> None:
        super().__init__(
            formula_key="ones_with_zero_diag",
            name=name,
            output=IOKey(shape=["N", "M"], type=MyTensor[float]),
            N=IOKey(type=int, value=N),
            M=IOKey(type=int | None, value=M),
        )
        self.factory_inputs = {"N": N, "M": M}
        self._set_constraint(fn=eye_constraints, keys=["output", "N", "M"])

    def __call__(  # type: ignore[override]
        self,
        N: ConnectionType = NOT_GIVEN,
        M: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(N=N, M=M, output=output)


class Eye(PrimitiveModel):
    N: Connection
    M: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        N: int | ToBeDetermined = TBD,
        M: int | ToBeDetermined | None = None,
    ) -> None:
        super().__init__(
            formula_key="eye",
            name=name,
            output=IOKey(shape=["N", "M"], type=MyTensor[float]),
            N=IOKey(type=int, value=N),
            M=IOKey(type=int | None, value=M),
        )
        self.factory_inputs = {"N": N, "M": M}

        self._set_constraint(fn=eye_constraints, keys=["output", "N", "M"])

    def __call__(  # type: ignore[override]
        self,
        N: ConnectionType = NOT_GIVEN,
        M: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(N=N, M=M, output=output)


class Cholesky(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="cholesky",
            name=name,
            output=IOKey(shape=["N", "N"], type=MyTensor[float]),
            input=IOKey(shape=["N", "N"], type=GenericTensorType),
        )
        self.factory_inputs = {"input": input}

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class GPRAlpha(PrimitiveModel):
    label_mu_diff: Connection
    L: Connection
    K_term: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        label_mu_diff: TensorValueType | ToBeDetermined = TBD,
        L: TensorValueType | ToBeDetermined = TBD,
        K_term: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="gpr_alpha",
            name=name,
            output=IOKey(shape=["N", 1], type=MyTensor[float]),
            label_mu_diff=IOKey(shape=["N", 1], type=GenericTensorType),
            L=IOKey(shape=["N", "N"], type=GenericTensorType),
            K_term=IOKey(shape=["N", "N"], type=GenericTensorType),
        )
        self.factory_inputs = {"label_mu_diff": label_mu_diff, "L": L, "K_term": K_term}

    def __call__(  # type: ignore[override]
        self,
        label_mu_diff: ConnectionType = NOT_GIVEN,
        L: ConnectionType = NOT_GIVEN,
        K_term: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            label_mu_diff=label_mu_diff, L=L, K_term=K_term, output=output
        )


class GPRVOuter(PrimitiveModel):
    K: Connection
    K_term: Connection
    L: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        K: TensorValueType | ToBeDetermined = TBD,
        K_term: TensorValueType | ToBeDetermined = TBD,
        L: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="gpr_v_outer",
            name=name,
            output=IOKey(shape=["N", "N"], type=MyTensor[float]),
            K=IOKey(shape=["N", "N"], type=GenericTensorType),
            K_term=IOKey(shape=["N", "N"], type=GenericTensorType),
            L=IOKey(shape=["N", "N"], type=GenericTensorType),
        )
        self.factory_inputs = {"K": K, "K_term": K_term, "L": L}

    def __call__(  # type: ignore[override]
        self,
        K: ConnectionType = NOT_GIVEN,
        K_term: ConnectionType = NOT_GIVEN,
        L: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(K=K, K_term=K_term, L=L, output=output)


class TransposedDiagonal(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="transposed_diag",
            name=name,
            output=IOKey(shape=["N", 1], type=GenericTensorType),
            input=IOKey(shape=["N", "N"], type=GenericTensorType),
        )
        self.factory_inputs = {"input": input}

        self._set_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Arange(PrimitiveModel):
    start: Connection
    stop: Connection
    step: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        start: int | float | ToBeDetermined = 0,
        stop: int | float | ToBeDetermined = TBD,
        step: int | float | ToBeDetermined = 1,
    ) -> None:
        self.factory_inputs = {"start": start, "stop": stop, "step": step}

        all_defined = False
        if (
            not isinstance(start, ToBeDetermined)
            and not isinstance(stop, ToBeDetermined)
            and not isinstance(step, ToBeDetermined)
        ):
            all_defined = True
            val = (start - stop) / step
            # If val has decimal part take absolute of integer
            # part of it and add 1.
            # Else no decimal part, simply take absolute of val.
            val = abs(val) if int(val) == val else abs(int(val)) + 1
            val = int(val)
            output_shp: list[int | str] = [] if val == 0 else [val]
        else:
            output_shp = ["N"]

        super().__init__(
            formula_key="arange",
            name=name,
            output=IOKey(shape=output_shp, type=GenericTensorType),
            start=IOKey(type=int | float, value=start),
            stop=IOKey(type=int | float, value=stop),
            step=IOKey(type=int | float, value=step),
        )
        self.set_canonical_input("stop")
        self.factory_inputs = {"start": start, "stop": stop, "step": step}

        if not all_defined:
            self._set_constraint(
                fn=arange_constraints, keys=["output", "start", "stop", "step"]
            )
        self._set_constraint(
            fn=general_tensor_type_constraint,
            keys=[PrimitiveModel.output_key, "start", "stop", "step"],
        )

    def __call__(  # type: ignore[override]
        self,
        start: ConnectionType = NOT_GIVEN,
        stop: ConnectionType = NOT_GIVEN,
        step: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(start=start, stop=stop, step=step, output=output)


class BroadcastTo(PrimitiveModel):
    input: Connection
    shape: Connection
    output: Connection

    def __init__(
        self,
        shape: tuple[int, ...] | ToBeDetermined = TBD,
        name: str | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="broadcast_to",
            name=name,
            output=IOKey(shape=[("output", ...)], type=GenericTensorType),
            input=IOKey(shape=[("input", ...)], type=GenericTensorType),
            shape=IOKey(type=tuple[int, ...], value=shape),
        )
        self.factory_inputs = {"input": input, "shape": shape}

        self.set_constraint(
            fn=broadcast_to_constraints, keys=["output", "shape", "input"]
        )
        self._set_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        shape: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, shape=shape, output=output)


class Eigvalsh(PrimitiveModel):
    K_term: Connection
    L: Connection
    threshold: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        K_term: TensorValueType | ToBeDetermined = TBD,
        L: TensorValueType | ToBeDetermined = TBD,
        threshold: ConstantType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="eigvalsh",
            name=name,
            output=IOKey(shape=["N", 1], type=MyTensor[float]),
            K_term=IOKey(shape=["N", "N"], type=GenericTensorType),
            L=IOKey(shape=["N", "N"], type=GenericTensorType),
            threshold=IOKey(shape=[], type=GenericTensorType),
        )
        self.factory_inputs = {"K_term": K_term, "L": L, "threshold": threshold}

    def __call__(  # type: ignore[override]
        self,
        K_term: ConnectionType = NOT_GIVEN,
        L: ConnectionType = NOT_GIVEN,
        threshold: ConstantType | ConnectionType = Constant.EPSILON,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(K_term=K_term, L=L, threshold=threshold, output=output)


class Squeeze(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="squeeze",
            name=name,
            output=IOKey(shape=[("Var_out", ...)], type=GenericTensorType),
            input=IOKey(shape=[("Var", ...)], type=GenericTensorType),
        )
        self.factory_inputs = {"input": input}

        self._set_constraint(fn=squeeze_constraints, keys=["output", "input"])
        self._set_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class AUCCore(PrimitiveModel):
    input: Connection
    label: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
        label: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="auc_core",
            name=name,
            output=IOKey(shape=[2, "M"], type=MyTensor[float]),
            input=IOKey(shape=["N"], type=GenericTensorType),
            label=IOKey(shape=["N"], type=GenericTensorType),
        )
        self.factory_inputs = {"input": input, "label": label}

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        label: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, label=label, output=output)


class Embedding(PrimitiveModel):
    input: Connection
    weight: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        num_embeddings: int | None = None,
        dim: int | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
        weight: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        out_dim: int | str = "dim" if dim is None else dim

        super().__init__(
            formula_key="primitive_embedding",
            name=name,
            output=IOKey(shape=[("N1", ...), "d1", out_dim], type=GenericTensorType),
            input=IOKey(shape=[("N1", ...), "d1"], type=MyTensor[int]),
            weight=IOKey(shape=[num_embeddings, out_dim], type=GenericTensorType),
        )
        self.factory_inputs = {"input": input, "weight": weight}

        self._set_constraint(
            fn=general_tensor_type_constraint,
            keys=[PrimitiveModel.output_key, "weight"],
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        weight: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, weight=weight, output=output)


class ScaledDotProduct(PrimitiveModel):
    query: Connection
    key: Connection
    value: Connection
    attn_mask: Connection
    dropout_p: Connection
    is_causal: Connection
    scale: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        is_causal: bool | ToBeDetermined = True,
        scale: None | int | float | ToBeDetermined = None,
        dropout_p: float | ToBeDetermined = 0.0,
        use_attn_mask: bool = False,
        query: TensorValueType | ToBeDetermined = TBD,
        key: TensorValueType | ToBeDetermined = TBD,
        value: TensorValueType | ToBeDetermined = TBD,
        attn_mask: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        # TODO: Reconsider how to get attn_mask, could it be A?
        assert (
            not isinstance(is_causal, bool) or not is_causal or not use_attn_mask
        ), "Causal attention is not support attn_mask!"
        assert isinstance(use_attn_mask, bool), "use_attn_mask must be a boolean value!"
        self.use_attn_mask = use_attn_mask

        formula_key = "scaled_dot_product_attention"
        kwargs: dict[str, IOKey] = {
            "output": IOKey(shape=[("Var", ...), "L", "O"], type=MyTensor[float]),
            "query": IOKey(shape=[("Var", ...), "L", "E"], type=GenericTensorType),
            "key": IOKey(shape=[("Var", ...), "S", "E"], type=GenericTensorType),
            "value": IOKey(shape=[("Var", ...), "S", "O"], type=GenericTensorType),
            "dropout_p": IOKey(type=float, value=dropout_p),
            "attn_mask": IOKey(type=NoneType, value=None),
            "is_causal": IOKey(type=bool, value=is_causal),
            "scale": IOKey(type=NoneType | int | float, value=scale),
        }

        if use_attn_mask:
            kwargs["attn_mask"] = IOKey(
                shape=["L", "S"], type=GenericTensorType, value=TBD
            )

        super().__init__(formula_key=formula_key, name=name, **kwargs)
        self.factory_inputs = {
            "query": query,
            "key": key,
            "value": value,
            "dropout_p": dropout_p,
            "is_causal": is_causal,
            "scale": scale,
            "attn_mask": attn_mask,
        }

    def __call__(  # type: ignore[override]
        self,
        query: ConnectionType = NOT_GIVEN,
        key: ConnectionType = NOT_GIVEN,
        value: ConnectionType = NOT_GIVEN,
        dropout_p: ConnectionType = NOT_GIVEN,
        is_causal: ConnectionType = NOT_GIVEN,
        scale: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        *,
        attn_mask: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        if (
            not self.use_attn_mask
            and attn_mask is not NOT_GIVEN
            and not isinstance(attn_mask, str)
            and isinstance(attn_mask, IOKey)
            and attn_mask._value is not None  # TODO: Here will be updated!
        ):
            raise KeyError(
                "Model does not have 'attn_mask' input. Got attn_mask argument!"
            )

        return super().__call__(
            query=query,
            key=key,
            value=value,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            output=output,
            attn_mask=attn_mask,
        )


class PositionalEncoding(PrimitiveModel):
    input: Connection
    hidden_dim: Connection
    max_len: Connection
    output: Connection

    # TODO: Try to move to Logical composite models.
    def __init__(
        self,
        hidden_dim: int | ToBeDetermined,
        name: str | None = None,
        max_len: int | ToBeDetermined = 5000,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        self.factory_args = {"hidden_dim": hidden_dim, "max_len": max_len}

        super().__init__(
            formula_key="positional_encoding",
            name=name,
            output=IOKey(shape=[("N1", ...)], type=GenericTensorType),
            input=IOKey(shape=[("N1", ...)], type=GenericTensorType),
            hidden_dim=IOKey(type=int, value=hidden_dim),
            max_len=IOKey(type=int, value=max_len),
        )
        self.factory_inputs = {
            "input": input,
            "hidden_dim": hidden_dim,
            "max_len": max_len,
        }

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        hidden_dim: ConnectionType = NOT_GIVEN,
        max_len: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input, hidden_dim=hidden_dim, max_len=max_len, output=output
        )


class SwapAxes(PrimitiveModel):
    input: Connection
    axis1: Connection
    axis2: Connection
    output: Connection

    def __init__(
        self,
        axis1: int | ToBeDetermined,
        axis2: int | ToBeDetermined,
        name: str | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        self.factory_args = {"axis1": axis1, "axis2": axis2}

        super().__init__(
            formula_key="swapaxes",
            name=name,
            output=IOKey(shape=[("Var_out", ...)], type=GenericTensorType),
            input=IOKey(shape=[("Var_in", ...)], type=GenericTensorType),
            axis1=IOKey(type=int, value=axis1),
            axis2=IOKey(type=int, value=axis2),
        )
        self.factory_inputs = {"input": input, "axis1": axis1, "axis2": axis2}

        self._set_constraint(
            fn=swap_axes_constraints, keys=["output", "input", "axis1", "axis2"]
        )
        self._set_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        axis1: ConnectionType = NOT_GIVEN,
        axis2: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, axis1=axis1, axis2=axis2, output=output)


class Where(PrimitiveModel):
    cond: Connection
    input1: Connection
    input2: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        cond: TensorValueType | ToBeDetermined = TBD,
        input1: TensorValueType | ToBeDetermined = TBD,
        input2: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="where",
            name=name,
            output=IOKey(shape=[("Var_out", ...)], type=GenericTensorType),
            cond=IOKey(shape=[("Var3", ...)], type=MyTensor[bool], value=TBD),
            input1=IOKey(shape=[("Var1", ...)], type=GenericTensorType),
            input2=IOKey(shape=[("Var2", ...)], type=GenericTensorType),
        )
        self.factory_inputs = {"cond": cond, "input1": input1, "input2": input2}

        self._set_constraint(
            fn=where_constrains, keys=["output", "cond", "input1", "input2"]
        )
        self._set_constraint(
            fn=general_tensor_type_constraint,
            keys=[PrimitiveModel.output_key, "input1", "input2"],
        )

    def __call__(  # type: ignore[override]
        self,
        cond: ConnectionType = NOT_GIVEN,
        input1: ConnectionType = NOT_GIVEN,
        input2: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(cond=cond, input1=input1, input2=input2, output=output)


class IsNan(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="isnan",
            name=name,
            output=IOKey(shape=[("Var", ...)], type=MyTensor[bool]),
            input=IOKey(shape=[("Var", ...)], type=GenericTensorType),
        )
        self.factory_inputs = {"input": input}

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Unique(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="unique",
            name=name,
            input=IOKey(shape=[("Var1", ...)], type=GenericTensorType),
            output=IOKey(shape=[("Var2", ...)], type=GenericTensorType),
        )
        self.factory_inputs = {"input": input}

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Trapezoid(PrimitiveModel):
    y: Connection
    x: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        x: TensorValueType | ToBeDetermined = TBD,
        y: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="trapezoid",
            name=name,
            output=IOKey(shape=[], type=GenericTensorType),
            y=IOKey(shape=[("Var", ...)], type=GenericTensorType),
            x=IOKey(shape=[("Var", ...)], type=GenericTensorType),
        )
        self.factory_inputs = {"y": y, "x": x}

    def __call__(  # type: ignore[override]
        self,
        y: ConnectionType = NOT_GIVEN,
        x: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(y=y, x=x, output=output)


class NanToNum(PrimitiveModel):
    input: Connection
    nan: Connection
    posinf: Connection
    neginf: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        nan: float | ToBeDetermined = 0.0,
        posinf: float | None | ToBeDetermined = None,
        neginf: float | None | ToBeDetermined = None,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="nan_to_num",
            name=name,
            output=IOKey(shape=[("Var", ...)], type=GenericTensorType),
            input=IOKey(shape=[("Var", ...)], type=GenericTensorType),
            nan=IOKey(type=float, value=nan),
            posinf=IOKey(type=float | None, value=posinf),
            neginf=IOKey(type=float | None, value=neginf),
        )
        self.factory_inputs = {
            "input": input,
            "nan": nan,
            "posinf": posinf,
            "neginf": neginf,
        }
        # TODO: Any constraints required?

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        nan: ConnectionType = NOT_GIVEN,
        posinf: ConnectionType = NOT_GIVEN,
        neginf: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input, nan=nan, posinf=posinf, neginf=neginf, output=output
        )


class Pad(PrimitiveModel):
    input: Connection
    pad_width: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        pad_width: list[tuple[int, int]] | ToBeDetermined = TBD,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="pad",
            name=name,
            output=IOKey(shape=[("Var2", ...)], type=GenericTensorType),
            input=IOKey(shape=[("Var1", ...)], type=GenericTensorType),
            pad_width=IOKey(type=tuple[tuple[int, int], ...], value=pad_width),
        )
        self.factory_inputs = {"input": input, "pad_width": pad_width}

        self._set_constraint(
            fn=pad_constraints, keys=[PrimitiveModel.output_key, "input", "pad_width"]
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        pad_width: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, pad_width=pad_width, output=output)


class ZerosLike(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="zeros_like",
            name=name,
            output=IOKey(shape=[("Var", ...)], type=GenericTensorType),
            input=IOKey(shape=[("Var", ...)], type=GenericTensorType),
        )
        self.factory_inputs = {"input": input}

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)
