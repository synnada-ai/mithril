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

from ..core import Constant, Dtype
from ..framework.common import (
    NOT_GIVEN,
    TBD,
    BaseKey,
    Tensor,
    ToBeDetermined,
)
from ..framework.constraints import (
    arange_constraints,
    bcast,
    bcast_error_check,
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
    randn_constraints,
    sliding_window_1d_constraints,
    sliding_window_2d_constraints,
    squeeze_constraints,
    stride_constraint,
    swap_axes_constraints,
    tuple_converter_constraint,
    where_constrains,
)
from ..framework.logical import Model
from ..framework.logical.model import Connection, ConnectionType, ExtendInfo
from ..framework.logical.primitive import PrimitiveModel
from ..framework.logical.user_essential_primitives import UserPrimitiveModel
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
    "Randn",
    "UserPrimitiveModel",
]

# Define types used to define keys:
ConstantType = float | int | Constant


class CustomPrimitiveModel(UserPrimitiveModel):
    def __init__(
        self, formula_key: str, name: str | None = None, **kwargs: BaseKey
    ) -> None:
        self.factory_args = {"formula_key": formula_key} | kwargs
        super().__init__(formula_key=formula_key, name=name, **kwargs)


########################## Supervised Loss Types ##########################
class SupervisedLoss(UserPrimitiveModel):
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
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        target: Tensor[int | float | bool] | ToBeDetermined = TBD,
        **kwargs: BaseKey,
    ) -> None:
        default_kwargs: dict[str, BaseKey] = {
            "output": BaseKey(shape=[("Var_1", ...)], type=Tensor),
            "input": BaseKey(shape=[("Var_2", ...)], type=Tensor, value=input),
            "target": BaseKey(shape=[("Var_3", ...)], type=Tensor, value=target),
        }
        # Finalize kwargs.
        kwargs = default_kwargs | kwargs
        super().__init__(formula_key=formula_key, name=name, **kwargs)

        # Set constraints.
        bcast_constraint = self._add_constraint(
            fn=bcast, keys=[PrimitiveModel.output_key, "input", "target"]
        )

        self._add_constraint(
            fn=bcast_error_check,
            keys=[PrimitiveModel.output_key, "input", "target"],
            dependencies={bcast_constraint},
        )

        if polymorphic_constraint:
            self._add_constraint(
                fn=general_tensor_type_constraint,
                keys=[PrimitiveModel.output_key, "input", "target"],
            )

        next(iter(self.dag)).safe_shapes = {
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
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        target: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="squared_error", name=name, input=input, target=target
        )


class AbsoluteError(SupervisedLoss):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        target: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="absolute_error", name=name, input=input, target=target
        )


class HingeLoss(SupervisedLoss):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        target: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            polymorphic_constraint=False,
            formula_key="hinge_loss",
            name=name,
            output=BaseKey(shape=["N", ("Var", ...)], type=Tensor[float]),
            input=input,
            target=target,
        )


class QuadHingeLoss(SupervisedLoss):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        target: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            polymorphic_constraint=False,
            formula_key="quad_hinge_loss",
            name=name,
            output=BaseKey(shape=["N", ("Var", ...)], type=Tensor[float]),
            input=input,
            target=target,
        )


class QuantileLoss(UserPrimitiveModel):
    """
    Takes N-dimensional input and target and produces N-dimensional output.
    """

    input: Connection
    target: Connection
    quantile: Connection
    output: Connection

    def __init__(
        self,
        quantile: int | float | ToBeDetermined = TBD,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        target: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="quantile_loss",
            name=name,
            output=BaseKey(shape=[("Var_1", ...)], type=Tensor),
            input=BaseKey(shape=[("Var_2", ...)], type=Tensor, value=input),
            target=BaseKey(shape=[("Var_3", ...)], type=Tensor, value=target),
            quantile=BaseKey(shape=[], type=Tensor[int | float], value=quantile),
        )

        bcast_constraint = self._add_constraint(
            fn=bcast, keys=[PrimitiveModel.output_key, "input", "target"]
        )

        self._add_constraint(
            fn=bcast_error_check,
            keys=[PrimitiveModel.output_key, "input", "target"],
            dependencies={bcast_constraint},
        )

        self._add_constraint(
            fn=general_tensor_type_constraint,
            keys=[PrimitiveModel.output_key, "input", "target", "quantile"],
        )

        next(iter(self.dag)).safe_shapes = {
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


class CrossEntropy(UserPrimitiveModel):
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
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        target: Tensor[int | float | bool] | ToBeDetermined = TBD,
        robust: bool | ToBeDetermined = TBD,
        cutoff: ConstantType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
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

        kwargs: dict[str, BaseKey] = {
            "output": BaseKey(shape=["N", ("Var", ...)], type=Tensor[float]),
            "input": BaseKey(shape=["N", "C", ("Var", ...)], type=Tensor, value=input),
            "target": BaseKey(
                shape=["N", ("VarTarget", ...)], type=Tensor, value=target
            ),
            "weights": BaseKey(type=weights_type, value=final_weights),
            "categorical": BaseKey(type=bool),
            "cutoff": BaseKey(shape=[], type=Tensor, value=cutoff),
            "robust": BaseKey(type=bool, value=robust),
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

        self._add_constraint(
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
        if next(iter(self.dag)).formula_key == "cross_entropy_with_log_probs":
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
            if isinstance(cutoff, Constant) and cutoff == Constant.MIN_POSITIVE_NORMAL:
                # NOTE: Since we can not provide Tensor objects as default
                # arguments, we need to convert default value.
                cutoff = Tensor(cutoff)
            kwargs |= {"cutoff": cutoff, "robust": robust}

        return super().__call__(**kwargs)


class KLDivergence(UserPrimitiveModel):
    """
    Takes N-dimensional input and target and produces N-dimensional output.
    """

    input: Connection
    target: Connection
    cutoff: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        target: Tensor[int | float | bool] | ToBeDetermined = TBD,
        cutoff: ConstantType | ToBeDetermined = TBD,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="kl_divergence",
            name=name,
            output=BaseKey(shape=[("Var_1", ...)], type=Tensor[float]),
            input=BaseKey(shape=[("Var_2", ...)], type=Tensor, value=input),
            target=BaseKey(shape=[("Var_3", ...)], type=Tensor, value=target),
            cutoff=BaseKey(shape=[], type=Tensor, value=cutoff),
        )

        next(iter(self.dag)).safe_shapes = {
            "output": ["N", ("Var", ...)],
            "input": ["N", ("Var", ...)],
            "target": ["N", ("Var", ...)],
        }
        bcast_constraint = self._add_constraint(
            fn=bcast, keys=[PrimitiveModel.output_key, "input", "target"]
        )

        self._add_constraint(
            fn=bcast_error_check,
            keys=[PrimitiveModel.output_key, "input", "target"],
            dependencies={bcast_constraint},
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        target: ConnectionType = NOT_GIVEN,
        cutoff: ConnectionType = Constant.MIN_POSITIVE_NORMAL,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        if isinstance(cutoff, Constant) and cutoff == Constant.MIN_POSITIVE_NORMAL:
            # NOTE: Since we can not provide Tensor objects as default
            # arguments, we need to convert default value.
            cutoff = Tensor(cutoff)
        return super().__call__(
            input=input, target=target, cutoff=cutoff, output=output
        )


class BinaryCrossEntropy(UserPrimitiveModel):
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
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        target: Tensor[int | float | bool] | ToBeDetermined = TBD,
        cutoff: Tensor[int | float | bool] | ToBeDetermined = TBD,
        robust: bool | ToBeDetermined = TBD,
        *,
        name: str | None = None,
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
        kwargs: dict[str, BaseKey] = {
            "output": BaseKey(shape=[("Var_out", ...)], type=Tensor[float]),
            "input": BaseKey(shape=[("Var_out", ...)], type=Tensor, value=input),
            "target": BaseKey(
                shape=[("Var_out", ...)],
                type=Tensor[int | float],
                value=target,
            ),
            "pos_weight": BaseKey(type=pos_weight_type, value=pos_weight),
            "cutoff": BaseKey(value=cutoff),
            "robust": BaseKey(type=bool, value=robust),
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

        bcast_constraint = self._add_constraint(
            fn=bcast, keys=[PrimitiveModel.output_key, "input", "target"]
        )

        self._add_constraint(
            fn=bcast_error_check,
            keys=[PrimitiveModel.output_key, "input", "target"],
            dependencies={bcast_constraint},
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


class Log(UserPrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        robust: bool = False,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        cutoff: Tensor[int | float | bool] | ToBeDetermined = TBD,
        name: str | None = None,
    ) -> None:
        self.robust = robust
        self.factory_args = {"robust": robust}

        if robust:
            super().__init__(
                formula_key="robust_log",
                name=name,
                output=BaseKey(shape=[("Var", ...)], type=Tensor[float]),
                input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
                cutoff=BaseKey(shape=[], type=Tensor, value=cutoff),
            )
        else:
            super().__init__(
                formula_key="log",
                name=name,
                output=BaseKey(shape=[("Var", ...)], type=Tensor[float]),
                input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
            )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        *,
        cutoff: ConnectionType = Constant.MIN_POSITIVE_NORMAL,
    ) -> ExtendInfo:
        kwargs = {"input": input, "output": output}

        default = (
            isinstance(cutoff, Constant) and cutoff == Constant.MIN_POSITIVE_NORMAL
        )
        if self.robust:
            if default:
                # NOTE: Since we can not provide Tensor objects as default
                # arguments, we need to convert default value.
                cutoff = Tensor(cutoff)  # type: ignore
            kwargs["cutoff"] = cutoff
        elif not default:
            raise ValueError("Cutoff cannot be specified when robust mode is off")

        return super().__call__(**kwargs)


class StableReciprocal(UserPrimitiveModel):
    input: Connection
    cutoff: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        cutoff: Tensor[int | float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="stable_reciprocal",
            name=name,
            output=BaseKey(shape=[("Var", ...)], type=Tensor[float]),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
            cutoff=BaseKey(shape=[], value=cutoff),
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        cutoff: ConnectionType = Constant.STABLE_RECIPROCAL_THRESHOLD,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        if (
            isinstance(cutoff, Constant)
            and cutoff == Constant.STABLE_RECIPROCAL_THRESHOLD
        ):
            # NOTE: Since we can not provide Tensor objects as default
            # arguments, we need to convert default value.
            cutoff = Tensor(cutoff)
        return super().__call__(input=input, cutoff=cutoff, output=output)


class Sign(UserPrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="sign",
            name=name,
            output=BaseKey(shape=[("Var", ...)], type=Tensor[float]),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Square(UserPrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="square",
            name=name,
            output=BaseKey(shape=[("Var", ...)], type=Tensor[float]),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


############################# Activation Types ##############################
class Activation(UserPrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        formula_key: str,
        polymorphic_constraint: bool = False,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
        **kwargs: BaseKey,
    ) -> None:
        # NOTE: Torch and JAX behave different for some activation functions.
        # For example JAX handles int type inputs for GELU or LeakyRelu while
        # Torch assumes only float inputs for these activations. Since JAX handles
        # more general case, default types are written taking this into account.
        default_kwargs: dict[str, BaseKey] = dict(
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
            output=BaseKey(shape=[("Var", ...)], type=Tensor[float]),
        )
        # Finalize kwargs.
        kwargs = default_kwargs | kwargs
        super().__init__(name=name, formula_key=formula_key, **kwargs)

        if polymorphic_constraint:
            self._add_constraint(
                fn=general_tensor_type_constraint,
                keys=[PrimitiveModel.output_key, "input"],
            )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Relu(Activation):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="relu",
            name=name,
            polymorphic_constraint=True,
            input=input,
            output=BaseKey(shape=[("Var", ...)], type=Tensor),
        )


class Gelu(Activation):
    def __init__(
        self,
        approximate: bool = False,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="gelu",
            approximate=BaseKey(value=approximate, type=bool),
            name=name,
            input=input,
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        approximate: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return Model.__call__(self, input=input, approximate=approximate, output=output)


class Sigmoid(Activation):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="sigmoid", name=name, input=input)


class Softmax(Activation):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        axis: int | None | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        axis_key = BaseKey(type=int | None, value=axis)
        super().__init__(formula_key="softmax", name=name, axis=axis_key, input=input)

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        axis: ConnectionType = -1,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return Model.__call__(self, input=input, axis=axis, output=output)


class Softplus(Activation):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="softplus", name=name, input=input)


class Tanh(Activation):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="tanh", name=name, input=input)


class LeakyRelu(Activation):
    input: Connection
    output: Connection
    slope: Connection

    def __init__(
        self,
        slope: Tensor[int | float | bool] | int | float | ToBeDetermined = TBD,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="leaky_relu",
            name=name,
            input=input,
            slope=BaseKey(value=slope),
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        slope: ConnectionType = 0.01,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return Model.__call__(self, input=input, slope=slope, output=output)


class StopGradient(UserPrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        # super().__init__(formula_key="stop_gradient", name=name)
        super().__init__(
            name=name,
            formula_key="stop_gradient",
            output=BaseKey(shape=[("Var", ...)], type=Tensor),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class CartesianDifference(UserPrimitiveModel):
    left: Connection
    right: Connection
    output: Connection

    def __init__(
        self,
        left: Tensor[int | float | bool] | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        # super().__init__(formula_key="cartesian_diff", name=name)
        super().__init__(
            name=name,
            formula_key="cartesian_diff",
            output=BaseKey(shape=["N", "M", "dim"], type=Tensor),
            left=BaseKey(shape=["N", "dim"], type=Tensor, value=left),
            right=BaseKey(shape=["M", "dim"], type=Tensor, value=right),
        )
        self._add_constraint(
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


class Concat(UserPrimitiveModel):
    output: Connection
    axis: Connection

    def __init__(
        self,
        n: int,
        axis: int | None | ToBeDetermined = 0,
        *,
        name: str | None = None,
        **kwargs: Tensor[int | float | bool] | ToBeDetermined,
    ) -> None:
        self.factory_args = {"n": n, "axis": axis}

        key_definitions: dict[str, BaseKey] = {}
        key_definitions["output"] = BaseKey(shape=[("Var_out", ...)], type=Tensor)
        key_definitions |= {
            f"input{idx+1}": BaseKey(
                shape=[(f"Var_{idx + 1}", ...)],
                type=Tensor,
                value=kwargs.get(f"input{idx + 1}", TBD),
            )
            for idx in range(n)
        }
        key_definitions["axis"] = BaseKey(type=int | None, value=axis)
        super().__init__(formula_key="concat", name=name, **key_definitions)

        input_keys = [key for key in self.input_keys if key != "axis"]
        self._add_constraint(
            fn=concat_constraints, keys=["output"] + ["axis"] + input_keys
        )
        self._add_constraint(
            fn=general_tensor_type_constraint,
            keys=[PrimitiveModel.output_key] + input_keys,
        )


class PrimitiveUnion(UserPrimitiveModel):
    output: Connection

    def __init__(
        self,
        n: int = 1,
        *,
        name: str | None = None,
        **kwargs: int | float | tuple[int | float, ...] | ToBeDetermined,
    ) -> None:
        self.factory_args = {"n": n}
        input_definitions = {
            f"input{idx + 1}": BaseKey(
                type=int | float | tuple[int | float, ...],
                value=kwargs.get(f"input{idx + 1}", TBD),
            )
            for idx in range(n)
        }

        super().__init__(
            formula_key="union",
            name=name,
            output=BaseKey(type=tuple[int | float, ...]),
            **input_definitions,
        )


class PermuteTensor(UserPrimitiveModel):
    input: Connection
    indices: Connection
    output: Connection

    def __init__(
        self,
        indices: Tensor[int | float | bool] | ToBeDetermined = TBD,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="permute_tensor",
            name=name,
            output=BaseKey(shape=["N", ("Var", ...)], type=Tensor),
            input=BaseKey(shape=["N", ("Var", ...)], type=Tensor, value=input),
            indices=BaseKey(shape=["N"], type=Tensor, value=indices),
        )

        self._add_constraint(
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


class PrimitiveConvolution1D(UserPrimitiveModel):
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
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        weight: Tensor[int | float | bool] | ToBeDetermined = TBD,
        stride: int | ToBeDetermined = TBD,
        padding: int | tuple[int, int] | ToBeDetermined = TBD,
        dilation: int | ToBeDetermined = TBD,
        *,
        bias: Tensor[int | float | bool] | ToBeDetermined = TBD,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"use_bias": use_bias}
        formula_key = "conv1d_bias"
        kwargs: dict[str, BaseKey] = {
            "output": BaseKey(shape=["N", "out_channels", "d_out"], type=Tensor),
            "input": BaseKey(shape=["N", "C_in", "d_in"], type=Tensor, value=input),
            "weight": BaseKey(
                shape=["out_channels", "C_in", "kernel_size"],
                type=Tensor,
                value=weight,
            ),
            "bias": BaseKey(shape=[1, "out_channels", 1], type=Tensor, value=bias),
            "stride": BaseKey(type=int, value=stride),
            "padding": BaseKey(type=int | tuple[int, int], value=padding),
            "dilation": BaseKey(type=int, value=dilation),
        }

        if not use_bias:
            formula_key = "conv1d"
            kwargs.pop("bias")

        super().__init__(formula_key=formula_key, name=name, **kwargs)

        self._add_constraint(
            fn=conv_1d_constraints,
            keys=["output", "input", "stride", "padding", "dilation", "weight"],
        )

        constraint_keys = ["input", "weight"]
        if use_bias:
            constraint_keys.append("bias")

        self._add_constraint(
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

        if "bias" not in self.input_keys and bias != NOT_GIVEN:
            raise ValueError(f"PrimitiveModel does not have 'bias' input. \
                             Got {bias} as bias argument!")
        elif "bias" in self.input_keys:
            kwargs |= {"bias": bias}

        return super().__call__(**kwargs)


class PrimitiveConvolution2D(UserPrimitiveModel):
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
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        weight: Tensor[int | float | bool] | ToBeDetermined = TBD,
        stride: int | tuple[int, int] | ToBeDetermined = TBD,
        padding: int
        | tuple[int, int]
        | tuple[tuple[int, int], tuple[int, int]]
        | ToBeDetermined = TBD,
        dilation: int | tuple[int, int] | ToBeDetermined = TBD,
        *,
        bias: Tensor[int | float | bool] | ToBeDetermined = TBD,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"use_bias": use_bias}
        formula_key = "conv2d_bias"
        kwargs: dict[str, BaseKey] = {
            "output": BaseKey(
                shape=["N", "out_channels", "H_out", "W_out"], type=Tensor
            ),
            "input": BaseKey(shape=["N", "C_in", "H", "W"], type=Tensor, value=input),
            "weight": BaseKey(
                shape=["out_channels", "C_in", "kernel_size_0", "kernel_size_1"],
                type=Tensor,
                value=weight,
            ),
            "bias": BaseKey(shape=[1, "out_channels", 1, 1], type=Tensor, value=bias),
            "stride": BaseKey(type=int | tuple[int, int], value=stride),
            "padding": BaseKey(
                type=int | tuple[int, int] | tuple[tuple[int, int], tuple[int, int]],
                value=padding,
            ),
            "dilation": BaseKey(type=int | tuple[int, int], value=dilation),
        }

        if not use_bias:
            formula_key = "conv2d"
            kwargs.pop("bias")

        super().__init__(formula_key=formula_key, name=name, **kwargs)

        self._add_constraint(
            fn=conv_2d_constraints,
            keys=["output", "input", "stride", "padding", "dilation", "weight"],
        )

        constraint_keys = ["input", "weight"]
        if use_bias:
            constraint_keys.append("bias")
        self._add_constraint(
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

        if "bias" not in self.input_keys and bias != NOT_GIVEN:
            raise ValueError(
                "PrimitiveModel does not have 'bias' input."
                " Got {bias} as bias argument!"
            )
        elif "bias" in self.input_keys:
            kwargs |= {"bias": bias}
        return super().__call__(**kwargs)


class Flatten(UserPrimitiveModel):
    input: Connection
    start_dim: Connection
    end_dim: Connection
    output: Connection

    def __init__(
        self,
        start_dim: int | ToBeDetermined = 0,
        end_dim: int | ToBeDetermined = -1,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"start_dim": start_dim, "end_dim": end_dim}

        key_definitions: dict[str, BaseKey] = {
            "output": BaseKey(shape=[("C_out", ...)], type=Tensor),
            "input": BaseKey(shape=[("C_in", ...)], type=Tensor, value=input),
            "start_dim": BaseKey(type=int, value=start_dim),
            "end_dim": BaseKey(type=int, value=end_dim),
        }
        super().__init__(formula_key="flatten", name=name, **key_definitions)

        self._add_constraint(
            fn=flatten_constrains,
            keys=[PrimitiveModel.output_key, "input", "start_dim", "end_dim"],
        )
        self._add_constraint(
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


class PrimitiveMaxPool1D(UserPrimitiveModel):
    input: Connection
    kernel_size: Connection
    stride: Connection
    padding: Connection
    dilation: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        kernel_size: int | ToBeDetermined = TBD,
        stride: int | ToBeDetermined = TBD,
        padding: int | tuple[int, int] | ToBeDetermined = TBD,
        dilation: int | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="max_pool1d",
            name=name,
            output=BaseKey(shape=["N", ("C_in", ...), "W_out"], type=Tensor),
            input=BaseKey(shape=["N", ("C_in", ...), "W"], type=Tensor, value=input),
            kernel_size=BaseKey(type=int, value=kernel_size),
            stride=BaseKey(type=int, value=stride),
            padding=BaseKey(type=tuple[int, int], value=padding),
            dilation=BaseKey(type=int, value=dilation),
        )
        self._add_constraint(
            fn=sliding_window_1d_constraints,
            keys=["output", "input", "stride", "padding", "dilation", "kernel_size"],
        )
        # TODO: Torch does not accept any int type inputs but JAX implementation does.
        self._add_constraint(
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


class PaddingConverter1D(UserPrimitiveModel):
    input: Connection
    kernel_size: Connection
    output: Connection

    def __init__(
        self,
        kernel_size: int | ToBeDetermined = TBD,
        input: int | PaddingType | tuple[int, int] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="padding_converter_1d",
            name=name,
            output=BaseKey(type=tuple[int, int]),
            input=BaseKey(type=int | PaddingType | tuple[int, int], value=input),
            kernel_size=BaseKey(type=int, value=kernel_size),
        )

        self._add_constraint(
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


class PaddingConverter2D(UserPrimitiveModel):
    input: Connection
    kernel_size: Connection
    output: Connection

    def __init__(
        self,
        kernel_size: int | tuple[int, int] | ToBeDetermined = TBD,
        input: int
        | PaddingType
        | tuple[int, int]
        | tuple[tuple[int, int], tuple[int, int]]
        | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="padding_converter_2d",
            name=name,
            output=BaseKey(
                type=tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]
            ),
            input=BaseKey(
                type=int
                | PaddingType
                | tuple[int, int]
                | tuple[tuple[int, int], tuple[int, int]],
                value=input,
            ),
            kernel_size=BaseKey(type=tuple[int, int], value=kernel_size),
        )

        self._add_constraint(
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


class StrideConverter(UserPrimitiveModel):
    input: Connection
    kernel_size: Connection
    output: Connection

    def __init__(
        self,
        kernel_size: int | tuple[int, int] | ToBeDetermined = TBD,
        input: int | PaddingType | tuple[int, int] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="stride_converter",
            name=name,
            output=BaseKey(type=int | tuple[int, int]),
            input=BaseKey(type=int | PaddingType | tuple[int, int] | None, value=input),
            kernel_size=BaseKey(type=int | tuple[int, int], value=kernel_size),
        )
        self._add_constraint(
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


class TupleConverter(UserPrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: int
        | PaddingType
        | tuple[int, int]
        | tuple[tuple[int, int], tuple[int, int]]
        | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="tuple_converter",
            name=name,
            output=BaseKey(
                type=tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]
            ),
            input=BaseKey(
                type=int
                | PaddingType
                | tuple[int, int]
                | tuple[tuple[int, int], tuple[int, int]]
                | None,
                value=input,
            ),
        )
        self._add_constraint(
            fn=tuple_converter_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class PrimitiveMaxPool2D(UserPrimitiveModel):
    input: Connection
    kernel_size: Connection
    stride: Connection
    padding: Connection
    dilation: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        kernel_size: int | tuple[int, int] | ToBeDetermined = TBD,
        stride: int | tuple[int, int] | ToBeDetermined = TBD,
        padding: int
        | tuple[int, int]
        | tuple[tuple[int, int], tuple[int, int]]
        | ToBeDetermined = TBD,
        dilation: int | tuple[int, int] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="max_pool2d",
            name=name,
            output=BaseKey(shape=["N", ("C_in", ...), "H_out", "W_out"], type=Tensor),
            input=BaseKey(
                shape=["N", ("C_in", ...), "H", "W"],
                type=Tensor,
                value=input,
            ),
            kernel_size=BaseKey(type=tuple[int, int], value=kernel_size),
            stride=BaseKey(type=tuple[int, int], value=stride),
            padding=BaseKey(
                type=tuple[int, int] | tuple[tuple[int, int], tuple[int, int]],
                value=padding,
            ),
            dilation=BaseKey(type=tuple[int, int], value=dilation),
        )

        self._add_constraint(
            fn=sliding_window_2d_constraints,
            keys=["output", "input", "stride", "padding", "dilation", "kernel_size"],
        )
        self._add_constraint(
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


class NormModifier(UserPrimitiveModel):
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
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="norm_modifier",
            name=name,
            output=BaseKey(shape=[], type=Tensor),
            input=BaseKey(shape=[], type=Tensor, value=input),
        )

        self._add_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class DistanceMatrix(UserPrimitiveModel):
    left: Connection
    right: Connection
    norm: Connection
    output: Connection

    # TODO: torch.cdist handles batches of matrices, for now we don't.
    def __init__(
        self,
        left: Tensor[int | float | bool] | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="distance_matrix",
            name=name,
            output=BaseKey(shape=["N", "M"], type=Tensor),
            left=BaseKey(shape=["N", "d"], type=Tensor, value=left),
            right=BaseKey(shape=["M", "d"], type=Tensor, value=right),
            norm=BaseKey(shape=[], type=Tensor),
        )

        self._add_constraint(
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


class PolynomialFeatures(UserPrimitiveModel):
    input: Connection
    degree: Connection
    output: Connection

    def __init__(
        self,
        degree: int | ToBeDetermined = TBD,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="polynomial_features",
            name=name,
            output=BaseKey(shape=["N", "d_out"], type=Tensor),
            input=BaseKey(shape=["N", "d_in"], type=Tensor, value=input),
            degree=BaseKey(type=int, value=degree),
        )

        self._add_constraint(
            fn=polynomial_features_constraints, keys=["output", "input", "degree"]
        )
        self._add_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        degree: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, degree=degree, output=output)


class TsnePJoint(UserPrimitiveModel):
    squared_distances: Connection
    target_perplexity: Connection
    threshold: Connection
    output: Connection

    disposable = True

    def __init__(
        self,
        squared_distances: Tensor[int | float | bool] | ToBeDetermined = TBD,
        target_perplexity: float | ToBeDetermined = TBD,
        threshold: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="tsne_p_joint",
            name=name,
            output=BaseKey(shape=["N", "M"], type=Tensor[float]),
            squared_distances=BaseKey(
                shape=["N", "M"], type=Tensor, value=squared_distances
            ),
            target_perplexity=BaseKey(
                shape=[], type=Tensor[float], value=target_perplexity
            ),
            threshold=BaseKey(shape=[], type=Tensor, value=threshold),
        )

    def __call__(  # type: ignore[override]
        self,
        squared_distances: ConnectionType = NOT_GIVEN,
        target_perplexity: float | ConnectionType = NOT_GIVEN,
        threshold: ConstantType | ConnectionType = Constant.EPSILON,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            squared_distances=squared_distances,
            target_perplexity=target_perplexity,
            threshold=threshold,
            output=output,
        )


class EyeComplement(UserPrimitiveModel):
    N: Connection
    M: Connection
    dtype: Connection
    output: Connection

    def __init__(
        self,
        N: int | ToBeDetermined = TBD,
        M: int | ToBeDetermined | None = None,
        dtype: Dtype | None = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="ones_with_zero_diag",
            name=name,
            output=BaseKey(shape=["N", "M"], type=Tensor[float]),
            N=BaseKey(type=int, value=N),
            M=BaseKey(type=int | None, value=M),
            dtype=BaseKey(type=Dtype | None, value=dtype),
        )
        self._add_constraint(fn=eye_constraints, keys=["output", "N", "M"])

    def __call__(  # type: ignore[override]
        self,
        N: ConnectionType = NOT_GIVEN,
        M: ConnectionType = NOT_GIVEN,
        dtype: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(N=N, M=M, dtype=dtype, output=output)


class Eye(UserPrimitiveModel):
    N: Connection
    M: Connection
    dtype: Connection
    output: Connection

    def __init__(
        self,
        N: int | ToBeDetermined = TBD,
        M: int | ToBeDetermined | None = None,
        dtype: Dtype | None = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="eye",
            name=name,
            output=BaseKey(shape=["N", "M"], type=Tensor[float]),
            N=BaseKey(type=int, value=N),
            M=BaseKey(type=int | None, value=M),
            dtype=BaseKey(type=Dtype | None, value=dtype),
        )
        self._add_constraint(fn=eye_constraints, keys=["output", "N", "M"])

    def __call__(  # type: ignore[override]
        self,
        N: ConnectionType = NOT_GIVEN,
        M: ConnectionType = NOT_GIVEN,
        dtype: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(N=N, M=M, dtype=dtype, output=output)


class Cholesky(UserPrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="cholesky",
            name=name,
            output=BaseKey(shape=["N", "N"], type=Tensor[float]),
            input=BaseKey(shape=["N", "N"], type=Tensor, value=input),
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class GPRAlpha(UserPrimitiveModel):
    label_mu_diff: Connection
    L: Connection
    K_term: Connection
    output: Connection

    def __init__(
        self,
        label_mu_diff: Tensor[int | float | bool] | ToBeDetermined = TBD,
        L: Tensor[int | float | bool] | ToBeDetermined = TBD,
        K_term: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="gpr_alpha",
            name=name,
            output=BaseKey(shape=["N", 1], type=Tensor[float]),
            label_mu_diff=BaseKey(shape=["N", 1], type=Tensor, value=label_mu_diff),
            L=BaseKey(shape=["N", "N"], type=Tensor, value=L),
            K_term=BaseKey(shape=["N", "N"], type=Tensor, value=K_term),
        )

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


class GPRVOuter(UserPrimitiveModel):
    K: Connection
    K_term: Connection
    L: Connection
    output: Connection

    def __init__(
        self,
        K: Tensor[int | float | bool] | ToBeDetermined = TBD,
        K_term: Tensor[int | float | bool] | ToBeDetermined = TBD,
        L: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="gpr_v_outer",
            name=name,
            output=BaseKey(shape=["N", "N"], type=Tensor[float]),
            K=BaseKey(shape=["N", "N"], type=Tensor, value=K),
            K_term=BaseKey(shape=["N", "N"], type=Tensor, value=K_term),
            L=BaseKey(shape=["N", "N"], type=Tensor, value=L),
        )

    def __call__(  # type: ignore[override]
        self,
        K: ConnectionType = NOT_GIVEN,
        K_term: ConnectionType = NOT_GIVEN,
        L: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(K=K, K_term=K_term, L=L, output=output)


class TransposedDiagonal(UserPrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="transposed_diag",
            name=name,
            output=BaseKey(shape=["N", 1], type=Tensor),
            input=BaseKey(shape=["N", "N"], type=Tensor, value=input),
        )

        self._add_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Arange(UserPrimitiveModel):
    start: Connection
    stop: Connection
    step: Connection
    dtype: Connection
    output: Connection

    def __init__(
        self,
        start: int | float | ToBeDetermined = 0,
        stop: int | float | ToBeDetermined = TBD,
        step: int | float | ToBeDetermined = 1,
        dtype: Dtype | None = None,
        *,
        name: str | None = None,
    ) -> None:
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
            output=BaseKey(shape=output_shp, type=Tensor),
            start=BaseKey(type=int | float, value=start),
            stop=BaseKey(type=int | float, value=stop),
            step=BaseKey(type=int | float, value=step),
            dtype=BaseKey(type=Dtype | None, value=dtype),
        )
        # self.set_canonical_input("stop")
        self.set_cin("stop", safe=False)

        if not all_defined:
            self._add_constraint(
                fn=arange_constraints, keys=["output", "start", "stop", "step"]
            )
        self._add_constraint(
            fn=general_tensor_type_constraint,
            keys=[PrimitiveModel.output_key, "start", "stop", "step"],
        )

    def __call__(  # type: ignore[override]
        self,
        start: ConnectionType = NOT_GIVEN,
        stop: ConnectionType = NOT_GIVEN,
        step: ConnectionType = NOT_GIVEN,
        dtype: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            start=start, stop=stop, step=step, dtype=dtype, output=output
        )


class Randn(UserPrimitiveModel):
    shape: Connection
    key: Connection
    dtype: Connection
    output: Connection

    def __init__(
        self,
        shape: tuple[int, ...] | ToBeDetermined = TBD,
        key: int | ToBeDetermined = TBD,
        dtype: Dtype | None = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="randn",
            name=name,
            output=BaseKey(shape=[("output", ...)], type=Tensor),
            shape=BaseKey(type=tuple[int, ...], value=shape),
            key=BaseKey(type=int, value=key),
            dtype=BaseKey(type=Dtype | None, value=dtype),
        )

        op_model = next(iter(self.dag))
        assert isinstance(op_model, PrimitiveModel)
        op_model.random_keys.add("key")  # since random_keys must be in primitive models
        self.add_constraint(randn_constraints, keys=["output", "shape"])

    def __call__(  # type: ignore[override]
        self,
        shape: ConnectionType = NOT_GIVEN,
        key: ConnectionType = NOT_GIVEN,
        dtype: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(shape=shape, key=key, dtype=dtype, output=output)


class BroadcastTo(UserPrimitiveModel):
    input: Connection
    shape: Connection
    output: Connection

    def __init__(
        self,
        shape: tuple[int, ...] | ToBeDetermined = TBD,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="broadcast_to",
            name=name,
            output=BaseKey(shape=[("output", ...)], type=Tensor),
            input=BaseKey(shape=[("input", ...)], type=Tensor, value=input),
            shape=BaseKey(type=tuple[int, ...], value=shape),
        )

        self.add_constraint(
            fn=broadcast_to_constraints, keys=["output", "shape", "input"]
        )
        self._add_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        shape: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, shape=shape, output=output)


class Eigvalsh(UserPrimitiveModel):
    K_term: Connection
    L: Connection
    threshold: Connection
    output: Connection

    def __init__(
        self,
        K_term: Tensor[int | float | bool] | ToBeDetermined = TBD,
        L: Tensor[int | float | bool] | ToBeDetermined = TBD,
        threshold: ConstantType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="eigvalsh",
            name=name,
            output=BaseKey(shape=["N", 1], type=Tensor[float]),
            K_term=BaseKey(shape=["N", "N"], type=Tensor, value=K_term),
            L=BaseKey(shape=["N", "N"], type=Tensor, value=L),
            threshold=BaseKey(shape=[], type=Tensor, value=threshold),
        )

    def __call__(  # type: ignore[override]
        self,
        K_term: ConnectionType = NOT_GIVEN,
        L: ConnectionType = NOT_GIVEN,
        threshold: ConstantType | ConnectionType = Constant.EPSILON,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(K_term=K_term, L=L, threshold=threshold, output=output)


class Squeeze(UserPrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="squeeze",
            name=name,
            output=BaseKey(shape=[("Var_out", ...)], type=Tensor),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
        )

        self._add_constraint(fn=squeeze_constraints, keys=["output", "input"])
        self._add_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class AUCCore(UserPrimitiveModel):
    input: Connection
    label: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        label: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="auc_core",
            name=name,
            output=BaseKey(shape=[2, "M"], type=Tensor[float]),
            input=BaseKey(shape=["N"], type=Tensor, value=input),
            label=BaseKey(shape=["N"], type=Tensor, value=label),
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        label: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, label=label, output=output)


class Embedding(UserPrimitiveModel):
    input: Connection
    weight: Connection
    output: Connection

    def __init__(
        self,
        num_embeddings: int | None = None,
        dim: int | None = None,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        weight: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        out_dim: int | str = "dim" if dim is None else dim

        super().__init__(
            formula_key="primitive_embedding",
            name=name,
            output=BaseKey(shape=[("N1", ...), "d1", out_dim], type=Tensor),
            input=BaseKey(shape=[("N1", ...), "d1"], type=Tensor[int], value=input),
            weight=BaseKey(shape=[num_embeddings, out_dim], type=Tensor, value=weight),
        )

        self._add_constraint(
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


class ScaledDotProduct(UserPrimitiveModel):
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
        is_causal: bool | ToBeDetermined = True,
        scale: None | int | float | ToBeDetermined = None,
        dropout_p: float | ToBeDetermined = 0.0,
        use_attn_mask: bool = False,
        query: Tensor[int | float | bool] | ToBeDetermined = TBD,
        key: Tensor[int | float | bool] | ToBeDetermined = TBD,
        value: Tensor[int | float | bool] | ToBeDetermined = TBD,
        attn_mask: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        # TODO: Reconsider how to get attn_mask, could it be A?
        assert (
            not isinstance(is_causal, bool) or not is_causal or not use_attn_mask
        ), "Causal attention is not support attn_mask!"
        assert isinstance(use_attn_mask, bool), "use_attn_mask must be a boolean value!"
        self.use_attn_mask = use_attn_mask

        formula_key = "scaled_dot_product_attention"
        kwargs: dict[str, BaseKey] = {
            "output": BaseKey(shape=[("Var", ...), "L", "O"], type=Tensor[float]),
            "query": BaseKey(shape=[("Var", ...), "L", "E"], type=Tensor, value=query),
            "key": BaseKey(shape=[("Var", ...), "S", "E"], type=Tensor, value=key),
            "value": BaseKey(shape=[("Var", ...), "S", "O"], type=Tensor, value=value),
            "dropout_p": BaseKey(type=float, value=dropout_p),
            "attn_mask": BaseKey(type=NoneType, value=None),
            "is_causal": BaseKey(type=bool, value=is_causal),
            "scale": BaseKey(type=NoneType | int | float, value=scale),
        }

        if use_attn_mask:
            kwargs["attn_mask"] = BaseKey(
                shape=["L", "S"], type=Tensor, value=attn_mask
            )

        super().__init__(formula_key=formula_key, name=name, **kwargs)

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
            and isinstance(attn_mask, BaseKey)
            and attn_mask.value is not None  # TODO: Here will be updated!
        ):
            raise KeyError(
                "PrimitiveModel does not have 'attn_mask' input."
                " Got attn_mask argument!"
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


class PositionalEncoding(UserPrimitiveModel):
    input: Connection
    hidden_dim: Connection
    max_len: Connection
    output: Connection

    # TODO: Try to move to Logical composite models.
    def __init__(
        self,
        hidden_dim: int | ToBeDetermined,
        max_len: int | ToBeDetermined = 5000,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"hidden_dim": hidden_dim, "max_len": max_len}

        super().__init__(
            formula_key="positional_encoding",
            name=name,
            output=BaseKey(shape=[("N1", ...)], type=Tensor),
            input=BaseKey(shape=[("N1", ...)], type=Tensor, value=input),
            hidden_dim=BaseKey(type=int, value=hidden_dim),
            max_len=BaseKey(type=int, value=max_len),
        )

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


class SwapAxes(UserPrimitiveModel):
    input: Connection
    axis1: Connection
    axis2: Connection
    output: Connection

    def __init__(
        self,
        axis1: int | ToBeDetermined,
        axis2: int | ToBeDetermined,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"axis1": axis1, "axis2": axis2}

        super().__init__(
            formula_key="swapaxes",
            name=name,
            output=BaseKey(shape=[("Var_out", ...)], type=Tensor),
            input=BaseKey(shape=[("Var_in", ...)], type=Tensor, value=input),
            axis1=BaseKey(type=int, value=axis1),
            axis2=BaseKey(type=int, value=axis2),
        )

        self._add_constraint(
            fn=swap_axes_constraints, keys=["output", "input", "axis1", "axis2"]
        )
        self._add_constraint(
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


class Where(UserPrimitiveModel):
    cond: Connection
    input1: Connection
    input2: Connection
    output: Connection

    def __init__(
        self,
        cond: Tensor[int | float | bool] | ToBeDetermined = TBD,
        input1: Tensor[int | float | bool] | ToBeDetermined = TBD,
        input2: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="where",
            name=name,
            output=BaseKey(shape=[("Var_out", ...)], type=Tensor),
            cond=BaseKey(shape=[("Var3", ...)], type=Tensor[bool], value=cond),
            input1=BaseKey(shape=[("Var1", ...)], type=Tensor, value=input1),
            input2=BaseKey(shape=[("Var2", ...)], type=Tensor, value=input2),
        )

        self._add_constraint(
            fn=where_constrains, keys=["output", "cond", "input1", "input2"]
        )
        self._add_constraint(
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


class IsNan(UserPrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="isnan",
            name=name,
            output=BaseKey(shape=[("Var", ...)], type=Tensor[bool]),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Unique(UserPrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="unique",
            name=name,
            input=BaseKey(shape=[("Var1", ...)], type=Tensor, value=input),
            output=BaseKey(shape=[("Var2", ...)], type=Tensor),
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Trapezoid(UserPrimitiveModel):
    y: Connection
    x: Connection
    output: Connection

    def __init__(
        self,
        x: Tensor[int | float | bool] | ToBeDetermined = TBD,
        y: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="trapezoid",
            name=name,
            output=BaseKey(shape=[], type=Tensor),
            y=BaseKey(shape=[("Var", ...)], type=Tensor, value=y),
            x=BaseKey(shape=[("Var", ...)], type=Tensor, value=x),
        )

    def __call__(  # type: ignore[override]
        self,
        y: ConnectionType = NOT_GIVEN,
        x: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(y=y, x=x, output=output)


class NanToNum(UserPrimitiveModel):
    input: Connection
    nan: Connection
    posinf: Connection
    neginf: Connection
    output: Connection

    def __init__(
        self,
        nan: float | ToBeDetermined = 0.0,
        posinf: float | None | ToBeDetermined = None,
        neginf: float | None | ToBeDetermined = None,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="nan_to_num",
            name=name,
            output=BaseKey(shape=[("Var", ...)], type=Tensor),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
            nan=BaseKey(type=float, value=nan),
            posinf=BaseKey(type=float | None, value=posinf),
            neginf=BaseKey(type=float | None, value=neginf),
        )

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


class Pad(UserPrimitiveModel):
    input: Connection
    pad_width: Connection
    output: Connection

    def __init__(
        self,
        pad_width: tuple[tuple[int, int], ...] | ToBeDetermined = TBD,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="pad",
            name=name,
            output=BaseKey(shape=[("Var2", ...)], type=Tensor),
            input=BaseKey(shape=[("Var1", ...)], type=Tensor, value=input),
            pad_width=BaseKey(type=tuple[tuple[int, int], ...], value=pad_width),
        )

        self._add_constraint(
            fn=pad_constraints, keys=[PrimitiveModel.output_key, "input", "pad_width"]
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        pad_width: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, pad_width=pad_width, output=output)


class ZerosLike(UserPrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="zeros_like",
            name=name,
            output=BaseKey(shape=[("Var", ...)], type=Tensor),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)
