import numpy as np
from typing import Any



def is_int_tuple_tuple(
    data: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]],
):
    return isinstance(data[0], tuple)



def get_submatrices2d(
    input: np.ndarray[Any, Any],
    output_size: tuple[int, ...],
    kernel_height_size: int,
    kernel_width_size: int,
    padding: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (0, 0)),
    stride: int = 1,
    dilate: int = 0,
) -> np.ndarray[Any, Any]:
    working_input = input
    working_pad = padding
    # dilate the input if necessary
    if dilate != 0:
        working_input = np.insert(working_input, range(1, input.shape[2]), 0, axis=2)
        working_input = np.insert(working_input, range(1, input.shape[3]), 0, axis=3)

    # pad the input if necessary
    if is_int_tuple_tuple(padding):
        working_input = np.pad(
            working_input,
            pad_width=(
                (0, 0),
                (0, 0),
                (working_pad[0][0], working_pad[0][1]),
                (working_pad[1][0], working_pad[1][1]),
            ),
            mode="constant",
            constant_values=(0.0,),
        )

    *_, out_h, out_w = output_size
    out_b, out_c, *_ = input.shape
    batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides

    return np.lib.stride_tricks.as_strided(
        working_input,
        (out_b, out_c, out_h, out_w, kernel_height_size, kernel_width_size),
        (
            batch_str,
            channel_str,
            stride * kern_h_str,
            stride * kern_w_str,
            kern_h_str,
            kern_w_str,
        ),
    )


if __name__ == "__main__":
    input = np.ones((1, 1, 10,10))
    matrices = get_submatrices2d(input, (9,9), 3, 3, ((0, 0), (0, 0)), 1, 0)
    print(matrices)

