from abc import ABC
import torch.nn.functional as F
from torch import Tensor
import torch.nn.modules as nn


class Conv2DSamePadding(nn.Conv2d, ABC):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2DSamePadding, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                                padding=padding, dilation=dilation, groups=groups, bias=bias)

    def conv2d_same_padding(self, input_data, weight, stride, dilation, groups=1):
        input_rows = input_data.shape[2]
        input_cols = input_data.shape[3]

        filter_rows = weight.shape[2]
        filter_cols = weight.shape[3]

        if isinstance(stride, int):
            stride_rows = stride
            stride_cols = stride
        else:
            stride_rows = stride[0]
            stride_cols = stride[1]

        if isinstance(dilation, int):
            dilation_rows = dilation
            dilation_cols = dilation
        else:
            dilation_rows = dilation[0]
            dilation_cols = dilation[1]

        out_rows = (input_rows + stride_rows - 1) // stride_rows
        padding_rows = max(0, (out_rows - 1) * stride_rows + (filter_rows - 1) * dilation_rows + 1 - input_rows)
        rows_odd = (padding_rows % 2 != 0)

        out_cols = (input_cols + stride_cols - 1) // stride_cols
        padding_cols = max(0, (out_cols - 1) * stride_cols + (filter_cols - 1) * dilation_cols + 1 - input_cols)
        cols_odd = (padding_cols % 2 != 0)

        if rows_odd or cols_odd:
            input_data = F.pad(input_data, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(input_data, weight, self.bias, stride, padding=(padding_rows // 2, padding_cols // 2),
                        dilation=dilation, groups=groups)

    def forward(self, input_data: Tensor) -> Tensor:
        return self.conv2d_same_padding(input_data, self.weight,
                                        stride=self.stride, dilation=self.dilation, groups=self.groups)


class Reshape(nn.Module, ABC):
    def __init__(self, channels, rows, cols):
        super(Reshape, self).__init__()
        self.channels = channels
        self.rows = rows
        self.cols = cols

    def forward(self, input_data: Tensor) -> Tensor:
        batch_size = input_data.shape[0]
        output_data = input_data.view(batch_size, self.channels, self.rows, self.cols)
        return output_data
