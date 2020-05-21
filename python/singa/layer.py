# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# =============================================================================

import math
import numpy as np

from singa import utils
from .tensor import Tensor
from . import singa_wrap as singa


class Layer(object):

    def __init__(self):
        self.allow_params = []
        pass

    def device_check(self, *inputs):
        x_device = inputs[0].device
        x_dev_id = x_device.id()
        for var in inputs:
            if var.device.id() != x_dev_id:
                var.to_device(x_device)

    def find_sublayers(self):
        # return a list whose elements are in form of (attribute_name,
        # sublayer)
        sublayers = []
        for attr in self.__dict__:
            if isinstance(self.__dict__[attr], Layer):
                sublayers.append((attr, self.__dict__[attr]))
        return sublayers

    name_sep=":"
    def set_names(self, prefix='', sep=':'):
        sep=self.name_sep
        sublayers = self.find_sublayers()
        for l_name, l in sublayers:
            l.name = prefix + l_name + sep
            l.set_names(l.name)

    states = []
    params = []
    def get_states(self):
        sublayers = self.find_sublayers()
        states = dict()
        if sublayers:
            for sublayer_name, sublayer in sublayers:
                states.update(sublayer.get_states())
        else:
            for state in self.states:
                states[state.name]=state
        return states

    def set_states(self, states):
        sublayers = self.find_sublayers()
        if sublayers:
            for sublayer_name, sublayer in sublayers:
                sublayer.set_states(states)
        else:
            for state in self.states:
                state.copy_data(states[state.name])

    def get_params(self):
        sublayers = self.find_sublayers()
        params = dict()

        if sublayers:
            for sublayer_name, sublayer in sublayers:
                params.update(sublayer.get_params())
        else:
            for param in self.params:
                params[param.name] = param

        return params

    def set_params(self, parameters):
        sublayers = self.find_sublayers()
        if sublayers:
            for sublayer_name, sublayer in sublayers:
                sublayer.set_params(parameters)
        else:
            for param in self.params:
                param.copy_data(parameters[param.name])


class Linear(Layer):
    """
    Generate a Linear operator
    """

    def initialize(self, x):
        prev_state = x.device.graph_enabled()
        x.device.EnableGraph(False)

        self.in_features = x.shape[1]
        w_shape = (self.in_features, self.out_features)
        b_shape = (self.out_features,)

        self.W = Tensor(shape=w_shape, requires_grad=True, stores_grad=True)
        self.W.name = self.name+"W"
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        self.W.gaussian(0.0, std)

        self.params = [self.W]

        if self.bias:
            self.b = Tensor(shape=b_shape, requires_grad=True, stores_grad=True)
            self.b.set_value(0.0)
            self.b.name = self.name+"b"
            self.params.append(self.b)

        self.states = self.params

        x.device.EnableGraph(prev_state)

    # TODO: replace current with
    #   def __init__(self, out_features, bias=True):
    def __init__(self, out_features, *args, bias=True, **kwargs):
        """
        Args:
            in_channels: int, the channel of input
            out_channels: int, the channel of output, also is the number of
                filters
            bias: bool
        """
        super(Linear, self).__init__()

        self.out_features = out_features

        # TODO: for backward compatibility, to remove
        if len(args) > 0:
            self.in_features = out_features
            self.out_features = args[0]
        if len(args) > 1:
            self.bias = args[1]

        self.bias = bias
        self.has_initialized = False

    def __call__(self, x):
        if not self.has_initialized:
            self.in_features = x.shape[1]
            self.initialize(x)
            self.has_initialized = True

        if self.bias:
            self.device_check(x, self.W, self.b)
        else:
            self.device_check(x, self.W)
        assert x.shape[1] == self.W.shape[0], (
            "Linear layer expects input features size %d received %d" %
            (self.W.shape[0], x.shape[1]))

        y = autograd.matmul(x, self.W)
        if self.bias:
            y = autograd.add_bias(y, self.b, axis=0)
        return y

class Conv2d(Layer):
    """
    Generate a Conv 2d operator
    """

    def initialize(self, x):
        prev_state = x.device.graph_enabled()
        x.device.EnableGraph(False)

        w_shape = (
            self.out_channels,
            int(self.in_channels / self.group),
            self.kernel_size[0],
            self.kernel_size[1],
        )

        self.W = Tensor(shape=w_shape, requires_grad=True, stores_grad=True)
        # std = math.sqrt(
        # 2.0 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1] +
        # self.out_channels))
        std = math.sqrt(
            2.0 / (w_shape[1] * self.kernel_size[0] * self.kernel_size[1] +
                   self.out_channels))
        self.W.gaussian(0.0, std)
        self.W.name = self.name + "W"

        self.params = [self.W]
        if self.bias:
            b_shape = (self.out_channels,)
            self.b = Tensor(shape=b_shape, requires_grad=True, stores_grad=True)
            self.b.set_value(0.0)
            self.b.name = self.name + "b"
            self.params.append(self.b)
        else:
            # to keep consistency when to do forward.
            self.b = None
            # Tensor(data=CTensor([]), requires_grad=False, stores_grad=False)
        self.states = self.params

        x.device.EnableGraph(prev_state)

    def __init__(self,
                 out_channels,
                 kernel_size,
                 *args,
                 stride=1,
                 padding=0,
                 dilation=1,
                 group=1,
                 bias=True,
                 pad_mode="NOTSET",
                 **kwargs):
        """
        Args:
            in_channels (int): the channel of input
            out_channels (int): the channel of output, also is the number of filters
            kernel_size (int or tuple): kernel size for two direction of each
                axis. For example, (2, 3), the first 2 means will add 2 at the
                beginning and also 2 at the end for its axis.and if a int is
                accepted, the kernel size will be initiated as (int, int)
            stride (int or tuple): stride, the logic is the same as kernel size.
            padding (int): tuple, list or None, padding, the logic is the same
                as kernel size. However, if you set pad_mode as "SAME_UPPER" or
                "SAME_LOWER" mode, you can set padding as None, and the padding
                will be computed automatically.
            dilation (int): only support 1
            group (int): group
            bias (bool): bias
            pad_mode (string): can be NOTSET, SAME_UPPER, or SAME_LOWER, where
                default value is NOTSET, which means explicit padding is used.
                SAME_UPPER or SAME_LOWER mean pad the input so that the output
                spatial size match the input. In case of odd number add the extra
                padding at the end for SAME_UPPER and at the beginning for SAME_LOWER.
        """
        super(Conv2d, self).__init__()

        # the old code create the layer like: Conv2d(8, 16, 3)， or Conv2d(8, 16, 3, stride=1)
        # the following code block is for backward compatibility
        if len(args) >0:
            self.in_channels=out_channels
            self.out_channel = kernel_size
            self.kernel_size = args[0]
        if len(args) > 1:
            self.stride = args[1]
        if len(args) > 2:
            self.padding = args[2]


        self.has_initialized = False

        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.group = group
        self.bias = bias
        self.pad_mode = pad_mode


        assert (self.out_channels >= self.group and self.out_channels %
                self.group == 0), "out_channels and group dismatched."

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            raise TypeError("Wrong kernel_size type.")

        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple):
            self.stride = stride
        else:
            raise TypeError("Wrong stride type.")

        self.odd_padding = (0, 0, 0, 0)
        if isinstance(padding, int):
            self.padding = (padding, padding)
        elif isinstance(padding, tuple) or isinstance(padding, list):
            if len(padding) == 2:
                self.padding = padding
            elif len(padding) == 4:
                _h_mask = padding[0] - padding[1]
                _w_mask = padding[2] - padding[3]
                # the odd paddding is the value that cannot be handled by the tuple padding (w, h) mode
                # so we need to firstly handle the input, then use the nomal padding method.
                self.odd_padding = (max(_h_mask, 0), max(-_h_mask, 0),
                                    max(_w_mask, 0), max(-_w_mask, 0))
                self.padding = (
                    padding[0] - self.odd_padding[0],
                    padding[2] - self.odd_padding[2],
                )
            else:
                raise TypeError("Wrong padding value.")

        if dilation != 1:
            raise ValueError("Not implemented yet")

        self.bias = bias

        self.inner_params = {
            "cudnn_prefer": "fastest",
            "workspace_MB_limit": 1024,
        }
        # TODO valid value of inner_params check

        for kwarg in kwargs:
            if kwarg not in self.inner_params:
                raise TypeError("Keyword argument not understood:", kwarg)
            else:
                self.inner_params[kwarg] = kwargs[kwarg]

        self.pad_mode = pad_mode

    def __call__(self, x):
        if "in_channels" in self.__dict__:
            assert x.shape[1] == self.in_channels, "in_channels mismatched"

        if not self.has_initialized:
            self.in_channels = x.shape[1]
            assert (self.group >= 1 and self.in_channels %
                    self.group == 0), "please set reasonable group."
            self.initialize(x)
            self.has_initialized = True

        # if same pad mode, re-compute the padding
        if self.pad_mode in ("SAME_UPPER", "SAME_LOWER"):
            self.padding, self.odd_padding = utils.get_padding_shape(
                self.pad_mode, x.shape[2:], self.kernel_size, self.stride)

        if self.bias:
            self.device_check(x, self.W, self.b)
        else:
            self.device_check(x, self.W)

        if x.device.id() == -1:
            if self.group != 1:
                raise ValueError("Not implemented yet")
            else:
                if (not hasattr(self, "handle")) or (x.shape[0] !=
                                                     self.handle.batchsize):
                    self.handle = singa.ConvHandle(
                        x.data,
                        self.kernel_size,
                        self.stride,
                        self.padding,
                        self.in_channels,
                        self.out_channels,
                        self.bias,
                        self.group,
                    )
        else:
            if (not hasattr(self,
                            "handle")) or (x.shape[0] != self.handle.batchsize):
                self.handle = singa.CudnnConvHandle(
                    x.data,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.in_channels,
                    self.out_channels,
                    self.bias,
                    self.group,
                )

        y = autograd.conv2d(self.handle, x, self.W, self.b, self.odd_padding)
        return y

class SeparableConv2d(Layer):
    """
    Generate a Conv 2d operator
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
    ):
        """
        Args:
            in_channels (int): the channel of input
            out_channels (int): the channel of output, also is the number of filters
            kernel_size (int or tuple): kernel size for two direction of each
                axis. For example, (2, 3), the first 2 means will add 2 at the
                beginning and also 2 at the end for its axis.and if a int is
                accepted, the kernel size will be initiated as (int, int)
            stride (int or tuple): stride, the logic is the same as kernel size.
            padding (int): tuple, list or None, padding, the logic is the same
                as kernel size. However, if you set pad_mode as "SAME_UPPER" or
                "SAME_LOWER" mode, you can set padding as None, and the padding
                will be computed automatically.
            bias (bool): bias
        """
        super(SeparableConv2d, self).__init__()

        self.depthwise_conv = Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            group=in_channels,
            bias=bias,
        )

        self.point_conv = Conv2d(in_channels, out_channels, 1, bias=bias)

    def __call__(self, x):
        y = self.depthwise_conv(x)
        y = self.point_conv(y)
        return y


class BatchNorm2d(Layer):
    """
    Generate a BatchNorm 2d operator
    """
    def initialize(self, x):
        prev_state = x.device.graph_enabled()
        x.device.EnableGraph(False)

        param_shape = (self.channels,)

        self.scale = Tensor(shape=param_shape,
                            requires_grad=True,
                            stores_grad=True)
        self.scale.name = self.name + "scale"
        self.scale.set_value(1.0)

        self.bias = Tensor(shape=param_shape,
                           requires_grad=True,
                           stores_grad=True)
        self.bias.set_value(0.0)
        self.bias.name = self.name + "bias"

        self.running_mean = Tensor(shape=param_shape,
                                   requires_grad=False,
                                   stores_grad=False)
        self.running_mean.set_value(0.0)
        self.running_mean.name = self.name + "running_mean"

        self.running_var = Tensor(shape=param_shape,
                                  requires_grad=False,
                                  stores_grad=False)
        self.running_var.set_value(1.0)
        self.running_var.name = self.name + "running_var"

        self.params = [self.scale, self.bias]
        self.states = self.params + [self.running_mean, self.running_var]

        x.device.EnableGraph(prev_state)

    def __init__(self, num_features, momentum=0.9):
        """
        Args:
            num_features (int): int, the channel of input
            momentum (float): Factor used in computing the running mean and
                variance.
        """
        super(BatchNorm2d, self).__init__()

        self.has_initialized = False

        self.channels = num_features
        self.momentum = momentum

    def __call__(self, x):
        assert x.shape[1] == self.channels, (
            "number of channels dismatched. %d vs %d" %
            (x.shape[1], self.channels))

        if not self.has_initialized:
            self.initialize(x)
            self.has_initialized = True

        self.device_check(x, self.scale, self.bias, self.running_mean,
                          self.running_var)

        if x.device.id() == -1:
            if not hasattr(self, "handle"):
                self.handle = singa.BatchNormHandle(self.momentum, x.data)
            elif x.shape[0] != self.handle.batchsize:
                self.handle = singa.BatchNormHandle(self.momentum, x.data)
        else:
            if not hasattr(self, "handle"):
                self.handle = singa.CudnnBatchNormHandle(self.momentum, x.data)
            elif x.shape[0] != self.handle.batchsize:
                self.handle = singa.CudnnBatchNormHandle(self.momentum, x.data)

        y = autograd.batchnorm_2d(
            self.handle,
            x,
            self.scale,
            self.bias,
            self.running_mean,
            self.running_var,
        )
        return y


class Pooling2d(Layer):
    """
    Generate a Pooling 2d operator
    """

    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 is_max=True,
                 pad_mode="NOTSET"):
        """
        Args:
            kernel_size (int or tuple): kernel size for two direction of each
                axis. For example, (2, 3), the first 2 means will add 2 at the
                beginning and also 2 at the end for its axis.and if a int is
                accepted, the kernel size will be initiated as (int, int)
            stride (int or tuple): stride, the logic is the same as kernel size.
            padding (int): tuple, list or None, padding, the logic is the same
                as kernel size. However, if you set pad_mode as "SAME_UPPER" or
                "SAME_LOWER" mode, you can set padding as None, and the padding
                will be computed automatically.
            is_max (bool): is max pooling or avg pooling
            pad_mode (string): can be NOTSET, SAME_UPPER, or SAME_LOWER, where
                default value is NOTSET, which means explicit padding is used.
                SAME_UPPER or SAME_LOWER mean pad the input so that the output
                spatial size match the input. In case of odd number add the extra
                padding at the end for SAME_UPPER and at the beginning for SAME_LOWER.
        """
        super(Pooling2d, self).__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            raise TypeError("Wrong kernel_size type.")

        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple):
            self.stride = stride
            assert stride[0] > 0 or (kernel_size[0] == 1 and padding[0] == 0), (
                "stride[0]=0, but kernel_size[0]=%d, padding[0]=%d" %
                (kernel_size[0], padding[0]))
        else:
            raise TypeError("Wrong stride type.")

        self.odd_padding = (0, 0, 0, 0)
        if isinstance(padding, int):
            self.padding = (padding, padding)
        elif isinstance(padding, tuple) or isinstance(padding, list):
            if len(padding) == 2:
                self.padding = padding
            elif len(padding) == 4:
                _h_mask = padding[0] - padding[1]
                _w_mask = padding[2] - padding[3]
                # the odd paddding is the value that cannot be handled by the tuple padding (w, h) mode
                # so we need to firstly handle the input, then use the nomal padding method.
                self.odd_padding = (max(_h_mask, 0), max(-_h_mask, 0),
                                    max(_w_mask, 0), max(-_w_mask, 0))
                self.padding = (
                    padding[0] - self.odd_padding[0],
                    padding[2] - self.odd_padding[2],
                )
            else:
                raise TypeError("Wrong padding value.")

        self.is_max = is_max
        self.pad_mode = pad_mode

    def __call__(self, x):
        # if same pad mode, re-compute the padding
        if self.pad_mode in ("SAME_UPPER", "SAME_LOWER"):
            self.padding, self.odd_padding = utils.get_padding_shape(
                self.pad_mode, x.shape[2:], self.kernel_size, self.stride)

        out_shape_h = (int(
            (x.shape[2] + 2 * self.padding[0] - self.kernel_size[0]) //
            self.stride[0]) + 1)
        out_shape_w = (int(
            (x.shape[3] + 2 * self.padding[1] - self.kernel_size[1]) //
            self.stride[1]) + 1)
        if x.device.id() == -1:
            if not hasattr(self, "handle"):
                self.handle = singa.PoolingHandle(
                    x.data,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.is_max,
                )
            elif (x.shape[0] != self.handle.batchsize or
                  out_shape_h != self.handle.pooled_height or
                  out_shape_w != self.handle.pooled_width):
                self.handle = singa.PoolingHandle(
                    x.data,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.is_max,
                )
        else:
            if not hasattr(self, "handle"):
                self.handle = singa.CudnnPoolingHandle(
                    x.data,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.is_max,
                )
            elif (x.shape[0] != self.handle.batchsize or
                  out_shape_h != self.handle.pooled_height or
                  out_shape_w != self.handle.pooled_width):
                self.handle = singa.CudnnPoolingHandle(
                    x.data,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.is_max,
                )

        y = autograd.pooling_2d(self.handle, x, self.odd_padding)
        return y


class MaxPool2d(Pooling2d):
    """
    Generate a Max Pooling 2d operator
    """

    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 odd_padding=(0, 0, 0, 0)):
        """
        Args:
            kernel_size (int or tuple): kernel size for two direction of each
                axis. For example, (2, 3), the first 2 means will add 2 at the
                beginning and also 2 at the end for its axis.and if a int is
                accepted, the kernel size will be initiated as (int, int)
            stride (int or tuple): stride, the logic is the same as kernel size.
            padding (int): tuple, list or None, padding, the logic is the same
                as kernel size. However, if you set pad_mode as "SAME_UPPER" or
                "SAME_LOWER" mode, you can set padding as None, and the padding
                will be computed automatically.
            odd_padding (tuple of four int): the odd paddding is the value
                that cannot be handled by the tuple padding (w, h) mode so
                it needs to firstly handle the input, then use the normal
                padding method.
        """
        super(MaxPool2d, self).__init__(kernel_size, stride, padding, True,
                                        odd_padding)


class AvgPool2d(Pooling2d):

    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 odd_padding=(0, 0, 0, 0)):
        """
        Args:
            kernel_size (int or tuple): kernel size for two direction of each
                axis. For example, (2, 3), the first 2 means will add 2 at the
                beginning and also 2 at the end for its axis.and if a int is
                accepted, the kernel size will be initiated as (int, int)
            stride (int or tuple): stride, the logic is the same as kernel size.
            padding (int): tuple, list or None, padding, the logic is the same
                as kernel size. However, if you set pad_mode as "SAME_UPPER" or
                "SAME_LOWER" mode, you can set padding as None, and the padding
                will be computed automatically.
            odd_padding (tuple of four int): the odd paddding is the value
                that cannot be handled by the tuple padding (w, h) mode so
                it needs to firstly handle the input, then use the normal
                padding method.
        """
        super(AvgPool2d, self).__init__(kernel_size, stride, padding, False,
                                        odd_padding)


class MaxPool1d(Pooling2d):
    """
    Generate a Max Pooling 1d operator
    """

    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 odd_padding=(0, 0, 0, 0)):
        """
        Args:
            kernel_size (int or tuple): kernel size for two direction of each
                axis. For example, (2, 3), the first 2 means will add 2 at the
                beginning and also 2 at the end for its axis.and if a int is
                accepted, the kernel size will be initiated as (int, int)
            stride (int or tuple): stride, the logic is the same as kernel size.
            padding (int): tuple, list or None, padding, the logic is the same
                as kernel size. However, if you set pad_mode as "SAME_UPPER" or
                "SAME_LOWER" mode, you can set padding as None, and the padding
                will be computed automatically.
            odd_padding (tuple of four int): the odd paddding is the value
                that cannot be handled by the tuple padding (w, h) mode so
                it needs to firstly handle the input, then use the normal
                padding method.
        """
        if stride is None:
            stride = kernel_size
        super(MaxPool1d, self).__init__((1, kernel_size), (1, stride),
                                        (0, padding), True, odd_padding)


class AvgPool1d(Pooling2d):
    """
    Generate a Avg Pooling 1d operator
    """

    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 odd_padding=(0, 0, 0, 0)):
        """
        Args:
            kernel_size (int or tuple): kernel size for two direction of each
                axis. For example, (2, 3), the first 2 means will add 2 at the
                beginning and also 2 at the end for its axis.and if a int is
                accepted, the kernel size will be initiated as (int, int)
            stride (int or tuple): stride, the logic is the same as kernel size.
            padding (int): tuple, list or None, padding, the logic is the same
                as kernel size. However, if you set pad_mode as "SAME_UPPER" or
                "SAME_LOWER" mode, you can set padding as None, and the padding
                will be computed automatically.
            odd_padding (tuple of four int): the odd paddding is the value
                that cannot be handled by the tuple padding (w, h) mode so
                it needs to firstly handle the input, then use the normal
                padding method.
        """
        if stride is None:
            stride = kernel_size
        super(AvgPool1d, self).__init__((1, kernel_size), (1, stride),
                                        (0, padding), False, odd_padding)


class RNN_Base(Layer):

    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError

    def step_forward(self,
                     x=None,
                     h=None,
                     c=None,
                     Wx=None,
                     Wh=None,
                     Bx=None,
                     Bh=None,
                     b=None):
        raise NotImplementedError


class RNN(RNN_Base):
    """
    Generate a RNN operator
    """

    def initialize(self, xs):
        prev_state = xs[0].device.graph_enabled()
        xs[0].device.EnableGraph(False)

        Wx_shape = (self.input_size, self.hidden_size)
        self.Wx = Tensor(shape=Wx_shape, requires_grad=True, stores_grad=True)
        self.Wx.gaussian(0.0, 1.0)
        self.Wx.name = self.name + "Wx"

        Wh_shape = (self.hidden_size, self.hidden_size)
        self.Wh = Tensor(shape=Wh_shape, requires_grad=True, stores_grad=True)
        self.Wh.gaussian(0.0, 1.0)
        self.Wh.name = self.name + "Wh"

        B_shape = (self.hidden_size,)
        self.b = Tensor(shape=B_shape, requires_grad=True, stores_grad=True)
        self.b.set_value(0.0)
        self.b.name = self.name + "b"

        self.params = (self.Wx, self.Wh, self.b)
        self.states = self.params

        xs[0].device.EnableGraph(prev_state)

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        nonlinearity="tanh",
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
    ):
        """
        Args:
            input_size (int):  The number of expected features in the input x
            hidden_size (int): The number of features in the hidden state h
            num_layers (int):  Number of recurrent layers. Default: 1
            nonlinearity (string): The non-linearity to use. Default: 'tanh'
            bias (bool):  If False, then the layer does not use bias weights.
                Default: True
            batch_first (bool):  If True, then the input and output tensors
                are provided as (batch, seq, feature). Default: False
            dropout (float): If non-zero, introduces a Dropout layer on the
                outputs of each RNN layer except the last layer, with dropout
                probability equal to dropout. Default: 0
            bidirectional (bool): If True, becomes a bidirectional RNN.
                Default: False
        """
        self.has_initialized = False

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

    def __call__(self, xs, h0):
        if not self.has_initialized:
            self.initialize(xs)
            self.has_initialized = True

        # xs: a tuple or list of input tensors
        if not isinstance(xs, tuple):
            xs = tuple(xs)
        inputs = xs + (h0,)
        self.device_check(*inputs)
        # self.device_check(inputs[0], *self.params)
        self.device_check(inputs[0], self.Wx, self.Wh, self.b)
        batchsize = xs[0].shape[0]
        out = []
        h = self.step_forward(xs[0], h0, self.Wx, self.Wh, self.b)
        out.append(h)
        for x in xs[1:]:
            assert x.shape[0] == batchsize
            h = self.step_forward(x, h, self.Wx, self.Wh, self.b)
            out.append(h)
        return out, h

    def step_forward(self, x, h, Wx, Wh, b):
        y2 = autograd.matmul(h, Wh)
        y1 = autograd.matmul(x, Wx)
        y = autograd.add(y2, y1)
        y = autograd.add_bias(y, b, axis=0)
        if self.nonlinearity == "tanh":
            y = autograd.tanh(y)
        elif self.nonlinearity == "relu":
            y = autograd.relu(y)
        else:
            raise ValueError
        return y


class LSTM(RNN_Base):
    """
    Generate a LSTM operator
    """
    def initialize(self, xs):
        prev_state = xs[0].device.graph_enabled()
        xs[0].device.EnableGraph(False)

        # 1. Wx_i input,  Bx_i
        # 2. Wx_f forget, Bx_f
        # 3. Wx_o output, Bx_o
        # 4. Wx_g candidate, Bx_g
        Wx_shape = (self.input_size, self.hidden_size)
        self.Wx_i = Tensor(shape=Wx_shape, requires_grad=True, stores_grad=True)
        self.Wx_f = Tensor(shape=Wx_shape, requires_grad=True, stores_grad=True)
        self.Wx_o = Tensor(shape=Wx_shape, requires_grad=True, stores_grad=True)
        self.Wx_g = Tensor(shape=Wx_shape, requires_grad=True, stores_grad=True)

        Wh_shape = (self.hidden_size, self.hidden_size)
        self.Wh_i = Tensor(shape=Wh_shape, requires_grad=True, stores_grad=True)
        self.Wh_f = Tensor(shape=Wh_shape, requires_grad=True, stores_grad=True)
        self.Wh_o = Tensor(shape=Wh_shape, requires_grad=True, stores_grad=True)
        self.Wh_g = Tensor(shape=Wh_shape, requires_grad=True, stores_grad=True)
        [ w.gaussian(0.0, 0.01) for w in [self.Wx_i, self.Wx_f, self.Wx_o, self.Wx_g, self.Wh_i, self.Wh_f, self.Wh_o, self.Wh_g] ]

        Bx_shape = (self.hidden_size,)
        self.Bx_i = Tensor(shape=Bx_shape, requires_grad=True, stores_grad=True)
        self.Bx_f = Tensor(shape=Bx_shape, requires_grad=True, stores_grad=True)
        self.Bx_o = Tensor(shape=Bx_shape, requires_grad=True, stores_grad=True)
        self.Bx_g = Tensor(shape=Bx_shape, requires_grad=True, stores_grad=True)
        self.Bh_i = Tensor(shape=Bx_shape, requires_grad=True, stores_grad=True)
        self.Bh_f = Tensor(shape=Bx_shape, requires_grad=True, stores_grad=True)
        self.Bh_o = Tensor(shape=Bx_shape, requires_grad=True, stores_grad=True)
        self.Bh_g = Tensor(shape=Bx_shape, requires_grad=True, stores_grad=True)
        [ b.set_value(0.0) for b in [self.Bx_i, self.Bx_f, self.Bx_o, self.Bx_g, self.Bh_i, self.Bh_f, self.Bh_o, self.Bh_g] ]

        self.Wx_i.name = self.name + "Wx_i"
        self.Wx_f.name = self.name + "Wx_f"
        self.Wx_o.name = self.name + "Wx_o"
        self.Wx_g.name = self.name + "Wx_g"
        self.Wh_i.name = self.name + "Wh_i"
        self.Wh_f.name = self.name + "Wh_f"
        self.Wh_o.name = self.name + "Wh_o"
        self.Wh_g.name = self.name + "Wh_g"
        self.Bx_i.name = self.name + "Bx_i"
        self.Bx_f.name = self.name + "Bx_f"
        self.Bx_o.name = self.name + "Bx_o"
        self.Bx_g.name = self.name + "Bx_g"
        self.Bh_i.name = self.name + "Bh_i"
        self.Bh_f.name = self.name + "Bh_f"
        self.Bh_o.name = self.name + "Bh_o"
        self.Bh_g.name = self.name + "Bh_g"

        self.params = [self.Wx_i , self.Wx_f , self.Wx_o , self.Wx_g , self.Wh_i , self.Wh_f , self.Wh_o , self.Wh_g , self.Bx_i , self.Bx_f , self.Bx_o , self.Bx_g , self.Bh_i , self.Bh_f , self.Bh_o , self.Bh_g]
        self.states = self.params

        xs[0].device.EnableGraph(prev_state)


    def __init__(
        self,
        input_size,
        hidden_size,
        nonlinearity="tanh",
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
    ):
        """
        Args:
            input_size (int):  The number of expected features in the input x
            hidden_size (int): The number of features in the hidden state h
            num_layers (int):  Number of recurrent layers. Default: 1
            nonlinearity (string): The non-linearity to use. Default: 'tanh'
            bias (bool):  If False, then the layer does not use bias weights.
                Default: True
            batch_first (bool):  If True, then the input and output tensors
                are provided as (batch, seq, feature). Default: False
            dropout (float): If non-zero, introduces a Dropout layer on the
                outputs of each RNN layer except the last layer, with dropout
                probability equal to dropout. Default: 0
            bidirectional (bool): If True, becomes a bidirectional RNN.
                Default: False
        """
        self.has_initialized = False

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

    def __call__(self, xs, h0_c0):
        if not self.has_initialized:
            self.initialize(xs)
            self.has_initialized = True

        # xs: a tuple or list of input tensors
        # h0_c0: a tuple of (h0, c0)
        h0, c0 = h0_c0
        if not isinstance(xs, list):
            xs = list(xs)
        inputs = xs + list((h0, c0))
        self.device_check(*inputs)
        self.device_check(inputs[0], *self.params)
        batchsize = xs[0].shape[0]
        out = []
        h, c = self.step_forward(xs[0], h0, c0)
        out.append(h)
        for x in xs[1:]:
            assert x.shape[0] == batchsize
            h, c = self.step_forward(x, h, c)
            out.append(h)
        return out, h, c

    def step_forward(self, x, h, c):
        # input
        y1 = autograd.matmul(x, self.Wx_i)
        y1 = autograd.add_bias(y1, self.Bx_i, axis=0)
        y2 = autograd.matmul(h, self.Wh_i)
        y2 = autograd.add_bias(y2, self.Bh_i, axis=0)
        i = autograd.add(y1, y2)
        i = autograd.sigmoid(i)

        # forget
        y1 = autograd.matmul(x, self.Wx_f)
        y1 = autograd.add_bias(y1, self.Bx_f, axis=0)
        y2 = autograd.matmul(h, self.Wh_f)
        y2 = autograd.add_bias(y2, self.Bh_f, axis=0)
        f = autograd.add(y1, y2)
        f = autograd.sigmoid(f)

        # output
        y1 = autograd.matmul(x, self.Wx_o)
        y1 = autograd.add_bias(y1, self.Bx_o, axis=0)
        y2 = autograd.matmul(h, self.Wh_o)
        y2 = autograd.add_bias(y2, self.Bh_o, axis=0)
        o = autograd.add(y1, y2)
        o = autograd.sigmoid(o)

        y1 = autograd.matmul(x, self.Wx_g)
        y1 = autograd.add_bias(y1, self.Bx_g, axis=0)
        y2 = autograd.matmul(h, self.Wh_g)
        y2 = autograd.add_bias(y2, self.Bh_g, axis=0)
        g = autograd.add(y1, y2)
        g = autograd.tanh(g)

        cout1 = autograd.mul(f, c)
        cout2 = autograd.mul(i, g)
        cout = autograd.add(cout1, cout2)

        hout = autograd.tanh(cout)
        hout = autograd.mul(o, hout)
        return hout, cout

''' import autograd at the end to resolve circular import
'''
from singa import autograd
