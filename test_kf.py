from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as functional

from torch import Tensor
from torch.nn import Module

from kf import KroneckerFactored


class MLP(KroneckerFactored):

    def __init__(self, units_no: List[int], **kwargs) -> None:
        super(MLP, self).__init__(**kwargs)
        self.my_modules: List[Module] = []
        self.depth = depth = len(units_no) - 1

        for idx in range(1, depth):
            linear_layer = nn.Linear(*units_no[idx-1:idx+1])
            setattr(self, f"linear_{idx:d}", linear_layer)
            relu_layer = nn.ReLU()
            setattr(self, f"relu_{idx:d}", relu_layer)
            self.my_modules.extend([linear_layer, relu_layer])

        output_layer = nn.Linear(*units_no[-2:])
        setattr(self, f"linear_{depth:d}", output_layer)
        self.my_modules.append(output_layer)

    def forward(self, x: Tensor) -> Tensor:
        for module in self.my_modules:
            x = module(x)
        return x


class LeNet(KroneckerFactored):

    def __init__(self,
                 filters_no: List[int],
                 filter_size: List[Union[int, Tuple[int, int]]],
                 stride: List[Union[int, Tuple[int, int]]],
                 units_no: List[int],
                 **kwargs) -> None:
        super(LeNet, self).__init__(**kwargs)

        if len(filters_no) != len(filter_size) + 1 or\
           len(filters_no) != len(stride) + 1:
            raise ValueError

        self.conv_modules: List[Module] = []
        self.fc_modules: List[Module] = []

        self.conv_depth = conv_depth = len(filters_no) - 1
        self.fc_depth = fc_depth = len(units_no) - 1
        self.depth = self.conv_depth + self.fc_depth

        for idx in range(1, conv_depth + 1):
            conv_layer = nn.Conv2d(*filters_no[idx-1:idx+1],
                                   kernel_size=filter_size[idx-1],
                                   stride=stride[idx-1])
            setattr(self, f"conv_{idx:d}", conv_layer)
            relu_layer = nn.ReLU()
            setattr(self, f"relu_{idx:d}", relu_layer)
            self.conv_modules.extend([conv_layer, relu_layer])

        for idx in range(1, fc_depth):
            linear_layer = nn.Linear(*units_no[idx-1:idx+1])
            setattr(self, f"linear_{idx+conv_depth:d}", linear_layer)
            relu_layer = nn.ReLU()
            setattr(self, f"relu_{idx+conv_depth:d}", relu_layer)
            self.fc_modules.extend([linear_layer, relu_layer])

        output_layer = nn.Linear(*units_no[-2:])
        setattr(self, f"linear_{conv_depth+fc_depth:d}", output_layer)
        self.fc_modules.append(output_layer)

    def forward(self, x: Tensor) -> Tensor:
        for module in self.conv_modules:
            x = module(x)
        x = x.view(x.size(0), -1)
        for module in self.fc_modules:
            x = module(x)
        return x


def test_mlp():
    size = [7, 5, 11, 13, 17, 3]
    batch_size = 10
    mlp = MLP(size, average_factors=False, verbose=True)
    mlp.do_kf = True

    for _idx in range(10):
        data = torch.randn(batch_size, size[0], requires_grad=True)
        target = torch.randn(batch_size, size[-1])
        output = mlp(data)
        loss = functional.mse_loss(output, target)
        loss.backward()
    kfhp = mlp.end_kf()
    dummy_vector = dict({})
    for name, param in mlp.named_parameters():
        dummy_vector[name] = torch.randn(param.size())
    print(kfhp.hessian_product_loss(dummy_vector).item())

    mlp.do_kf = True
    mlp.average_factors = True
    for _idx in range(10):
        data = torch.randn(batch_size, size[0], requires_grad=True)
        target = torch.randn(batch_size, size[-1])
        output = mlp(data)
        loss = functional.mse_loss(output, target)
        loss.backward()
    kfhp = mlp.end_kf()
    dummy_vector = dict({})
    for name, param in mlp.named_parameters():
        dummy_vector[name] = torch.randn(param.size())

    print(kfhp.hessian_product_loss(dummy_vector).item())


def test_simple():
    print("SIMPLE")
    s_net = LeNet([1, 2], [(3, 3)], [(1, 1)], [18, 1])
    s_net.zero_grad()
    s_net.do_kf = True
    data = torch.randn(1, 1, 5, 5, requires_grad=True)
    output = s_net(data)
    output.backward()
    kfhp = s_net.end_kf()
    dummy_vector = dict({})
    for name, param in s_net.named_parameters():
        dummy_vector[name] = torch.randn(param.size())
    print(kfhp.hessian_product_loss(dummy_vector).item())


def test_lenet():
    print("LENET")
    filters_no = [3, 2, 3, 2]
    filter_size = [(4, 7), (3, 5), (5, 3)]
    stride = [(1, 1), (2, 3), (1, 1)]
    size = [7, 5, 11, 13, 17, 3]
    in_size = (21, 23)
    h, w = in_size
    for ((k_h, k_w), (s_h, s_w)) in zip(filter_size, stride):
        h, w = (h - k_h + 0) // s_h + 1, (w - k_w + 0) // s_w + 1
    size = [h * w * filters_no[-1]] + size
    batch_size = 10
    mlp = LeNet(filters_no, filter_size, stride, size, average_factors=False)

    mlp.do_kf = True

    for _idx in range(10):
        data = torch.randn(batch_size, filters_no[0], *in_size, requires_grad=True)
        target = torch.randn(batch_size, size[-1])
        output = mlp(data)
        loss = functional.mse_loss(output, target)
        loss.backward()
    kfhp = mlp.end_kf()
    dummy_vector = dict({})
    for name, param in mlp.named_parameters():
        dummy_vector[name] = torch.randn(param.size())

    print(kfhp.hessian_product_loss(dummy_vector).item())

    mlp.average_factors = True
    mlp.do_kf = True

    for _idx in range(10):
        data = torch.randn(batch_size, filters_no[0], *in_size, requires_grad=True)
        target = torch.randn(batch_size, size[-1])
        output = mlp(data)
        loss = functional.mse_loss(output, target)
        loss.backward()
    kfhp = mlp.end_kf()
    dummy_vector = dict({})
    for name, param in mlp.named_parameters():
        dummy_vector[name] = torch.randn(param.size())

    print(kfhp.hessian_product_loss(dummy_vector).item())


def main():
    test_mlp()
    test_simple()
    test_lenet()


if __name__ == "__main__":
    print("Running torch version", torch.__version__)
    main()
