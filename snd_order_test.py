from typing import Callable, Tuple, Type
import torch
import torch.nn as nn
import torch.nn.modules.activation as activation
import torch.autograd as autograd

from torch import Tensor
from torch.nn import Module


def is_activation(module: Module) -> bool:
    # This is not a good test.
    exceptions = [nn.Softmax, nn.Softmax2d, nn.Softmin]
    if any(isinstance(module, Ex) for Ex in exceptions):
        return False
    return type(module).__module__ == 'torch.nn.modules.activation'


def snd_order(activation: Module,
              data: Tensor,
              grad_outputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    assert is_activation(activation), "Function must be applied element-wise"
    data.requires_grad = True
    f_x = activation(data)
    df_dx, = autograd.grad(f_x, data,
                           grad_outputs=grad_outputs,
                           create_graph=True, retain_graph=True)
    d2f_dx2, = autograd.grad(df_dx, data,
                             grad_outputs=torch.ones_like(df_dx),
                             create_graph=True, retain_graph=False)
    return f_x, df_dx, d2f_dx2


def test_activation(Activation: Type,
                    f: Callable[[Tensor], Tensor],
                    df: Callable[[Tensor], Tensor],
                    d2f: Callable[[Tensor], Tensor]) -> None:
    batch_size, v_size = 15, 13
    activation = Activation()
    data = torch.randn(batch_size, v_size)
    grad_outputs = torch.randn(batch_size, v_size)

    f_x, df_dx, d2f_dx2 = snd_order(activation, data, grad_outputs)
    test_f_x = f(data)
    test_df_dx = df(data) * grad_outputs
    test_d2f_dx2 = d2f(data) * grad_outputs

    assert torch.allclose(f_x, test_f_x)
    assert torch.allclose(df_dx, test_df_dx)
    assert torch.allclose(d2f_dx2, test_d2f_dx2)


def test_sigmoid() -> None:
    def f(x: Tensor) -> Tensor:
        return 1. / (1. + torch.exp(-x))

    def df(x: Tensor) -> Tensor:
        f_x = f(x)
        return f_x * (1 - f_x)

    def d2f(x: Tensor) -> Tensor:
        f_x = f(x)
        return f_x * (1 - f_x) * (1 - 2 * f_x)

    test_activation(nn.Sigmoid, f, df, d2f)


def test_tanh() -> None:
    def f(x: Tensor) -> Tensor:
        e2x = torch.exp(2 * x)
        return (e2x - 1) / (e2x + 1)

    def df(x: Tensor) -> Tensor:
        f_x = f(x)
        return 1 - f_x * f_x

    def d2f(x: Tensor) -> Tensor:
        f_x = f(x)
        return 2 * (f_x * f_x - 1) * f_x

    test_activation(nn.Tanh, f, df, d2f)


def test_relu() -> None:
    def f(x: Tensor) -> Tensor:
        return torch.clamp(x, min=0)

    def df(x: Tensor) -> Tensor:
        return (x > 0).float()

    def d2f(x: Tensor) -> Tensor:
        return torch.zeros_like(x)

    test_activation(nn.ReLU, f, df, d2f)


def main():
    test_sigmoid()
    test_tanh()
    test_relu()


if __name__ == "__main__":
    main()
