from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as functional
# import torch.autograd as autograd

from torch import Tensor
from torch.nn import Module


class ArchitectureNotSupported(Exception):
    pass


def is_parametric(module: nn.Module) -> bool:
    # TODO: support other types of layers (e.g. conv)
    return isinstance(module, nn.Linear)


def is_activation(module: nn.Module) -> bool:
    # TODO: make this smarter
    return isinstance(module, nn.ReLU)


class KroneckerFactored(nn.Module):

    ACTIVATION = 1
    PARAMETRIC = 2

    FORWARD = 1
    BACKWARD = 2
    DONE = 3

    def __init__(self) -> None:
        super(KroneckerFactored, self).__init__()
        self.__my_handles = []
        self.__kf_mode = False
        self.__do_check = True
        self.__verbose = True
        self.__reset_state()

    @property
    def do_kf(self) -> None:
        return self.__kf_mode

    @do_kf.setter
    def do_kf(self, value: bool) -> None:
        if (value and self.__kf_mode) or (not value and not self.__kf_mode):
            return
        self.__kf_mode = value
        if not value:
            self.__drop_hooks()
        else:
            self.__set_hooks()
            self.__reset_state()

    def __set_hooks(self):
        # Use forward_pre_hooks to check stuff; drop them for speed
        for module in self.modules():
            if self.__do_check:
                self.__my_handles.append(
                    module.register_forward_pre_hook(self._kf_pre_hook)
                )
            self.__my_handles.extend([
                module.register_forward_hook(self._kf_fwd_hook),
                module.register_backward_hook(self._kf_bwd_hook)
            ])

    def __drop_hooks(self):
        for handle in self.__my_handles:
            handle.remove()
        self.__my_handles.clear()

    def __reset_state(self):
        """This should be called whenever a new hessian is needed"""
        self.__soft_reset_state()

    def __soft_reset_state(self):
        """This should be called before each batch"""
        self.__prev_layer = None
        self.__prev_layer_name = ""
        self.__phase = self.FORWARD
        self.__layer_idx = 0

    def _kf_pre_hook(self, module, _inputs):
        """This hook only checks the architecture"""
        prev_layer = self.__prev_layer
        prev_name = self.__prev_layer_name
        crt_name = module._get_name()
        msg = f"{prev_name:s} -> {crt_name:s}"

        if self.__verbose:
            print(f"[{self.__layer_idx:d}] {crt_name:s} before FWD")

        if isinstance(module, KroneckerFactored):
            self.__soft_reset_state()
        elif is_parametric(module):
            if not (prev_layer is None or prev_layer == self.ACTIVATION):
                raise ArchitectureNotSupported(msg)
            self.__prev_layer = self.PARAMETRIC
        elif is_activation(module):
            if prev_layer != self.PARAMETRIC:
                raise ArchitectureNotSupported(msg)
            self.__prev_layer = self.ACTIVATION
        else:
            raise ArchitectureNotSupported(msg)
        self.__prev_layer_name = crt_name

    def _kf_fwd_hook(self, module, inputs, output) -> None:
        if self.__verbose:
            crt_name = module._get_name()
            print(f"[{self.__layer_idx:d}] {crt_name:s} after FWD")

        if isinstance(module, KroneckerFactored):
            self.__fwd_hook(module, inputs, output)
            return
        if isinstance(module, nn.Linear):
            self.__linear_fwd_hook(module, inputs, output)
        elif isinstance(module, nn.ReLU):
            self.__relu_fwd_hook(module, inputs, output)
        else:
            raise ArchitectureNotSupported("You shouldn't be here!")
        self.__layer_idx += 1

    def _kf_bwd_hook(self, module, grad_input, grad_output) -> None:
        if self.__verbose:
            crt_name = module._get_name()
            print(f"[{self.__layer_idx:d}] {crt_name:s} BWD")

        if isinstance(module, KroneckerFactored):
            self.__bwd_hook(module, grad_input, grad_output)
            return
        if isinstance(module, nn.Linear):
            self.__linear_bwd_hook(module, grad_input, grad_output)
        elif isinstance(module, nn.ReLU):
            self.__relu_bwd_hook(module, grad_input, grad_output)
        else:
            raise ArchitectureNotSupported("You shouldn't be here")
        self.__layer_idx -= 1
        if self.__layer_idx < 0:
            self.__phase = self.DONE
            if self.__verbose:
                print("Done with this batch!")

    # Magic happens below

    def __fwd_hook(self, _module, _inputs, _output):
        self.__phase = self.BACKWARD
        self.__layer_idx -= 1

    def __bwd_hook(self, _module, grad_inputs, _grad_output):
        # DO NOT USE THIS! Seems to be a bug in pytorch, this hook is
        # called before ending backpropgation.
        pass

    def __linear_fwd_hook(self, module, inputs, output):
        assert self.__phase == self.FORWARD
        pass

    def __relu_fwd_hook(self, module, inputs, output):
        assert self.__phase == self.FORWARD
        pass

    def __linear_bwd_hook(self, module, grad_input, grad_output):
        assert self.__phase == self.BACKWARD
        pass

    def __relu_bwd_hook(self, module, grad_input, grad_output):
        assert self.__phase == self.BACKWARD
        pass


class MLP(KroneckerFactored):

    def __init__(self, units_no: List[int]) -> None:
        super(MLP, self).__init__()
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

        print([type(m) for m in self.my_modules])

    def forward(self, x: Tensor) -> Tensor:
        for module in self.my_modules:
            x = module(x)
        return x


def main():
    mlp = MLP([10, 10, 10, 10, 10])
    mlp(torch.randn(1, 10))
    mlp.do_kf = True
    loss = functional.mse_loss(mlp(torch.randn(1, 10)), torch.randn(1, 10))
    loss.backward()


if __name__ == "__main__":
    print("Running torch version", torch.__version__)
    main()
