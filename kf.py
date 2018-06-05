from typing import List, Optional, Tuple

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

    def __init__(self,
                 do_checks: bool = True,
                 verbose: bool = True,
                 average_factors: bool = True) -> None:
        super(KroneckerFactored, self).__init__()
        self.__my_handles = []
        self.__kf_mode = False  # One must activate this
        self.__do_checks = do_checks
        self.__verbose = verbose
        self.__average_factors = average_factors
        self.__reset_state()

    @property
    def do_kf(self) -> bool:
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

    @property
    def output_hessian(self) -> Optional[Tensor]:
        return self.__output_hessian

    @output_hessian.setter
    def output_hessian(self, value: Tensor) -> None:
        if not torch.is_tensor(value) or \
           value.size() != self.__expected_output_hessian_size:
            raise ValueError
        if self.__phase != self.BACKWARD:
            raise Exception("Bad time to set the output_hessian")
        self.__output_hessian = value

    def __set_hooks(self):
        # Use forward_pre_hooks to check stuff; drop them for speed
        for module in self.modules():
            if self.__do_checks:
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
        self.__preactivations = dict({})
        self.__inputs_cov = dict({})
        self.__weights = dict({})
        self.__biases = dict({})
        self.__fst_order_transfer = dict({})
        self.__snd_order_transfer = dict({})
        self.__output_hessian = None

    def __soft_reset_state(self):
        """This should be called before each batch"""
        self.__prev_layer = None
        self.__prev_layer_name = ""
        self.__phase = self.FORWARD
        self.__layer_idx = 0
        self.__last_linear = -1

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

        # Extract inputs
        assert len(inputs) == 1
        inputs = inputs[0]

        batch_size = inputs.size(0)
        inputs = inputs.view(batch_size, -1)
        inputs_cov = torch.mm(inputs.transpose(0, 1), inputs)
        inputs_cov.detach()
        layer_idx = self.__layer_idx
        if layer_idx not in self.__inputs_cov:
            if self.__average_factors:
                self.__inputs_cov[layer_idx] = inputs_cov
            else:
                self.__inputs_cov[layer_idx] = [inputs_cov]
        else:
            if self.__average_factors:
                self.__inputs_cov[layer_idx] += inputs_cov
            else:
                self.__inputs_cov[layer_idx].append(inputs_cov)

        self.__weights[layer_idx] = module.weight
        self.__biases[layer_idx] = module.bias
        self.__last_linear = layer_idx
        out_no = output.size(1)
        self.__expected_output_hessian_size = torch.Size([out_no, out_no])
        print(self.__expected_output_hessian_size)

    def __relu_fwd_hook(self, _module, inputs, _output):
        assert self.__phase == self.FORWARD
        grad = (inputs[0] > 0).type_as(inputs[0]).detach()
        self.__fst_order_transfer[self.__layer_idx] = grad
        self.__snd_order_transfer[self.__layer_idx] = None

    def __linear_bwd_hook(self, module, grad_input, grad_output):
        assert self.__phase == self.BACKWARD
        print(type(grad_input))
        print([type(t) for t in grad_input])
        print([(t.size() if t is not None else t) for t in grad_input])
        print(len(grad_input))

        layer_idx = self.__layer_idx
        if layer_idx == self.__last_linear:
            if self.__output_hessian is None:
                assert len(grad_output) == 1
                grad_output = grad_output[0]
                act = torch.mm(grad_output.transpose(0, 1), grad_output)
                self.__preactivations[layer_idx] = act.detach()
            else:
                self.__preactivations[layer_idx] = self.__output_hessian
        else:
            next_pa = self.__preactivations[layer_idx + 2]
            next_w = self.__weights[layer_idx + 2]
            act = torch.mm(torch.mm(next_w.transpose(0, 1), next_pa), next_w)

            fst_act = self.__fst_order_transfer[layer_idx + 1].view(-1)

            left_go = fst_act.unsqueeze(1).expand_as(act).detach()
            right_go = fst_act.unsqueeze(0).expand_as(act).detach()
            act = left_go * act * right_go
            snd_act = self.__snd_order_transfer[layer_idx + 1]

            if snd_act is not None:
                raise ArchitectureNotSupported("ReLU only, dude")

            self.__preactivations[layer_idx] = act.detach()

    def __relu_bwd_hook(self, module, grad_input, grad_output):
        assert self.__phase == self.BACKWARD

    def get_factors(self) -> Tuple[Tensor, Tensor]:
        if self.__average_factors:
            return (self.__inputs_cov / self.__batches_no,
                    self.__preactivations / self.__batches_no)
        else:
            return self.__inputs_cov, self.__preactivations


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
    size = [7, 5, 11, 13, 17, 3]
    batch_size = 1
    data = torch.randn(batch_size, size[0], requires_grad=True)
    target = torch.randn(batch_size, size[-1])
    mlp = MLP(size)
    mlp.do_kf = True
    output = mlp(data)
    loss = functional.mse_loss(output, target)
    loss.backward()


if __name__ == "__main__":
    print("Running torch version", torch.__version__)
    main()
