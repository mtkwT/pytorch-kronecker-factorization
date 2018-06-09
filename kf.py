from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.autograd as autograd

from torch import Tensor
from torch.nn import Module

# This module implements KF block approximation for the hessian
# http://proceedings.mlr.press/v70/botev17a/botev17a.pdf

# Pre-activation hessians are kept separately for each sample in the
# batch. This is needed as I could not find a better solution for
# diag(f'(z[l])) * W[l+1]^T * H^(l+1) * W(l+1) * diag(f'(z[l])) when
# batches of examples are passed through the network.


# TODO: Add Fisher factorization for convolutional layers
# (https://arxiv.org/pdf/1602.01407.pdf). A KF block approximation
# like the one for the linear case seems impractical for now.


class ArchitectureNotSupported(Exception):
    pass


def is_parametric(module: Module) -> bool:
    # TODO: support other types of layers (e.g. conv)
    return isinstance(module, nn.Linear)


def is_activation(module: Module) -> bool:
    # TODO: improve this (it sucks now!)
    not_elmentwise = [nn.Softmax, nn.Softmax2d, nn.Softmin]
    if any(isinstance(module, Activation) for Activation in not_elmentwise):
        return False
    return type(module).__module__ == 'torch.nn.modules.activation'


class KFHessianProduct(object):

    def __init__(self, inputs_cov, preactivations):
        self.inputs_cov = inputs_cov
        self.preactivations = preactivations
        for value in inputs_cov.values():
            self._single_product = not isinstance(value, list)
            break

    def __compute_product(self, module_name, weight=None, bias=None):
        print(type(self.inputs_cov[module_name][0]))
        print(type(self.preactivations[module_name][0]))
        params = torch.cat([weight, bias.unsqueeze(1)], 1)
        if self._single_product:
            return torch.matmul(torch.matmul(self.preactivations[module_name],
                                             params),
                                self.inputs_cov[module_name]).sum()
        else:
            loss = 0
            zipped = zip(self.preactivations[module_name], self.inputs_cov[module_name])
            n_no = 1
            for preact, incov in zipped:
                loss += torch.matmul(torch.matmul(preact, params), incov).sum()
                n_no += 1
            return loss / float(n_no)

    def hessian_product_loss(self, vector: Tensor) -> None:
        per_layer = dict({})
        for param_name, values in vector.items():
            module_name, param_name = param_name.split(".")
            per_layer.setdefault(module_name, dict({}))[param_name] = values
        loss = None
        for module_name, params in per_layer.items():
            layer_loss = self.__compute_product(module_name, **params)
            loss = layer_loss if loss is None else (layer_loss + loss)
        return loss


class KroneckerFactored(nn.Module):

    ACTIVATION = 1
    PARAMETRIC = 2

    FORWARD = 1
    BACKWARD = 2
    DONE = 3

    def __init__(self,
                 do_checks: bool=True,
                 verbose: bool=True,
                 average_factors: bool=True) -> None:
        super(KroneckerFactored, self).__init__()
        self.__my_handles = []
        self.__kf_mode = False  # One must activate this
        self.__do_checks = do_checks
        self.__verbose = verbose
        self.__average_factors = average_factors

        self.__preactivations = None
        self.__output_hessian = None

    @property
    def verbose(self) -> bool:
        return self.__verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        self.__verbose = value

    @property
    def do_kf(self) -> bool:
        return self.__kf_mode

    @do_kf.setter
    def do_kf(self, value: bool) -> None:
        if (value and self.__kf_mode) or (not value and not self.__kf_mode):
            return
        self.__kf_mode = value
        if not value:
            self.__reset_state()
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
        self.__output_hessian = None
        self.__df_dx = dict({})
        self.__d2f_dx2 = dict({})
        self.__batches_no = 0
        self.__module_names = dict({})
        self.__match_parameters()

    def __soft_reset_state(self):
        """This should be called before each batch"""
        self.__prev_layer = None
        self.__prev_layer_name = ""
        self.__phase = self.FORWARD
        self.__layer_idx = 0
        self.__last_linear = -1
        self.__next_parametric = None  # type: Optional[Module]
        self.__batch_preactivations = dict({})

    def __match_parameters(self):
        for module_name, module in self.named_modules():
            self.__module_names[id(module)] = module_name

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
        elif is_activation(module):
            self.__activation_fwd_hook(module, inputs, output)
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
            self.__next_parametric = module
        elif is_activation(module):
            self.__activation_bwd_hook(module, grad_input, grad_output)
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
        inputs = torch.cat([inputs, torch.ones_like(inputs[:, 0:1])], dim=1)
        inputs_cov = torch.matmul(inputs.t(), inputs)
        inputs_cov /= float(inputs.size(0))

        inputs_cov.detach_()
        layer_idx = self.__layer_idx
        module_name = self.__module_names[id(module)]
        if module_name not in self.__inputs_cov:
            if self.__average_factors:
                self.__inputs_cov[module_name] = inputs_cov
            else:
                self.__inputs_cov[module_name] = [inputs_cov]
        else:
            if self.__average_factors:
                self.__inputs_cov[module_name] += inputs_cov
            else:
                self.__inputs_cov[module_name].append(inputs_cov)

        # Assume this is the last layer
        self.__last_linear = layer_idx
        batch_size, out_no = output.size()
        self.__expected_output_hessian_size = torch.Size([batch_size, out_no, out_no])

    def __linear_bwd_hook(self, module, grad_input, grad_output):
        assert self.__phase == self.BACKWARD

        layer_idx = self.__layer_idx

        if layer_idx == self.__last_linear:
            if self.__output_hessian is None:
                assert self.__next_parametric is None
                assert len(grad_output) == 1
                grad_output = grad_output[0]
                assert len(grad_output.size()) == 2
                act = torch.bmm(grad_output.unsqueeze(2), grad_output.unsqueeze(1))
                act.detach_()
            else:
                act = self.__output_hessian.detach()
        else:
            print(len(grad_input))
            grad_input = grad_input[1]  # This seems to be the g_input
            grad_output = grad_output[0]

            next_pa = self.__batch_preactivations[layer_idx + 2]  # size: b * n_out * n_out
            next_w = self.__next_parametric.weight
            df_dx = self.__df_dx[layer_idx + 1]

            b_sz, next_out_no, _ = next_pa.size()
            _next_out_no, crt_out_no = next_w.size()

            assert df_dx.size() == torch.Size([b_sz, crt_out_no])
            left_diag = df_dx.unsqueeze(2).expand(b_sz, crt_out_no, next_out_no)
            left_w = next_w.t().unsqueeze(0).expand(b_sz, crt_out_no, next_out_no)
            left = torch.mul(left_diag, left_w)
            act = torch.matmul(left, next_pa)
            right = left.transpose(1, 2)
            act = torch.matmul(act, right)
            snd_order = self.__d2f_dx2[layer_idx + 1]
            for i in range(act.size(0)):
                act[i] += torch.diag(snd_order[i])
            act.detach_()
            del self.__batch_preactivations[layer_idx + 2]

        self.__batch_preactivations[layer_idx] = act
        module_name = self.__module_names[id(module)]
        if self.__average_factors:
            if module_name in self.__preactivations:
                self.__preactivations[module_name] += act.mean(0)
            else:
                self.__preactivations[module_name] = act.mean(0)
        else:
            self.__preactivations.setdefault(module_name, []).append(act.mean(0))

        if self.__layer_idx == 0:
            self.__batches_no += 1

    def __activation_fwd_hook(self, _module, inputs, output):
        layer_idx = self.__layer_idx
        inputs, = inputs
        df_dx, = autograd.grad(output, inputs,
                               grad_outputs=torch.ones_like(inputs),
                               create_graph=True, retain_graph=True)
        d2f_dx2, = autograd.grad(df_dx, inputs,
                                 grad_outputs=torch.ones_like(df_dx),
                                 create_graph=True, retain_graph=False)
        self.__df_dx[layer_idx] = df_dx.detach()
        self.__d2f_dx2[layer_idx] = d2f_dx2.detach()

    def __activation_bwd_hook(self, _module, _grad_input, grad_output):
        layer_idx = self.__layer_idx
        assert layer_idx in self.__df_dx and layer_idx in self.__d2f_dx2
        grad_output, = grad_output
        print(self.__df_dx[layer_idx].size())
        print(self.__d2f_dx2[layer_idx].size())
        self.__d2f_dx2[layer_idx] *= grad_output.detach()

    def end_kf(self):
        if self.__average_factors:
            coeff = 1.0 / float(self.__batches_no)
            for values in self.__inputs_cov.values():
                values.mul_(coeff)
            for values in self.__preactivations.values():
                values.mul_(coeff)
        kfhp = KFHessianProduct(self.__inputs_cov, self.__preactivations)
        self.do_kf = False
        return kfhp


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

        print([type(m) for m in self.my_modules])

    def forward(self, x: Tensor) -> Tensor:
        for module in self.my_modules:
            x = module(x)
        return x


def main():
    size = [7, 5, 11, 13, 17, 3]
    batch_size = 10
    mlp = MLP(size, average_factors=False)
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

    print(kfhp.hessian_product_loss(dummy_vector))


if __name__ == "__main__":
    print("Running torch version", torch.__version__)
    main()
