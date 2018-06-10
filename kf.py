from typing import List, Optional, Tuple, Union

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


# Add Fisher factorization for convolutional layers
# (https://arxiv.org/pdf/1602.01407.pdf). A KF block approximation
# like the one for the linear case seems impractical for now.


class ArchitectureNotSupported(Exception):
    pass


def is_convolutional(module: Module) -> bool:
    return isinstance(module, nn.Conv2d)


def is_linear(module: Module) -> bool:
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

    def __get_linear_params(self, weight=None, bias=None):
        return torch.cat([weight, bias.unsqueeze(1)], 1)

    def __get_conv_params(self, weight=None, bias=None):
        ch_out, ch_in, h_in, w_in = weight.size()
        return torch.cat([weight.view(ch_out, -1), bias.unsqueeze(1)], dim=1)

    def __compute_product(self, module_name, **kwargs):
        if module_name.startswith("linear"):
            params = self.__get_linear_params(**kwargs)
        elif module_name.startswith("conv"):
            params = self.__get_conv_params(**kwargs)
        else:
            raise NotImplemented

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
    LINEAR = 2
    CONVOLUTIONAL = 3
    OTHER = 4

    FORWARD = 1
    BACKWARD = 2
    DONE = 3

    def __init__(self,
                 do_checks: bool=True,
                 use_fisher: bool=True,
                 verbose: bool=True,
                 average_factors: bool=True) -> None:
        super(KroneckerFactored, self).__init__()
        self.__my_handles = []
        self.__kf_mode = False  # One must activate this
        self.__use_fisher = use_fisher
        self.__do_checks = do_checks
        self.__verbose = verbose
        self.__average_factors = average_factors

        self.__reset_state()
        self.__soft_reset_state()

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
        self.__prev_layer_name = "input"
        self.__phase = self.FORWARD
        self.__layer_idx = 0
        self.__last_linear = -1
        self.__next_parametric = None  # type: Optional[Module]
        self.__batch_preactivations = dict({})
        self.__conv_special_inputs = dict({})
        self.__maybe_exact = True

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

    def __match_parameters(self):
        for module_name, module in self.named_modules():
            self.__module_names[id(module)] = module_name

    def _kf_pre_hook(self, module, _inputs):
        """This hook only checks the architecture"""
        use_fisher = self.__use_fisher
        prev_layer = self.__prev_layer
        prev_name = self.__prev_layer_name
        crt_name = module._get_name()
        msg = f"Need Fisher for {prev_name:s} -> {crt_name:s}."

        if self.__verbose:
            print(f"[{self.__layer_idx:d}] {crt_name:s} before FWD")

        if isinstance(module, KroneckerFactored):
            self.__soft_reset_state()
        elif is_linear(module):
            if not (use_fisher or prev_layer is None or prev_layer == self.ACTIVATION):
                raise ArchitectureNotSupported(msg)
            self.__prev_layer = self.LINEAR
        elif is_activation(module):
            if not (use_fisher or prev_layer == self.LINEAR):
                raise ArchitectureNotSupported(msg)
            self.__prev_layer = self.ACTIVATION
        elif is_convolutional(module):
            if not use_fisher:
                raise ArchitectureNotSupported(msg)
            self.__prev_layer = self.CONVOLUTIONAL
        else:
            if not use_fisher:
                raise ArchitectureNotSupported(msg)
            self.__prev_layer = self.OTHER
        self.__prev_layer_name = crt_name

    def _kf_fwd_hook(self, module, inputs, output) -> None:
        if self.__verbose:
            crt_name = module._get_name()
            print(f"[{self.__layer_idx:d}] {crt_name:s} after FWD")

        if isinstance(module, KroneckerFactored):
            self.__fwd_hook(module, inputs, output)
            return
        if is_linear(module):
            self.__linear_fwd_hook(module, inputs, output)
        elif is_activation(module):
            self.__activation_fwd_hook(module, inputs, output)
        elif is_convolutional(module):
            self.__conv_fwd_hook(module, inputs, output)
        else:
            if not self.__use_fisher:
                raise ArchitectureNotSupported("You shouldn't be here!")
        self.__layer_idx += 1

    def _kf_bwd_hook(self, module, grad_input, grad_output) -> None:
        if self.__verbose:
            crt_name = module._get_name()
            print(f"[{self.__layer_idx:d}] {crt_name:s} BWD")

        if isinstance(module, KroneckerFactored):
            self.__bwd_hook(module, grad_input, grad_output)
            return
        if is_linear(module):
            if self.__maybe_exact:
                self.__maybe_exact = self.__prev_layer is None or \
                    self.__prev_layer == self.ACTIVATION
            self.__linear_bwd_hook(module, grad_input, grad_output)
            self.__next_parametric = module
            self.__prev_layer = self.LINEAR
        elif is_activation(module):
            self.__maybe_exact = (self.__maybe_exact and self.__prev_layer == self.LINEAR)
            if self.__maybe_exact:
                self.__activation_bwd_hook(module, grad_input, grad_output)
            self.__prev_layer = self.ACTIVATION
        elif is_convolutional(module):
            self.__maybe_exact = False
            self.__conv_bwd_hook(module, grad_input, grad_output)
            self.__prev_layer = self.CONVOLUTIONAL
        else:
            self.__maybe_exact = False
            self.__prev_layer = self.OTHER

        if not self.__maybe_exact:
            self.__d2f_dx2.clear()
            self.__df_dx.clear()
            self.__batch_preactivations.clear()

        self.__layer_idx -= 1
        if self.__layer_idx < 0:
            self.__phase = self.DONE
            if self.__verbose:
                print("Done with this batch!")

    # Magic happens below

    def __fwd_hook(self, _module, _inputs, _output):
        self.__phase = self.BACKWARD
        self.__layer_idx -= 1
        self.__maybe_exact = True
        self.__prev_layer = None

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

    def __conv_fwd_hook(self, module, inputs, output):
        module_name = self.__module_names[id(module)]
        assert isinstance(inputs, tuple) and len(inputs) == 1
        assert isinstance(output, Tensor)
        inputs, = inputs

        ch_out, ch_in, k_h, k_w = module.weight.size()
        s_h, s_w = module.stride
        b_sz, ch_in_, h_in, w_in = inputs.size()
        h_out = (h_in - k_h + 0) // s_h + 1
        w_out = (w_in - k_w + 0) // s_w + 1
        b_sz_, ch_out_, h_out_, w_out_ = output.size()

        print(h_out, w_out)

        assert ch_in_ == ch_in
        assert h_out_ == h_out
        assert w_out == w_out_ and \
            ch_out_ == ch_out and b_sz_ == b_sz
        print(ch_out, ch_in, k_h, k_w)

        x = inputs.new().resize_(b_sz, h_out, w_out, ch_in, k_h, k_w)

        for idx_h in range(0, h_out):
            start_h = idx_h * s_h
            for idx_w in range(0, w_out):
                start_w = idx_w * s_w
                x[:, idx_h, idx_w, :, :, :].copy_(inputs[:, :, start_h:(start_h + k_h), start_w:(start_w + k_w)])

        x = x.view(b_sz * h_out * w_out, ch_in * k_h * k_w)
        if self.__do_checks:
            self.__conv_special_inputs[module_name] = x

        x = torch.cat([x, x.new().resize_(x.size(0), 1).fill_(1)], dim=1)

        if self.__do_checks:
            weight_extra = torch.cat([module.weight
                                      #.view(ch_out, ch_in, -1)
                                      #.transpose(1, 2).contiguous()
                                      .view(ch_out, -1),
                                      module.bias.view(ch_out, -1)], dim=1)
            y = torch.matmul(x, weight_extra.t()).view(b_sz, h_out * w_out, ch_out)\
                .transpose(1, 2)\
                .view(b_sz, ch_out, h_out, w_out)
            print((y - output).abs().sum())
            assert (y - output).abs().max() < 1e-5
            print("----------------")
            # assert torch.allclose(y, output)

        inputs_cov = torch.mm(x.t(), x) / b_sz
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

    def __conv_bwd_hook(self, module, grad_input, grad_output):
        print([t.size() for t in grad_input])
        assert isinstance(grad_input, tuple) and len(grad_input) == 3
        assert isinstance(grad_output, tuple) and len(grad_output) == 1
        module_name = self.__module_names[id(module)]
        dx, dw, db = grad_input
        dy, = grad_output
        b_sz, ch_out, h_out, w_out = dy.size()
        dy = dy.view(b_sz, ch_out, -1).transpose(1, 2).contiguous().view(-1, ch_out)
        if self.__do_checks:
            ch_out_, ch_in, k_h, k_w = module.weight.size()
            assert ch_out == ch_out_
            x = self.__conv_special_inputs[module_name]
            b_sz = dx.size(0)
            ch_out = dy.size(1)

            dw_ = torch.mm(dy.t(), x)
            print(dw_.view(ch_out, k_h, k_w, -1))
            print(dw)

        act = torch.mm(dy.t(), dy) / (b_sz * h_out * w_out)

        if self.__average_factors:
            if module_name in self.__preactivations:
                self.__preactivations[module_name] += act
            else:
                self.__preactivations[module_name] = act
        else:
            self.__preactivations.setdefault(module_name, []).append(act)

    def __linear_bwd_hook(self, module, grad_input, grad_output):
        assert self.__phase == self.BACKWARD
        if self.__maybe_exact:
            self.__linear_exact_hessian(module, grad_input, grad_output)
        else:
            self.__linear_fisher(module, grad_input, grad_output)

        if self.__layer_idx == 0:
            self.__batches_no += 1

    def __linear_fisher(self, module, grad_input, grad_output):
        module_name = self.__module_names[id(module)]
        assert isinstance(grad_output, tuple) and len(grad_output) == 1
        grad_output, = grad_output
        act = torch.matmul(grad_output.t(), grad_output)
        act /= float(grad_output.size(0))
        if self.__average_factors:
            if module_name in self.__preactivations:
                self.__preactivations[module_name] += act
            else:
                self.__preactivations[module_name] = act
        else:
            self.__preactivations.setdefault(module_name, []).append(act)

    def __linear_exact_hessian(self, module, grad_input, grad_output):
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


def test_mlp():
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


def test_simple():
    s_net = LeNet([1, 2], [(3, 3)], [(1, 1)], [18, 1])
    s_net.zero_grad()
    s_net.do_kf = True
    data = torch.randn(1, 1, 5, 5, requires_grad=True)
    output = s_net(data)
    output.backward()


def test_lenet():
    filters_no = [3, 2, 3, 2]
    filter_size = [(4, 7), (3, 5), (5, 3)]
    stride = [(1, 1), (2, 3), (1, 1)]
    size = [7, 5, 11, 13, 17, 3]
    in_size = (21, 23)
    h, w = in_size
    for ((k_h, k_w), (s_h, s_w)) in zip(filter_size, stride):
        h, w = (h - k_h + 0) // s_h + 1, (w - k_w + 0) // s_w + 1
        print(h, w)
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

    print(kfhp.hessian_product_loss(dummy_vector))


def main():
    test_mlp()
    test_simple()
    test_lenet()


if __name__ == "__main__":
    print("Running torch version", torch.__version__)
    main()
