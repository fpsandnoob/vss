from abc import ABC, abstractmethod
import torch
from typing import Tuple
from diffusers import VQModel
from einops import rearrange, repeat, reduce

from src.models.components.measurement_op import NonLinearOp, LinearOp, OpCompose, NoiseOp

class NormModule(torch.nn.Module):
    def __init__(self, ord=1) -> None:
        super().__init__()
        self.ord = ord
        
    def forward(self, x, y):
        return torch.linalg.vector_norm(x - y, ord=self.ord)

class Optimizer(ABC):
    @abstractmethod
    def __call__(self, x, grad, scale):
        pass


class GenericGuidanceModule(ABC):
    def __init__(
        self,
        degrade_op: Tuple[OpCompose, LinearOp, NonLinearOp],
        noise_op: Tuple[OpCompose, NoiseOp]=None,
    ) -> None:
        self.degrade_op = degrade_op
        self.noise_op = noise_op
        super().__init__()

    @abstractmethod
    def grad_and_diff(self, data, **kwargs):
        # generic forward
        pass

    @abstractmethod
    def conditioning(self, data, measurements, **kwargs):
        pass

class SGDOptimizer(Optimizer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x, grad, scale):
        return x - scale * grad


class MomentumOptimizer(Optimizer):
    def __init__(self, momentum=0.9) -> None:
        super().__init__()
        self.momentum = momentum
        self.m = None

    def __call__(self, x, grad, scale):
        if self.m is None:
            self.m = grad
        else:
            self.m = (1 - self.momentum) * grad + self.momentum * self.m
        g = (1 - self.momentum) * grad + self.momentum * self.m
        return x - scale * g


class AdamOptimizer(Optimizer):
    def __init__(self, etas=(0.9, 0.999), varepsilon=1e-9) -> None:
        super().__init__()
        self.etas = etas
        self.m = None
        self.v = None
        self.varepsilon = varepsilon

    def __call__(self, x, grad, scale):
        if self.m is None:
            self.m = torch.zeros_like(grad).type_as(grad)
        else:
            self.m = (1 - self.etas[0]) * grad + self.etas[0] * self.m
        if self.v is None:
            self.v = torch.zeros_like(grad).type_as(grad)
        else:
            self.v = (1 - self.etas[1]) * grad**2 + self.etas[1] * self.v

        self.m_hat = self.m / (1 - self.etas[0])
        self.v_hat = self.v / (1 - self.etas[1])

        return x - scale * self.m_hat / (torch.sqrt(self.v_hat) + self.varepsilon)


class PixelGuidenceModule(GenericGuidanceModule):
    def __init__(
        self,
        degrade_op: Tuple[OpCompose, LinearOp, NonLinearOp],
        noise_op: Tuple[OpCompose, NoiseOp],
        diff_module: torch.nn.Module,
    ) -> None:
        super().__init__(degrade_op, noise_op)
        self.diff_module = diff_module
        self.l2_dist = torch.nn.MSELoss()

    def grad_and_diff(self, x_prev, x_0_hat, x_t, measurement, **kwargs):
        # calculate grad and diff
        # print('x_0_hat:', x_0_hat.max(), x_0_hat.min())
        deg_x_0_hat = self.degrade_op.forward(x_0_hat)
        # print('deg_x_0_hat', deg_x_0_hat.max(), deg_x_0_hat.min())
        # print('measurement', measurement.max(), measurement.min())

        diff_val = self.diff_module(measurement, deg_x_0_hat)
        l2_dist = self.l2_dist(deg_x_0_hat, measurement)

        # norm = torch.linalg.norm(diff_val)
        norm_grad = torch.autograd.grad(diff_val, x_t)[0]

        return norm_grad, diff_val, l2_dist


class LatentGuidenceModule(GenericGuidanceModule):
    def __init__(
        self,
        degrade_op: Tuple[OpCompose, LinearOp, NonLinearOp],
        noise_op: Tuple[OpCompose, NoiseOp],
        diff_module: torch.nn.Module,
        vae: VQModel,
    ) -> None:
        super().__init__(degrade_op, noise_op)
        if diff_module is not None:
            self.diff_module = diff_module.cuda()
        self.vae = vae
        self.l2_dist = torch.nn.MSELoss()

    def grad_and_diff(self, l_prev, l_0_hat, l_t, measurement, **kwargs):
        # with torch.enable_grad():
        x_0_hat = self.vae.decode(l_0_hat).sample
        # x0_hat_hu = (x_0_hat / 2 + 0.5).clamp(0, 1) * (3200 + 2048) - 2048
        # x_0_hat = ((x0_hat_hu / 1000) * 0.0192) + 0.0192
        deg_x_0_hat = self.degrade_op.forward(x_0_hat)
        
        # print(measurement.max(), measurement.min())
        # print(deg_x_0_hat.max(), measurement.min())
        diff_val = self.diff_module(measurement, deg_x_0_hat)
        l2_dist = self.l2_dist(deg_x_0_hat, measurement)

        # print(diff_val.dtype, l_t.dtype)
        norm_grad = torch.autograd.grad(diff_val, l_t)[0]

        return norm_grad, diff_val, l2_dist


class PixelManifoldConstraintGradient(PixelGuidenceModule):
    def __init__(
        self,
        scale,
        degrade_op: Tuple[OpCompose, LinearOp, NonLinearOp],
        noise_op: Tuple[OpCompose, NoiseOp],
        diff_module: torch.nn.Module,
    ) -> None:
        super().__init__(degrade_op, noise_op, diff_module)
        self.scale = scale

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, mask, **kwargs):
        # with torch.enable_grad():
        # calculate grad and diff
        # deg_measurement = self.degrade_op.forward(measurement)
        norm_grad, diff_val, l2_dist = self.grad_and_diff(
            x_prev=x_prev, x_0_hat=x_0_hat, x_t=x_t, measurement=measurement, **kwargs
        )
        if not isinstance(self.degrade_op, (SuperResolutionOp, CTSparseViewOp)):
            norm_grad = norm_grad * (1 - self.degrade_op.mask)
            x_t = x_prev * (1 - self.degrade_op.mask) - self.scale * norm_grad + x_prev * self.degrade_op.mask
        else:
            x_t = x_prev - self.scale * norm_grad
            x_t = x_t + self.degrade_op.backproject(measurement).clamp(1, -1) - self.degrade_op.backproject(self.degrade_op.forward(x_t)).clamp(1, -1)

        # print(x_t.dtype, x_t.max(), x_t.min())

        # X + A^T(Y - AX)
        # new_measurement = torch.sqrt(alphas_cumprod) * measurement + torch.sqrt(1 - alphas_cumprod) * measurement
        # x_t = self.degrade_op.backproject(measurement) + x_0_hat - self.degrade_op.backproject(self.degrade_op.forward(x_0_hat))
        
        return x_t, diff_val, l2_dist


class LatentManifoldConstraintGradient(LatentGuidenceModule):
    def __init__(
        self,
        scale,
        degrade_op: Tuple[OpCompose, LinearOp, NonLinearOp],
        noise_op: Tuple[OpCompose, NoiseOp],
        diff_module: torch.nn.Module,
        vae: VQModel,
    ) -> None:
        super().__init__(degrade_op, noise_op, diff_module, vae)
        self.scale = scale

    def conditioning(self, l_prev, l_t, l_0_hat, measurement, alphas_cumprod, **kwargs):
        # calculate grad and diff
        norm_grad, diff_val, l2_dist = self.grad_and_diff(
            l_prev=l_prev, l_0_hat=l_0_hat, l_t=l_t, measurement=measurement, **kwargs
        )
        l_t = l_prev - self.scale * norm_grad * (1 - self.degrade_op.mask)

        # E(D(L) + A^T(Y - AD(L)))
         
        # new_measurement = torch.sqrt(alphas_cumprod) * measurement + torch.sqrt(1 - alphas_cumprod) * measurement
        # decode_x_t_hat = self.degrade_op.backproject(new_measurement) + decode_l_t - self.degrade_op.backproject(self.degrade_op.forward(decode_l_t))
        decode_x_t_hat = decode_l_t * (1 - self.degrade_op.mask)  + decode_l_t * self.degrade_op.mask
        l_t = self.vae.encode(decode_x_t_hat).latents
        return l_t, diff_val, l2_dist

class PixelDDNM(PixelGuidenceModule):
    def __init__(
        self,
        degrade_op: Tuple[OpCompose, LinearOp, NonLinearOp],
        noise_op: Tuple[OpCompose, NoiseOp],
    ) -> None:
        super().__init__(degrade_op, noise_op, None)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        # with torch.enable_grad():
        # calculate grad and diff

        # X + A^T(Y - AX)
        # x_t = x_0_hat + self.degrade_op.backproject(
        #     measurement - self.degrade_op.forward(x_0_hat)
        # )
        # x_t = self.degrade_op.backproject(measurement) + x_0_hat - self.degrade_op.backproject(self.degrade_op.forward(x_0_hat))
        # x_t = x_0_hat * (1 - self.degrade_op.mask) + measurement
        x_0_hat = x_0_hat - self.degrade_op.backproject(self.degrade_op.forward(x_0_hat) - measurement)
        return x_0_hat, None, torch.tensor(0.0)


class LatentDDNM(LatentGuidenceModule):
    def __init__(
        self,
        degrade_op: Tuple[OpCompose, LinearOp, NonLinearOp],
        noise_op: Tuple[OpCompose, NoiseOp],
        vae: VQModel,
    ) -> None:
        super().__init__(degrade_op, noise_op, None, vae)

    def conditioning(self, l_prev, l_t, l_0_hat, measurement, **kwargs):
        # E(D(L) + A^T(Y - AD(L)))
        decode_l_t = self.vae.decode(l_0_hat).sample
        decode_x_t_hat = self.degrade_op.backproject(measurement) + decode_l_t - self.degrade_op.backproject(self.degrade_op.forward(decode_l_t))
        l_t = self.vae.encode(decode_x_t_hat).latents
        return l_t, None, torch.tensor(0.0)

class PixelDeepLatentIterativeReconstruct(PixelGuidenceModule):
    def __init__(
        self,
        scale,
        optimizer,
        degrade_op: Tuple[OpCompose, LinearOp, NonLinearOp],
        noise_op: Tuple[OpCompose, NoiseOp],
        diff_module: torch.nn.Module,
    ) -> None:
        super().__init__(degrade_op, noise_op, diff_module)
        self.scale = scale
        self.optimizer = optimizer
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.time_list = []

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        # calculate grad and diff
        self.starter.record()
        norm_grad, diff_val, l2_dist = self.grad_and_diff(
            x_prev=x_prev, x_0_hat=x_0_hat, x_t=x_t, measurement=measurement, **kwargs
        )
        x_prev = self.optimizer(x_prev, norm_grad, self.scale)
        self.ender.record()
        torch.cuda.synchronize()
        curr_time = self.starter.elapsed_time(self.ender)
        # print(curr_time)
        self.time_list.append(curr_time)

        return x_prev, diff_val, l2_dist


class LatentDeepLatentIterativeReconstruct(LatentGuidenceModule):
    
    def __init__(
        self,
        scale,
        optimizer,
        degrade_op: Tuple[OpCompose, LinearOp, NonLinearOp],
        noise_op: Tuple[OpCompose, NoiseOp],
        diff_module: torch.nn.Module,
        vae: VQModel,
    ) -> None:
        super().__init__(degrade_op, noise_op, diff_module, vae)
        self.scale = scale
        self.optimizer = optimizer
        # self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # self.time_list = []

    def conditioning(self, l_prev, l_t, l_0_hat, measurement, **kwargs):
        # calculate grad and diff
        
        # self.starter.record()
        norm_grad, diff_val, l2_dist = self.grad_and_diff(
            l_prev=l_prev, l_0_hat=l_0_hat, l_t=l_t, measurement=measurement, **kwargs
        )
        l_prev = self.optimizer(l_prev, norm_grad, self.scale)
        # self.ender.record()
        # torch.cuda.synchronize()
        # curr_time = self.starter.elapsed_time(self.ender)
        # print(curr_time)
        # self.time_list.append(curr_time)
        
        return l_prev, diff_val, l2_dist

class PixelPosteriorSampling(PixelDeepLatentIterativeReconstruct):
    def __init__(
        self,
        scale: 1,
        degrade_op: Tuple[OpCompose, LinearOp, NonLinearOp],
        noise_op: Tuple[OpCompose, NoiseOp],
    ) -> None:
        super().__init__(scale, SGDOptimizer(), degrade_op, noise_op, NormModule(ord=2))

class LatentPosteriorSampling(LatentDeepLatentIterativeReconstruct):
    def __init__(
        self,
        scale: 1,
        degrade_op: Tuple[OpCompose, LinearOp, NonLinearOp],
        noise_op: Tuple[OpCompose, NoiseOp],
        vae: VQModel,
    ) -> None:
        super().__init__(scale, SGDOptimizer(), degrade_op, noise_op, NormModule(ord=2), vae)
        # super().__init__(1, MomentumOptimizer(), degrade_op, noise_op, NormModule(p=2), vae)
        # super().__init__(1, AdamOptimizer(), degrade_op, noise_op, NormModule(p=2), vae)
