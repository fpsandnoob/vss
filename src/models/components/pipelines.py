from typing import Any, Dict, List, Optional, Tuple, Union
import PIL
from diffusers import (
    VQModel,
    UNet2DModel,
    ImagePipelineOutput,
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
)
from diffusers.utils import randn_tensor
import numpy as np
import torch
from traitlets import Callable
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, Adafactor
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from kornia.losses import total_variation
from src.models.components.measurement_ops import RandomBoxInpaintingOP
from einops import rearrange, repeat, reduce
import kornia
import tqdm

from .guidence_modules import PixelGuidenceModule, LatentGuidenceModule


class PixelReconstructionPipeline(DiffusionPipeline):
    def __init__(
        self,
        unet: UNet2DModel,
        condition_module: PixelGuidenceModule,
        scheduler: DDPMScheduler,
    ) -> None:
        super().__init__()
        self.register_modules(
            unet=unet,
            condition_module=condition_module,
            scheduler=scheduler,
        )
        # self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # self.time_list = []

    def __call__(
        self,
        measurement,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        *args,
        **kwargs
    ) -> Union[ImagePipelineOutput, Tuple]:
        noise_shape = (
            measurement.shape[0],
            self.unet.in_channels,
            self.unet.sample_size,
            self.unet.sample_size,
        )
        noise = randn_tensor(
            noise_shape,
            generator=generator,
            device=measurement.device,
            dtype=self.unet.dtype,
        )

        self.scheduler.set_timesteps(num_inference_steps)

        mask = None
        if isinstance(self.condition_module.degrade_op, RandomBoxInpaintingOP):
            mask = self.condition_module.degrade_op.mask

        losses = []
        prev_momentum = None
        with torch.enable_grad():
            for t in self.progress_bar(self.scheduler.timesteps):
                noise.requires_grad = True
                # self.starter.record()
                model_output = self.unet(noise, t).sample
                model_output = self.scheduler.scale_model_input(model_output, t)
                x_samples = self.scheduler.step(
                    model_output,
                    t,
                    noise,
                )
                # self.ender.record()
                # torch.cuda.synchronize()
                # curr_time = self.starter.elapsed_time(self.ender)
                # print(curr_time)
                # self.time_list.append(curr_time)

                x_prev = x_samples.prev_sample
                x0_hat = x_samples.pred_original_sample

                x_t, diff_val, l2_dist = self.condition_module.conditioning(
                    x_prev=x_prev,
                    x_t=noise,
                    x_0_hat=x0_hat,
                    measurement=measurement,
                    mask=mask,
                    step=t,
                )

                losses.append(l2_dist)
                noise = x_t.detach()

                # latents_xt_neg_1 = x_samples.prev_sample
                # latents_x0_hat = x_samples.pred_original_sample

                # latents_x0_hat_pro = self.condition_module.degrade_op.forward(latents_x0_hat)
                # norm = torch.norm(
                #     measurement - latents_x0_hat_pro,
                #     # p=2
                #     p=1
                # )

                # loss = torch.mean((measurement - latents_x0_hat_pro) ** 2)
                # losses.append(loss.detach().cpu().item())

                # norm_grad = torch.autograd.grad(norm, noise)[0]

                # # sgd-like update
                # # if optimizer_order == 0:
                # beta_1=0.9
                # if prev_momentum is None:
                #         momentum = norm_grad
                # else:
                #     momentum = beta_1 * prev_momentum + (1 - beta_1) * norm_grad
                # prev_momentum = momentum
                # momentum = (1- beta_1) * norm_grad + beta_1 * momentum
                # noise = latents_xt_neg_1 - momentum * 0.5

                # noise = noise.detach()

        # print("time :", np.average(self.condition_module.time_list))
        image = x_t
        print(image.max(), image.min())
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().detach()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,), losses

        return ImagePipelineOutput(images=image)


    def __init__(
        self,
        unet: UNet2DModel,
        condition_module: PixelGuidenceModule,
        scheduler: DDPMScheduler,
    ) -> None:
        super().__init__()
        self.scheduler=scheduler
        self.register_modules(
            unet=unet,
            condition_module=condition_module,
            scheduler=scheduler,
        )
        # self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # self.time_list = []

    def __call__(
        self,
        measurement,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        *args,
        **kwargs
    ) -> Union[ImagePipelineOutput, Tuple]:
        noise_shape = (
            measurement.shape[0],
            self.unet.in_channels,
            self.unet.sample_size,
            self.unet.sample_size,
        )
        noise = randn_tensor(
            noise_shape,
            generator=generator,
            device=measurement.device,
            dtype=self.unet.dtype,
        )

        self.scheduler.set_timesteps(num_inference_steps)

        mask = None
        if isinstance(self.condition_module.degrade_op, RandomBoxInpaintingOP):
            mask = self.condition_module.degrade_op.mask

        losses = []
        prev_momentum = None
        with torch.enable_grad():
            for t in self.progress_bar(self.scheduler.timesteps):
                noise.requires_grad = True
                # self.starter.record()
                model_output = self.unet(noise, t).sample
                model_output = self.scheduler.scale_model_input(model_output, t)
                x_samples = self.scheduler.step(
                    model_output,
                    t,
                    noise,
                )
                # self.ender.record()
                # torch.cuda.synchronize()
                # curr_time = self.starter.elapsed_time(self.ender)
                # print(curr_time)
                # self.time_list.append(curr_time)

                x_prev = x_samples.prev_sample
                x0_hat = x_samples.pred_original_sample

                x_t, diff_val, l2_dist = self.condition_module.conditioning(
                    x_prev=x_prev,
                    x_t=noise,
                    x_0_hat=x0_hat,
                    measurement=measurement,
                    mask=mask,
                    step=t,
                )

                best_diff = 99 * torch.rand_like(x_prev)
                device = model_output.device

                for _ in range(20):
                    variance_noise = randn_tensor(
                        model_output.shape, generator=generator, device=device, dtype=model_output.dtype
                    )
                    x_t_sample = (self.scheduler._get_variance(t, predicted_variance=None) ** 0.5) * variance_noise + x_t
                    x_t_sample = (
                        x_t_sample - torch.sqrt(
                            1. - self.scheduler.alphas_cumprod[t]
                        ) * model_output
                    )
                    difference = self.scheduler.alphas_cumprod[t] ** 0.5 * measurement - self.condition_module.degrade_op.forward(x_t_sample)

                    if torch.linalg.norm(difference) < torch.linalg.norm(best_diff):
                        best_diff = difference
                        best_noise = variance_noise

                if t != 0:
                    x_t = x_t + (self.scheduler._get_variance(t, predicted_variance=None) ** 0.5) * best_noise

                losses.append(l2_dist)
                noise = x_t.detach()

                # latents_xt_neg_1 = x_samples.prev_sample
                # latents_x0_hat = x_samples.pred_original_sample

                # latents_x0_hat_pro = self.condition_module.degrade_op.forward(latents_x0_hat)
                # norm = torch.norm(
                #     measurement - latents_x0_hat_pro,
                #     # p=2
                #     p=1
                # )

                # loss = torch.mean((measurement - latents_x0_hat_pro) ** 2)
                # losses.append(loss.detach().cpu().item())

                # norm_grad = torch.autograd.grad(norm, noise)[0]

                # # sgd-like update
                # # if optimizer_order == 0:
                # beta_1=0.9
                # if prev_momentum is None:
                #         momentum = norm_grad
                # else:
                #     momentum = beta_1 * prev_momentum + (1 - beta_1) * norm_grad
                # prev_momentum = momentum
                # momentum = (1- beta_1) * norm_grad + beta_1 * momentum
                # noise = latents_xt_neg_1 - momentum * 0.5

                # noise = noise.detach()

        # print("time :", np.average(self.condition_module.time_list))
        image = noise
        print(image.max(), image.min())
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().detach()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,), losses

        return ImagePipelineOutput(images=image)
    
class PixelReconstructionREDDiffPipeline(DiffusionPipeline):
    def __init__(
        self,
        unet: UNet2DModel,
        condition_module: LatentGuidenceModule,
        scheduler: DDPMScheduler,
    ) -> None:
        super().__init__()
        self.register_modules(
            unet=unet,
            condition_module=condition_module,
            scheduler=scheduler,
        )

        # self.sigma_1 = torch.nn.Parameter(torch.ones(1)).cuda()
        # self.sigma_2 = torch.nn.Parameter(torch.ones(1)).cuda()
        

        # self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # self.time_list = []

    def find_fixed_point(self, image, t, noise, alpha):
        lt = alpha.sqrt() * image + (1 - alpha).sqrt() * noise

        model_output = self.unet(lt, t).sample
        model_output = self.scheduler.scale_model_input(model_output, t)

        et = model_output
        et = et.detach()
        loss_noise = torch.mul((et - noise).detach(), image)
        return loss_noise

    def __call__(
        self,
        measurement,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        num_corase_search: int = 0,
        num_fine_search: int = 100,
        sigma_x0: float = 0.0,
        eta: float = 0.0,
        grad_term_weight: float = 1,
        obs_weight: float = 0.1,
        cond_awd: bool = False,
        awd: bool = True,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        unet_phi=None,
        **kwargs
    ) -> Union[ImagePipelineOutput, Tuple]:
        self.unet: UNet2DModel
        self.scheduler: DDPMScheduler
        self.condition_module: LatentGuidenceModule

        noise_shape = (
            measurement.shape[0],
            self.unet.in_channels,
            self.unet.sample_size,
            self.unet.sample_size,
        )
        im = randn_tensor(
            noise_shape,
            generator=generator,
            device=measurement.device,
            dtype=self.unet.dtype,
        )
        # latents = self.condition_module.vae.encode(self.condition_module.degrade_op.backproject(measurement)).latents
        
        self.scheduler.set_timesteps(num_inference_steps)
        ss = [-1] + list(list(range(num_inference_steps))[:-1])
        
        losses = []

        with torch.enable_grad():
            for t, s in zip(self.progress_bar(self.scheduler.timesteps), reversed(ss)):
                # s = torch.ones(latents.shape[0]).type_as(t) * s
                
                # with torch.no_grad():
                #     x0_hat = self.condition_module.vae.decode(latents).sample
                #     deg_x0_hat = self.condition_module.degrade_op.forward(x0_hat)
                #     x_measurement = self.condition_module.degrade_op.backproject(measurement)
                #     x_goodness = x0_hat - self.condition_module.degrade_op.backproject(deg_x0_hat) + x_measurement
                #     latents = self.condition_module.vae.encode(x_goodness).latents
                
                im.requires_grad = True
                
                # print(im.max(), im.min()
                degrade_im = self.condition_module.degrade_op.forward(im)
                # print(degrade_im.max(), degrade_im.min(), measurement.max(), measurement.min())
                loss_diff_val = torch.linalg.vector_norm(degrade_im - measurement)
                l2_dist = torch.nn.functional.mse_loss(degrade_im, measurement)
                
                # x_measurement = self.condition_module.degrade_op.backproject(measurement)
                # x_goodness = x0_hat - self.condition_module.degrade_op.backproject(deg_x0_hat) + x_measurement
                # # deg_x0_hat = self.condition_module.degrade_op.forward(x_goodness)
                # loss_diff_bp = self.condition_module.diff_module(
                #     measurement, deg_x0_hat
                # )
                

                alpha_t = self.scheduler.alphas_cumprod[t]

                sigma_x0 = sigma_x0  # 0.0001
                noise_xt = torch.randn_like(im).type_as(im)

                loss_noise = self.find_fixed_point(im, t, noise_xt, alpha_t)
                       
                snr_inv = (1 - alpha_t).sqrt() / alpha_t.sqrt()

                w_t = grad_term_weight * snr_inv
                v_t = obs_weight

                # print(loss_noise.mean(), loss_diff_val)
                total_loss = w_t * loss_noise.mean() + v_t * loss_diff_val
                
                grad = torch.autograd.grad(total_loss, im)[0]
                # torch.nn.utils.clip_grad.clip_grad_value_(grad, 1)
                im_prev = self.condition_module.optimizer(
                    im, grad, self.condition_module.scale
                )

                losses.append(l2_dist)
                im = im_prev.detach()
                
                
        image = im
        print(image.max(), image.min())
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().detach()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,), losses

        return ImagePipelineOutput(images=image)


class LatentReconstructionPipeline(DiffusionPipeline):
    def __init__(
        self,
        unet: UNet2DModel,
        vae: VQModel,
        condition_module: LatentGuidenceModule,
        scheduler: DDPMScheduler,
        latents_height,
        latents_width,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            condition_module=condition_module,
            scheduler=scheduler,
        )

        self.latents_height = latents_height
        self.latents_width = latents_width

        # self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # self.time_list = []

    def __call__(
        self,
        measurement,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        *args,
        **kwargs
    ) -> Union[ImagePipelineOutput, Tuple]:
        latent_shape = (
            measurement.shape[0],
            self.unet.in_channels,
            self.latents_height,
            self.latents_width,
        )
        latents = randn_tensor(
            latent_shape,
            generator=generator,
            device=measurement.device,
            dtype=self.unet.dtype,
        )

        self.scheduler.set_timesteps(num_inference_steps)

        losses = []
        # optimizer_order = 1
        # p=1
        # guide_rate=0.5
        # prev_momentum = None
        # prev_velocity = None
        # beta_1 = 0.9
        # beta_2 = 0.999
        with torch.enable_grad():
            for t in self.progress_bar(self.scheduler.timesteps):
                latents.requires_grad = True
                # self.starter.record()
                model_output = self.unet(latents, t).sample
                model_output = self.scheduler.scale_model_input(model_output, t)
                l_samples = self.scheduler.step(
                    model_output,
                    t,
                    latents,
                )
                # self.ender.record()
                # torch.cuda.synchronize()
                # curr_time = self.starter.elapsed_time(self.ender)
                # print(curr_time)
                # self.time_list.append(curr_time)

                l_prev = l_samples.prev_sample
                l0_hat = l_samples.pred_original_sample

                prev_t = (
                    t
                    - self.scheduler.config.num_train_timesteps
                    // self.scheduler.num_inference_steps
                )
                variance = self.scheduler._get_variance(t, prev_t)

                l_prev, diff_val, l2_dist = self.condition_module.conditioning(
                    l_prev=l_prev,
                    l_t=latents,
                    l_0_hat=l0_hat,
                    measurement=measurement,
                    alphas_cumprod=self.scheduler.alphas_cumprod[t.long()],
                    variance=variance,
                )

                losses.append(l2_dist)
                # losses.append(torch.tensor(1.0))
                latents = l_prev.detach()

                # latents_xt_neg_1 = l_samples.prev_sample
                # latents_x0_hat = l_samples.pred_original_sample

                # latents_x0_hat_pro = self.condition_module.degrade_op.forward(self.vae.decode(latents_x0_hat).sample)
                # norm = torch.norm(
                #     measurement - latents_x0_hat_pro,
                #     p=p
                #     # p=1
                # )

                # loss = torch.mean((measurement - latents_x0_hat_pro) ** 2)
                # losses.append(loss.detach().cpu().item())

                # norm_grad = torch.autograd.grad(norm, latents)[0]

                # norm_grad, diff_val, l2_dist = self.condition_module.grad_and_diff(l_prev=latents_xt_neg_1, l_t=latents, l_0_hat=latents_x0_hat, measurement=measurement)

                # # sgd-like update
                # if optimizer_order == 0:
                #     latents = latents_xt_neg_1 - norm_grad * guide_rate

                # # sgd with momentum-like update
                # elif optimizer_order == 1:
                #     if prev_momentum is None:
                #         momentum = norm_grad
                #     else:
                #         momentum = beta_1 * prev_momentum + (1 - beta_1) * norm_grad
                #     prev_momentum = momentum
                #     momentum = (1- beta_1) * norm_grad + beta_1 * momentum
                #     latents = latents_xt_neg_1 - momentum * guide_rate

                # # adam-like update
                # elif optimizer_order == 2:
                #     if prev_momentum is None:
                #         m = norm_grad
                #         v = norm_grad ** 2
                #     else:
                #         m = beta_1 * prev_momentum + (1 - beta_1) * norm_grad
                #         v = beta_2 * prev_velocity + (1 - beta_2) * norm_grad ** 2
                #     prev_momentum = m
                #     prev_velocity = v
                #     m = m / (1 - beta_1)
                #     v = v / (1 - beta_2)
                #     grad = m / (torch.sqrt(v) + 1e-8)
                #     latents = latents_xt_neg_1 - grad * guide_rate

                # latents = latents.detach()
                # print(latents.max(), latents.min())

        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().detach()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,), losses

        return ImagePipelineOutput(images=image)


class LatentReconstructionREDDiffPipeline(DiffusionPipeline):
    def __init__(
        self,
        unet: UNet2DModel,
        vae: VQModel,
        condition_module: LatentGuidenceModule,
        scheduler: DDPMScheduler,
        latents_height,
        latents_width,
    ) -> None:
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            condition_module=condition_module,
            scheduler=scheduler,
        )

        # self.sigma_1 = torch.nn.Parameter(torch.ones(1)).cuda()
        # self.sigma_2 = torch.nn.Parameter(torch.ones(1)).cuda()
        
        self.latents_height = latents_height
        self.latents_width = latents_width
        self.l2_dist = torch.nn.MSELoss()

        # self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # self.time_list = []

    def find_fixed_point(self, latents, t, noise, alpha):
        lt = alpha.sqrt() * latents + (1 - alpha).sqrt() * noise

        model_output = self.unet(lt, t).sample
        model_output = self.scheduler.scale_model_input(model_output, t)

        et = model_output
        et = et.detach()
        loss_noise = torch.mul((et - noise).detach(), latents)
        return loss_noise

    def __call__(
        self,
        measurement,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        num_corase_search: int = 0,
        num_fine_search: int = 100,
        sigma_x0: float = 0.0,
        grad_term_weight: float = 10.0,
        obs_weight: float = 2.0,
        # obs_weight: float = 2.0,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        unet_phi=None,
        phi_weights=None
    ) -> Union[ImagePipelineOutput, Tuple]:
        self.unet: UNet2DModel
        self.scheduler: DDPMScheduler
        self.condition_module: LatentGuidenceModule

        latent_shape = (
            measurement.shape[0],
            self.unet.in_channels,
            self.latents_height,
            self.latents_width,
        )
        latents = randn_tensor(
            latent_shape,
            generator=generator,
            device=measurement.device,
            dtype=self.unet.dtype,
        )
        
        self.scheduler.set_timesteps(num_inference_steps)
        ss = [-1] + list(list(range(num_inference_steps))[:-1])
        
        phi_optimizer = Adafactor(phi_weights, lr=1e-4, scale_parameter=False, relative_step=False, warmup_init=False)
        loss_fn = torch.nn.MSELoss()

        losses = []

        with torch.enable_grad():
            for t, s in zip(self.progress_bar(self.scheduler.timesteps), reversed(ss)):
                s = torch.ones(latents.shape[0]).type_as(t) * s
                
                
                latents.requires_grad = True
                
                x0_hat = self.condition_module.vae.decode(latents).sample
                
                # x0_hat_hu = (x0_hat / 2 + 0.5).clamp(0, 1) * (3200 + 2048) - 2048
                # x0_hat = ((x0_hat_hu / 1000) * 0.0192) + 0.0192
                
                variation_loss = total_variation(x0_hat, reduction='sum')
                
                deg_x0_hat = self.condition_module.degrade_op.forward(x0_hat)
       
                loss_diff_val = torch.linalg.vector_norm(measurement - deg_x0_hat, ord=1.0)
                loss_diff_val = loss_diff_val.mean()
                l2_dist = self.l2_dist(deg_x0_hat, measurement)
                

                alpha_t = self.scheduler.alphas_cumprod[t]
                alpha_s = self.scheduler.alphas_cumprod[s]

                sigma_x0 = sigma_x0  # 0.0001
                noise_x0 = torch.randn_like(latents).type_as(latents)
                noise_xt = torch.randn_like(latents).type_as(latents)

                # VLDS
                # with torch.no_grad():
                #     lt = alpha_t.sqrt() * latents + (1 - alpha_t).sqrt() * noise_xt
                    
                #     et = self.unet(lt, t).sample
                #     et = self.scheduler.scale_model_input(et, t)
                #     et = et.detach()
                    
                # loss_noise = torch.mul((et - noise_xt).detach(), latents)
                
                # VSS
                with torch.no_grad():
                    latents_repeat = repeat(latents, "b c h w -> (b k) c h w", k=1)
                    noise_xt = torch.randn_like(latents_repeat).type_as(latents_repeat)
                    
                    lt = alpha_t.sqrt() * latents_repeat + (1 - alpha_t).sqrt() * noise_xt
                    
                    et_phi = unet_phi(lt, t).sample
                    et_phi = self.scheduler.scale_model_input(et_phi, t)
                    et_phi = et_phi.detach()
                    
                    et = self.unet(lt, t).sample
                    et = self.scheduler.scale_model_input(et, t)
                    et = et.detach()
                    
                loss_noise = torch.mul((et - et_phi).detach(), latents_repeat)
                
                snr_inv = (1 - alpha_t).sqrt() / alpha_t.sqrt()

                w_t = grad_term_weight  * snr_inv
                v_t = obs_weight
             
                # 0.05 for 32 views 
                total_loss =  w_t * loss_noise.mean() + v_t * loss_diff_val + 1e-2 * variation_loss
                
                
                grad = torch.autograd.grad(total_loss, latents)[0]
                l_prev = self.condition_module.optimizer(
                    # latents, grad, current_scale
                    latents, grad, self.condition_module.scale 
                )

                losses.append(l2_dist.mean())
                latents = l_prev.detach()
                
                clip_valve = torch.quantile(latents.abs(), q=0.95)
                latents = latents.clamp(-clip_valve, clip_valve) / clip_valve
                
                for i in range(1):
                    phi_optimizer.zero_grad()
                    noise_x0 = torch.randn_like(latents).type_as(latents)
                    noise_lt = self.scheduler.add_noise(latents, noise_x0, t)
                    model_output = unet_phi(noise_lt, t).sample
                    model_output = self.scheduler.scale_model_input(model_output, t)
                    et = model_output
                    et_phi = et
                    loss_noise_phi = loss_fn(et_phi, noise_x0)
                    loss_noise_phi.backward()
                    phi_optimizer.step()
                    
                
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().detach()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,), losses

        return ImagePipelineOutput(images=image)

class LatentReconstructionVPredictionREDDiffPipeline(DiffusionPipeline):
    def __init__(
        self,
        unet: UNet2DModel,
        vae: VQModel,
        condition_module: LatentGuidenceModule,
        scheduler: DDPMScheduler,
        latents_height,
        latents_width,
    ) -> None:
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            condition_module=condition_module,
            scheduler=scheduler,
        )

        # self.sigma_1 = torch.nn.Parameter(torch.ones(1)).cuda()
        # self.sigma_2 = torch.nn.Parameter(torch.ones(1)).cuda()
        
        self.latents_height = latents_height
        self.latents_width = latents_width
        self.l2_dist = torch.nn.MSELoss()

        # self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # self.time_list = []

    def find_fixed_point(self, latents, t, noise, alpha):
        lt = alpha.sqrt() * latents + (1 - alpha).sqrt() * noise

        model_output = self.unet(lt, t).sample
        model_output = self.scheduler.scale_model_input(model_output, t)

        et = model_output
        et = et.detach()
        loss_noise = torch.mul((et - noise).detach(), latents)
        return loss_noise
    
    def optimize_latent(self, x_0_hat, y, degrade_op):
        x_0_hat = x_0_hat.float()
        y = y.float()
        # mask = mask.float()
        
        # ones = torch.ones_like(x_0_hat).type_as(x_0_hat)
        A = lambda x: degrade_op.forward(x)
        # _, _AT = torch.func.vjp(A, ones)
        AT = lambda y: degrade_op.backproject(y)
        
        def cg_A(x):
            return AT(A(x))
        
        cg_y = AT(y)
        x_0_hat = self.CG(cg_A, cg_y, x_0_hat)
        return x_0_hat
    
    def CG(self, A_fn,b_cg,x,n_inner=10):
        # with torch.no_grad():
        eps = 1e-5
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.reshape(1,-1),r.reshape(1,-1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old/torch.matmul(p.reshape(1,-1),Ap.reshape(1,-1).T)
    
            x += a * p
            r -= a * Ap

            rs_new = torch.matmul(r.reshape(1,-1),r.reshape(1,-1).T)
            # print(torch.sqrt(rs_new))
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new/rs_old) * p
            rs_old = rs_new
        return x

    def __call__(
        self,
        measurement,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        num_corase_search: int = 0,
        num_fine_search: int = 100,
        sigma_x0: float = 0.0,
        grad_term_weight: float = 1.0,
        # obs_weight: float = 0.1,
        obs_weight: float = 2.0,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        unet_phi=None,
        phi_weights=None
    ) -> Union[ImagePipelineOutput, Tuple]:
        self.unet: UNet2DModel
        self.scheduler: DDPMScheduler
        self.condition_module: LatentGuidenceModule

        latent_shape = (
            measurement.shape[0],
            self.unet.in_channels,
            self.latents_height,
            self.latents_width,
        )
        # latents = randn_tensor(
            # latent_shape,
            # generator=generator,
            # device=measurement.device,
            # dtype=self.unet.dtype,
        # )
        latents = self.condition_module.vae.encode(self.condition_module.degrade_op.backproject(measurement)).latents
        latents = latents * 0.05 + torch.rand_like(latents) * 0.95
        
        self.scheduler.set_timesteps(num_inference_steps)
        ss = [-1] + list(list(range(num_inference_steps))[:-1])
        
        # phi_optimizer = torch.optim.AdamW(phi_weights, lr=1e-4)
        phi_optimizer = Adafactor(phi_weights, lr=1e-4, scale_parameter=False, relative_step=False, warmup_init=False)
        loss_fn = torch.nn.MSELoss()

        losses = []

        # sampling_t = torch.randint(0, 999, (2000, ), dtype=torch.long)
        ttt = self.progress_bar(self.scheduler.timesteps)
        # ttt = self.progress_bar(sampling_t)
        
        with torch.enable_grad():
            for t, s in zip(ttt, reversed(ss)):
                s = torch.ones(latents.shape[0]).type_as(t) * s

                latents.requires_grad = True
                x0_hat = self.condition_module.vae.decode(latents).sample
                # x0_hat = repeat(x0_hat, "b c h w -> (b k) c h w", k=5)
                # x0_hat = x0_hat + torch.randn_like(x0_hat) * (self.scheduler._get_variance(t) ** 0.5)
                # print(x0_hat.shape, x0_hat.max(), x0_hat.min())
                
                # x0_hat_hu = (x0_hat / 2 + 0.5).clamp(0, 1) * (3200 + 2048) - 2048
                # x0_hat = ((x0_hat_hu / 1000) * 0.0192) + 0.0192
                
                # print('x0_hat', x0_hat_hu.max(), x0_hat_hu.min())
                # print('x0_hat', x0_hat.max(), x0_hat.min())
                # variation_loss = total_variation(x0_hat, reduction='sum')
                
                deg_x0_hat = self.condition_module.degrade_op.forward(x0_hat)
                # print(deg_x0_hat.shape, deg_x0_hat.max(), deg_x0_hat.min())
                # print(measurement.shape, measurement.max(), measurement.min())
                
                loss_diff_val = torch.linalg.vector_norm(measurement - deg_x0_hat, ord=2)
                loss_diff_val = loss_diff_val.mean()
                l2_dist = self.l2_dist(deg_x0_hat, measurement)    

                alpha_t = self.scheduler.alphas_cumprod[t]
                alpha_s = self.scheduler.alphas_cumprod[s]

                sigma_x0 = sigma_x0  # 0.0001
                noise_x0 = torch.randn_like(latents).type_as(latents)
                noise_xt = torch.randn_like(latents).type_as(latents)
                    
                # with torch.no_grad():
                #     lt = alpha_t.sqrt() * latents + (1 - alpha_t).sqrt() * noise_xt
                    
                #     # et_phi = unet_phi(lt, t).sample
                #     # et_phi = self.scheduler.scale_model_input(et_phi, t)
                #     # et_phi = et_phi.detach()
                    
                #     et = self.unet(lt, t).sample
                #     et = self.scheduler.scale_model_input(et, t)
                #     et = et.detach()
                    
                # loss_noise = torch.mul((et - noise_xt).detach(), latents)


                # with torch.no_grad():
                #     latents_repeat = repeat(latents, "b c h w -> (b k) c h w", k=1)
                #     noise_xt = torch.randn_like(latents_repeat).type_as(latents_repeat)
                #     lt = alpha_t.sqrt() * latents_repeat + (1 - alpha_t).sqrt() * noise_xt
                    
                #     # et_phi = unet_phi(lt, t).sample
                #     # et_phi = self.scheduler.scale_model_input(et_phi, t)
                #     # et_phi = et_phi.detach()
                    
                #     et = self.unet(lt, t).sample
                #     et = self.scheduler.scale_model_input(et, t)
                #     et = et.detach()
                    
                # loss_noise = torch.mul((et - noise_xt).detach(), latents_repeat)
                
                # with torch.no_grad():
                #     lt = alpha_t.sqrt() * latents + (1 - alpha_t).sqrt() * noise_xt
                    
                #     et_phi = unet_phi(lt, t).sample
                #     et_phi = self.scheduler.scale_model_input(et_phi, t)
                #     et_phi = et_phi.detach()
                    
                #     et = self.unet(lt, t).sample
                #     et = self.scheduler.scale_model_input(et, t)
                #     et = et.detach()
                    
                # loss_noise = torch.mul((et - et_phi).detach(), latents)
                
                # v_prediction
                # with torch.no_grad():
                #     lt = alpha_t.sqrt() * latents + (1 - alpha_t).sqrt() * noise_xt
                    
                #     v_t_phi = unet_phi(lt, t).sample
                #     v_t_phi = self.scheduler.scale_model_input(v_t_phi, t)
                #     v_t_phi = v_t_phi.detach()
                    
                #     v_t = self.unet(lt, t).sample
                #     v_t = self.scheduler.scale_model_input(v_t, t)
                #     v_t = v_t.detach()
                    
                # loss_noise = torch.mul((v_t - v_t_phi).detach(), latents)
                
                with torch.no_grad():
                    lt = alpha_t.sqrt() * latents + (1 - alpha_t).sqrt() * noise_xt
                    
                    # v_t_phi = unet_phi(lt, t).sample
                    # v_t_phi = self.scheduler.scale_model_input(v_t_phi, t)
                    # v_t_phi = v_t_phi.detach()
                    
                    target = self.scheduler.get_velocity(latents, noise_xt, t)
                    
                    v_t = self.unet(lt, t).sample
                    v_t = self.scheduler.scale_model_input(v_t, t)
                    v_t = v_t.detach()
                    
                loss_noise = torch.mul((v_t - target).detach(), latents)
                    
                
                # with torch.no_grad():
                #     latents_repeat = repeat(latents, "b c h w -> (b k) c h w", k=50)
                #     noise_xt = torch.randn_like(latents_repeat).type_as(latents_repeat)
                    
                #     lt = alpha_t.sqrt() * latents_repeat + (1 - alpha_t).sqrt() * noise_xt
                    
                #     et_phi = unet_phi(lt, t).sample
                #     et_phi = self.scheduler.scale_model_input(et_phi, t)
                #     et_phi = et_phi.detach()
                    
                #     et = self.unet(lt, t).sample
                #     et = self.scheduler.scale_model_input(et, t)
                #     et = et.detach()
                    
                # loss_noise = torch.mul((et - et_phi).detach(), latents_repeat)
                    
                    # loss_noise = torch.exp(loss_noise)
                    # loss_noise = torch.mean(loss_noise, dim=0)
                    # loss_noise = torch.log(loss_noise)
                    
                snr_inv = (1 - alpha_t).sqrt() / alpha_t.sqrt()

                w_t = grad_term_weight 
                v_t = obs_weight
             
                # print('loss:', loss_noise.mean())
                # print(torch.any(variation_loss.isinf()), variation_loss)
                
                # 0.05 for 32 views 
                # total_loss = w_t * loss_noise.mean() + v_t * loss_diff_val + 0.05 * variation_loss
                
                total_loss = w_t * loss_noise.mean() + v_t * loss_diff_val
                
                grad = torch.autograd.grad(total_loss, latents)[0]
                # torch.nn.utils.clip_grad.clip_grad_value_(grad, 1)
                # current_scale = ((self.condition_module.scale - self.condition_module.scale * 1e-3) / 1000) * (1000 - t) + self.condition_module.scale * 1e-3
                # print(current_scale)
                l_prev = self.condition_module.optimizer(
                    # latents, grad, current_scale
                    latents, grad, self.condition_module.scale 
                )

                losses.append(l2_dist.mean().cpu().detach())
                ttt.set_postfix({'l2 loss:': l2_dist.mean().cpu().detach().numpy()})
                latents = l_prev.detach()
                
                clip_valve = torch.quantile(latents.abs(), q=0.95)
                latents = latents.clamp(-clip_valve, clip_valve) / clip_valve
                
                # for i in range(1):
                #     phi_optimizer.zero_grad()
                #     noise_x0 = torch.randn_like(latents).type_as(latents)
                #     # lt = alpha_t.sqrt() * latents + (1 - alpha_t).sqrt() * noise_x0
                #     noise_lt = self.scheduler.add_noise(latents, noise_x0, t)
                #     # clean_latents = self.scheduler.step(noise_x0, t, noise_lt).pred_original_sample
                #     target = self.scheduler.get_velocity(latents, noise_x0, t)
                #     model_output = unet_phi(noise_lt, t).sample
                #     model_output = self.scheduler.scale_model_input(model_output, t)
                #     et = model_output
                #     # et = et.detach()
                #     et_phi = et
                #     loss_noise_phi = loss_fn(et_phi, target)
                #     loss_noise_phi.backward()
                #     # torch.nn.utils.clip_grad.clip_grad_norm_(unet_phi.parameters(), max_norm=1)
                #     phi_optimizer.step()
                    
                    
                    # latents_repeat = repeat(latents, "b c h w -> (b k) c h w", k=10)
                    # phi_optimizer.zero_grad()
                    # noise_x0 = torch.randn_like(latents_repeat).type_as(latents_repeat)
                    # # lt = alpha_t.sqrt() * latents + (1 - alpha_t).sqrt() * noise_x0
                    # noise_lt = self.scheduler.add_noise(latents_repeat, noise_x0, t)
                    # # clean_latents = self.scheduler.step(noise_x0, t, noise_lt).pred_original_sample
                    # model_output = unet_phi(noise_lt, t).sample
                    # model_output = self.scheduler.scale_model_input(model_output, t)
                    # et = model_output
                    # # et = et.detach()
                    # et_phi = et
                    # loss_noise_phi = loss_fn(et_phi, noise_x0)
                    # loss_noise_phi.backward()
                    # # torch.nn.utils.clip_grad.clip_grad_norm_(unet_phi.parameters(), max_norm=1)
                    # phi_optimizer.step()
                
                # with torch.no_grad():
                #     x0_hat = self.condition_module.vae.decode(latents).sample
                #     x0_hat = self.optimize_latent(x0_hat, measurement, self.condition_module.degrade_op)
                #     latents = self.condition_module.vae.encode(x0_hat).latents
                    
                
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # image = kornia.filters.GaussianBlur2d((5, 5), (1.5, 1.5))(image)
        image = image.cpu().detach()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,), losses

        return ImagePipelineOutput(images=image)