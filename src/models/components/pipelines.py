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
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
import torch
from kornia.losses import total_variation
from src.models.components.measurement_ops import RandomBoxInpaintingOP
from einops import rearrange, repeat, reduce
from torchvision.utils import save_image
from src.models.components.utils import get_window

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

        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().detach()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,), losses

        return ImagePipelineOutput(images=image)


class LatentReconstructionUNetVSSPipeline(DiffusionPipeline):
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
    
    def save_intermediate_images(self, latents, t, output_dir):
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # image = sharpness(image, 0.5)
        # image = kornia.filters.GaussianBlur2d((5, 5), (1.5, 1.5))(image)
        image = image.cpu().detach()
        
        image = get_window(image)
        
        save_image(image, output_dir + f"/{t}.png")

    def __call__(
        self,
        measurement,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        grad_term_weight: float = 1,
        obs_weight: float = 0.00075,
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
        
        phi_optimizer = torch.optim.AdamW(phi_weights, lr=1e-4)
        
        latents = self.condition_module.vae.encode(self.condition_module.degrade_op.backproject(measurement)).latents * 1
        
        loss_fn = torch.nn.MSELoss()

        losses = []
        
        try:
            with torch.enable_grad():
                for t in self.progress_bar(self.scheduler.timesteps):
                    
                    if t < 5:
                        inner_iter = 20
                    elif t < 50:
                        inner_iter = 10
                    else:
                        inner_iter = 1
                    
                    for _ in range(inner_iter):
       
                        latents.requires_grad = True
                        prev_latents = latents
                        
                        output = self.condition_module.vae.decode(latents, force_not_quantize=False)

                        
                        x0_hat = output.sample
                        deg_x0_hat = self.condition_module.degrade_op.forward(x0_hat)
                        loss_diff_val = torch.linalg.vector_norm(measurement - deg_x0_hat, ord=1.6)
                        
                
                        alpha_t = self.scheduler.alphas_cumprod[t]

                        noise_xt = torch.randn_like(latents).type_as(latents)
                        
                        # with torch.no_grad():
                        #     # lt = alpha_t.sqrt() * latents + (1 - alpha_t).sqrt() * noise_xt
                            
                        #     lt = self.scheduler.add_noise(latents, noise_xt, timesteps=t)
                            
                        #     # et_phi = unet_phi(lt, t).sample
                        #     # et_phi = self.scheduler.scale_model_input(et_phi, t)
                        #     # et_phi = et_phi.detach()
                            
                        #     et = self.unet(lt, timestep=t, return_dict=True).sample
                        #     et = self.scheduler.scale_model_input(et, t)
                        #     et = et.detach()
                            
                        # loss_noise = torch.mul((et - noise_xt).detach(), latents).mean()

                        with torch.no_grad():
                            lt = alpha_t.sqrt() * latents + (1 - alpha_t).sqrt() * noise_xt 
                            
                            et_phi = unet_phi(lt, timestep=t).sample
                            et_phi = self.scheduler.scale_model_input(et_phi, t)
                            et_phi = et_phi.detach()
                            
                            et = self.unet(lt, timestep=t).sample
                            et = self.scheduler.scale_model_input(et, t)
                            et = et.detach()
                            
                        loss_noise = torch.mul((et - et_phi).detach(), latents).mean()
                        
                        snr_inv = (1 - alpha_t).sqrt() / alpha_t.sqrt()

                        w_t = grad_term_weight  * snr_inv 
                        v_t = obs_weight
                        
                        total_loss =  w_t * loss_noise.mean() + v_t * loss_diff_val
                        
                        grad = torch.autograd.grad(total_loss, latents)[0]
                        

                        latents = self.condition_module.optimizer(
                            latents, grad, self.condition_module.scale 
                        )
                        
                        losses.append(loss_diff_val.mean())
                        clip_valve = torch.quantile(latents.abs(), q=0.997)
                        latents = latents.clamp(-clip_valve, clip_valve)
                        
                        
                        latents = latents.detach()
                        
                        for i in range(1):
                            phi_optimizer.zero_grad()
                            noise_x0 = torch.randn_like(latents).type_as(latents)
                            noise_lt = self.scheduler.add_noise(latents, noise_x0, t)
                            model_output = unet_phi(noise_lt, timestep=t).sample
                            model_output = self.scheduler.scale_model_input(model_output, t)
                            et = model_output
                            et_phi = et
                            loss_noise_phi = loss_fn(et_phi, noise_x0)
                            loss_noise_phi.backward()
                            phi_optimizer.step()
                    
        finally:
            pass
                    
                
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().detach()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,), losses

        return ImagePipelineOutput(images=image)