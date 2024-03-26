from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from diffusers.models import VQModel
from diffusers.utils import BaseOutput
from diffusers import UNet2DConditionModel, SchedulerMixin, UNet2DModel, DDPMScheduler
from diffusers.utils import randn_tensor
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
import numpy as np
from transformers.models.vit.modeling_vit import ViTEncoder
import torch
from torch import nn
from torch.nn import functional as F

class ConditionalLatentDDPMTrainModel(nn.Module):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        vqmodel: VQModel,
        conditional_model: ViTEncoder,
        scheduler: SchedulerMixin,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.vqmodel = vqmodel
        self.conditional_model = conditional_model
        self.scheduler = scheduler

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> BaseOutput:
        # construt positive and negative condition
        conditional_sinograms = y.float()
        positive_condition_state = self.conditional_model(
            conditional_sinograms
        ).last_hidden_state

        latents = self.vqmodel.encode(x).latents
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (x.shape[0],), device=x.device
        ).long()

        noise = torch.randn(latents.shape).type_as(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet(
            noisy_latents, timesteps, encoder_hidden_states=positive_condition_state
        ).sample

        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        return loss

class UnConditionalLatentDDPMTrainModel(nn.Module):
    def __init__(
        self,
        unet: UNet2DModel,
        vqmodel: VQModel,
        scheduler: SchedulerMixin,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.vqmodel = vqmodel
        self.scheduler = scheduler

    def forward(self, x: torch.Tensor) -> BaseOutput:
        # construt positive and negative condition
        
        # latents = self.vqmodel.encode(x).latents
        
        # for kl vae
        with torch.no_grad():
            latents = self.vqmodel.encode(x).latent_dist.sample() * 0.35246044771165214
        
        # print(latents.shape, latents.max(), latents.min())
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (x.shape[0],), device=x.device
        ).long()

        noise = torch.randn(latents.shape).type_as(latents)
        # noisy_latents = self.scheduler.add_noise(latents, noise, timesteps) + 0.1 * torch.randn_like(latents).type_as(latents)

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        target = self.scheduler.get_velocity(latents, noise, timesteps)
        # print("using get_velocity")
        
        noise_pred = self.unet(
            noisy_latents, timesteps
        ).sample

        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

        return loss

class ConditionalDDPMTrainModel(nn.Module):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        conditional_model: ViTEncoder,
        scheduler: SchedulerMixin,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.conditional_model = conditional_model
        self.scheduler = scheduler
        self.lpips = LPIPSwithpixelloss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> BaseOutput:
        # construt positive and negative condition
        conditional_sinograms = y.float()
        positive_condition_state = self.conditional_model(
            conditional_sinograms
        ).last_hidden_state

        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (x.shape[0],), device=x.device
        ).long()

        noise = torch.randn(x.shape).type_as(x)
        noisy_latents = self.scheduler.add_noise(x, noise, timesteps)

        noise_pred = self.unet(
            noisy_latents, timesteps, encoder_hidden_states=positive_condition_state
        ).sample

        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        # loss, log = self.lpips(noise_pred, noise)
        # loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean") + loss * 0.001

        return loss, {"total_loss": loss}

class SemanticLossSchedule:
    def __init__(self, num_train_timesteps, gamma_end, gamma_start, gamma_schedule) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.gamma_end = gamma_end
        self.gamma_start = gamma_start
        self.gamma_schedule = gamma_schedule

class PatchedDDPMScheduler(DDPMScheduler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        **kwargs,
    ):
        super().__init__(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            trained_betas,
            variance_type,
            clip_sample,
            prediction_type,
            **kwargs,
        )

    def _get_variance(self, t, predicted_variance=None, variance_type=None):
        num_inference_steps = (
            self.num_inference_steps
            if self.num_inference_steps
            else self.config.num_train_timesteps
        )
        prev_t = t - self.config.num_train_timesteps // num_inference_steps
        alpha_prod_t = self.alphas_cumprod[t][:, None, None, None]
        # alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        alpha_prod_t_prev = torch.where(
            prev_t > 0,
            self.alphas_cumprod[prev_t],
            torch.tensor(1.0).type_as(self.alphas_cumprod[prev_t]),
        )[:, None, None, None]
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        if variance_type is None:
            variance_type = self.config.variance_type

        # hacks - were probably added for training stability
        if variance_type == "fixed_small":
            variance = torch.clamp(variance, min=1e-20)
        # for rl-diffuser https://arxiv.org/abs/2205.09991
        elif variance_type == "fixed_small_log":
            variance = torch.log(torch.clamp(variance, min=1e-20))
            variance = torch.exp(0.5 * variance)
        elif variance_type == "fixed_large":
            variance = current_beta_t
        elif variance_type == "fixed_large_log":
            # Glide max_log
            variance = torch.log(current_beta_t)
        elif variance_type == "learned":
            return predicted_variance
        elif variance_type == "learned_range":
            min_log = torch.log(variance)
            max_log = torch.log(self.betas[t])
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log

        return variance

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: torch.FloatTensor,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        # message = (
        #     "Please make sure to instantiate your scheduler with `prediction_type` instead. E.g. `scheduler ="
        #     " DDPMScheduler.from_pretrained(<model_id>, prediction_type='epsilon')`."
        # )
        # predict_epsilon = deprecate(
        #     "predict_epsilon", "0.13.0", message, take_from=kwargs
        # )
        # if predict_epsilon is not None:
        #     new_config = dict(self.config)
        #     new_config["prediction_type"] = "epsilon" if predict_epsilon else "sample"
        #     self._internal_dict = FrozenDict(new_config)

        t = timestep

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in [
            "learned",
            "learned_range",
        ]:
            model_output, predicted_variance = torch.split(
                model_output, sample.shape[1], dim=1
            )
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t][:, None, None, None]
        # alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else self.one
        alpha_prod_t_prev = torch.where(
            t > 0,
            self.alphas_cumprod[t - 1],
            torch.tensor(1.0).type_as(self.alphas_cumprod[t - 1]),
        )[:, None, None, None]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        print("patched", model_output.shape, beta_prod_t.shape)
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        self.betas = self.betas.type_as(alpha_prod_t_prev)
        self.alphas = self.alphas.type_as(alpha_prod_t_prev)
        pred_original_sample_coeff = (
            alpha_prod_t_prev ** (0.5) * self.betas[t][:, None, None, None]
        ) / beta_prod_t
        current_sample_coeff = (
            self.alphas[t][:, None, None, None] ** (0.5)
            * beta_prod_t_prev
            / beta_prod_t
        )

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * sample
        )

        # 6. Add noise
        variance = 0
        # if t > 0:
        device = model_output.device
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=device,
            dtype=model_output.dtype,
        )
        if self.variance_type == "fixed_small_log":
            variance = (
                self._get_variance(t, predicted_variance=predicted_variance)
                * variance_noise
            )
        else:
            variance = (
                self._get_variance(t, predicted_variance=predicted_variance) ** 0.5
            ) * variance_noise

        pred_prev_sample = pred_prev_sample + variance * torch.where(
            (t > 0)[:, None, None, None],
            torch.tensor(1.0).type_as(alpha_prod_t_prev),
            torch.tensor(0.0).type_as(alpha_prod_t_prev),
        )

        if not return_dict:
            return (pred_prev_sample,)

        return DDPMSchedulerOutput(
            prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample
        )

if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    model = VQTrainModel(
        in_channels=1,
        out_channels=1,
        down_block_types=(
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(64, 128, 256),
        layers_per_block=2,
    )
    model.eval()
    x = torch.randn(1, 1, 128, 128)
    out = model(x)
    print(out.sample.shape, out.emb_loss.shape)