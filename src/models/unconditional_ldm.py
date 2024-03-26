import inspect
from typing import Any, List, Optional, OrderedDict, Tuple, Union

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, PeakSignalNoiseRatio
from torchvision.utils import make_grid
from torchmetrics.classification.accuracy import Accuracy
from diffusers import (
    SchedulerMixin,
    DDPMPipeline,
    DDIMPipeline,
    DiffusionPipeline,
    ImagePipelineOutput,
    AutoencoderKL,
)
from diffusers.utils import randn_tensor
from transformers import ViTMAEConfig, ViTMAEModel

from src.models.components.conditional_latent_ddpm import (
    ConditionalDDPMTrainModel,
    ConditionalLatentDDPMTrainModel,
    SemanticLossSchedule,
    UnConditionalLatentDDPMTrainModel,
)

class MyLatentPipeline(DiffusionPipeline):
    def __init__(self, unet, vae, scheduler, latents_height, latents_width, vae_type):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
        )

        self.latents_height = latents_height
        self.latents_width = latents_width
        self.vae_type = vae_type

    def __call__(
        self,
        batch,
        batch_size: int,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        scale_factor=1.0,
    ) -> Union[ImagePipelineOutput, Tuple]:
        # Sample gaussian noise to begin loop
        latents_shape = (
            batch_size,
            self.unet.in_channels,
            # 6,
            self.latents_height,
            self.latents_width,
        )
        latents = randn_tensor(
            latents_shape,
            generator=generator,
            device=self.device,
            dtype=self.unet.dtype,
        )

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # if guidance_scale == 1.0:
            # else:
            #     latents_input = torch.cat([latents, latents], dim=0)
            #     context = torch.cat([positive_condition, negetive_condition], dim=0)

            # 1. predict noise model_output
            model_output = self.unet(latents, t).sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            # if guidance_scale != 1.0:
            #     noise_pred_uncond, noise_prediction_sinogram = model_output.chunk(2)
            #     model_output = noise_pred_uncond + guidance_scale * (
            #         noise_prediction_sinogram - noise_pred_uncond
            #     )

            latents = self.scheduler.step(
                model_output,
                t,
                latents,
            ).prev_sample
            # print("ddpm:", latents.max(), latents.min())
            
        # if self.vae_type == "klvae":
            # image = self.vae.decode(1.0 / scale_factor * latents).sample
        # else:
        
        # latents = self.vae.encode(batch.float().permute(0, 2, 1, 3, 4)).latents
        # print("vae:", latents.max(), latents.min())
        
        latents = latents / scale_factor
        
        data = self.vae.decode(latents)
        image = data.sample
        # print(data.emb_loss)
        # print(batch.max(), batch.min())
        image = (image / 2 + 0.5).clamp(0, 1)
        # batch = (batch / 2 + 0.5).clamp(0, 1)
        # print(-torch.log10(torch.nn.functional.mse_loss(image, batch.float().permute(0, 2, 1, 3, 4))) * 10)
        # image = image.clamp(0, 1)
        image = image.cpu().numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


class LDMLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        unet: torch.nn.Module,
        vae: AutoencoderKL,
        vae_ckpt_path: str,
        semantic_loss: torch.nn.Module,
        semantic_loss_weight: float,
        gamma_schedule: SemanticLossSchedule,
        train_noise_scheduler: SchedulerMixin,
        inference_noise_scheduler: SchedulerMixin,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        prediction_type: str = "epsilon",
        inspection_batch_size: int = 16,
        ddpm_num_inference_steps: int = 1000,
        guidance_scale: float = 1.0,
        condition_discard_rate: float = 0.0,
        freeze_vae: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.vae = vae
        new_vae_state_dict = OrderedDict()
        for k, v in torch.load(vae_ckpt_path)["state_dict"].items():
            if "loss" not in k:
                new_vae_state_dict[k[4:]] = v
        self.vae.load_state_dict(new_vae_state_dict)
        if freeze_vae:
            for param in self.vae.parameters():
                param.requires_grad = False

        self.unet = unet
        self.net = UnConditionalLatentDDPMTrainModel(
            unet,
            vae,
            train_noise_scheduler,
        )
        # loss function
        self.criterion = torch.nn.MSELoss()

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.prediction_type = prediction_type
        self.train_noise_scheduler = train_noise_scheduler
        self.inference_noise_scheduler = inference_noise_scheduler
        self.inspection_batch_size = inspection_batch_size
        self.ddpm_num_inference_steps = ddpm_num_inference_steps
        self.guidance_scale = guidance_scale

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        # self.val_loss = MeanMetric()
        # self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        # self.val_acc_best = MaxMetric()
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)

        # self.scale_factor = 0.3859907388687134
        # self.register_buffer("scale_factor", torch.tensor(0.3859907388687134))

    def forward(self, x: torch.Tensor, scale_factor):
        return self.net(x)

    # def on_train_batch_start(self, batch: Any, batch_idx: int):
    #     if self.current_epoch == 0 and batch_idx == 0 and self.global_step == 0:
    #         x = batch.to(self.device).float()
    #         encoder_posterior = self.vae.encode(x).latent_dist
    #         z = encoder_posterior.sample()
    #         del self.scale_factor
    #         # print("z:", z.max(), z.min())
    #         # print("z_var:", z.flatten().var())
    #         # self.register_buffer("scale_factor", 1.0 / z.flatten().var())
    #         self.register_buffer("scale_factor", torch.tensor(5e-2))
    #         print(f"setting self.scale_factor to {self.scale_factor}")
    #         print("### USING STD-RESCALING ###")

    def training_step(self, batch: Any, batch_idx: int):
        x = batch.float()
        # x = x.permute(0, 2, 1, 3, 4)
        loss = self(x, None)

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}

    def val_inspection(self, batch):
        pipeline = MyLatentPipeline(
            self.unet,
            self.vae,
            self.inference_noise_scheduler,
            latents_height=32,
            latents_width=32,
            vae_type="vqvae"
        )

        pipeline.set_progress_bar_config(disable=True)

        generator = torch.Generator(device=pipeline.device).manual_seed(0)
        images = pipeline(
            batch,
            generator=generator,
            batch_size=self.inspection_batch_size,
            num_inference_steps=self.ddpm_num_inference_steps,
            output_type="numpy",
            scale_factor=0.35246044771165214,
        ).images

        # images_processed = (images * 255).round().astype("uint8")
        # images_processed = torch.from_numpy(images_processed).permute(0, 3, 1, 2)
        pred_grid = make_grid(
            # torch.from_numpy(images).permute(0, 2, 1, 3, 4).view(-1, 1, 256, 256),
            torch.from_numpy(images),
            nrow=8,
            value_range=(0, 1),
            normalize=True,
        )[None]
        self.logger.experiment.add_images(
            "val_inspection", pred_grid, self.current_epoch
        )

        # gt_grid = make_grid(x, nrow=8, normalize=True, value_range=(-1, 1))[None]
        # self.logger.experiment.add_images(
        #     "val_inspection_gt", gt_grid, self.current_epoch
        # )

        # loss = self.criterion(images, x)
        # self.log(
        #     "val_inspection/loss",
        #     loss,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )

        # images = (images / 2 + 0.5).clamp(0, 1)
        # x = (x / 2 + 0.5).clamp(0, 1)

        # psnr = self.val_psnr(images, x)
        # self.log(
        #     "val_inspection/psnr",
        #     psnr,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )

    def validation_step(self, batch: Any, batch_idx: int):
        if batch_idx == 0:
            self.val_inspection(batch)
            # pass

        # update and log metrics
        return {"loss": 0}

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(
                optimizer=optimizer,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None)
