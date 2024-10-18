import copy
import os
from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import (
    MaxMetric,
    MeanMetric,
    # FrechetInceptionDistance,
    # LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    MultiScaleStructuralSimilarityIndexMeasure,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.classification.accuracy import Accuracy
from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnProcessor,
    SlicedAttnAddedKVProcessor,
)
from diffusers.loaders import AttnProcsLayers
from diffusers import UNet2DModel

from src.models.components.guidance_module import GenericGuidanceModule
from src.models.components.vq_model import CustomResidualVQ
from src.models.components.utils import get_window

class EvalModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        guidance_module: GenericGuidanceModule,
        pipeline: DiffusionPipeline,
        diffusion_mdoel: torch.nn.Module,
        vqvae: torch.nn.Module,
        ckpt_path: str,
        im_out_dir: str,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # self.save_hyperparameters(logger=False)
        
        self.phi_param_list = [diffusion_mdoel, vqvae, ckpt_path]


        self.unet, self.vqvae = self.restore_ckpt(
            diffusion_mdoel, vqvae, ckpt_path
        )

        self.guidance_module = guidance_module(vae=self.vqvae, noise_op=None)

        self.pipeline = pipeline(
            unet=self.unet, vae=self.vqvae, condition_module=self.guidance_module
        )
        self.downsample = torch.nn.Upsample(scale_factor=0.5)
        self.upsample = torch.nn.Upsample(scale_factor=2)

        self.diffusion_model_phi, self.phi_parameters = self.make_diffusion_model_phi(
            diffusion_mdoel, vqvae, ckpt_path
        )

        self.diffusion_model_phi = self.diffusion_model_phi.cuda()
        
        self.im_out_dir = im_out_dir
        os.makedirs(self.im_out_dir, exist_ok=True)

        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.msssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1)
        self.fid = FrechetInceptionDistance()
        self.lpip = LearnedPerceptualImagePatchSimilarity()
        
    def make_diffusion_model_phi(self, diffusion_model, vqvae, ckpt_path):
        diffusion_model_phi, _ = self.restore_ckpt(
            copy.deepcopy(diffusion_model), copy.deepcopy(vqvae), ckpt_path
        )

        diffusion_model_phi = diffusion_model_phi.cuda()
        
        for param in diffusion_model_phi.parameters():
            param.requires_grad_(True)
        phi_parameters = diffusion_model_phi.parameters()
            
        return diffusion_model_phi, phi_parameters

    def restore_ckpt(self, diffusion_model, vqvae, ckpt_path):
        vae_ckpt = {}
        diffusion_model_ckpt = {}

        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        
        for k, v in ckpt.items():
            if k.startswith("vae."):
                vae_ckpt[k[4:]] = v
            if k.startswith("unet."):
                diffusion_model_ckpt[k[5:]] = v
                
        vqvae.load_state_dict(vae_ckpt, strict=False)
        diffusion_model.load_state_dict(diffusion_model_ckpt)

        # vqvae = vqvae.eval()
        # diffusion_model = diffusion_model.eval()
        
        return diffusion_model, vqvae

    def test_step(self, batch: Any, batch_idx: int):
        # if os.path.exists(os.path.join(self.im_out_dir, f"{batch_idx}.png")):
        #     return torch.tensor(0.0)

        # if batch_idx != 65:
        if batch_idx != 451:
            # if batch_idx != 99:
            return torch.tensor(0.0)

        degrade_op = self.guidance_module.degrade_op

        # print(batch.shape)
        # batch_down = self.downsample(batch.clone())
        
        # batch = (batch / 2 + 0.5).clamp(0, 1)
        # gt = batch
        measurement = degrade_op.forward(batch)
        print(batch.max(), batch.min())
        print(measurement.max(), measurement.min())
        
        # del unet_phi, phi_parameters
        unet_phi, phi_parameters = self.make_diffusion_model_phi(*self.phi_param_list)
        # unet_phi = None
        # phi_parameters = None

        generator = torch.Generator().manual_seed(114514)
        image, losses = self.pipeline(
            measurement.float(),
            generator=generator,
            num_inference_steps=1000,
            output_type="numpy",
            return_dict=False,
            unet_phi=unet_phi,
            phi_weights=phi_parameters
        )
        image = image[0]
        
        # image = self.upsample(image)
        # print(image.shape)
        
        for im in image:
            # if im.shape[0] == 1:
            #     image = image.squeeze(0)
            save_image(im, os.path.join(self.im_out_dir, f"{batch_idx}.png"))
            save_image(get_window(im), os.path.join(self.im_out_dir, f"{batch_idx}_w.png"))
            torch.save(im, os.path.join(self.im_out_dir, f"{batch_idx}.pt"))
            
        gt = (batch / 2 + 0.5).clamp(0, 1)
        print(gt.shape)
        for gt_im in gt:
            # if gt_im.shape[0] == 1:
            #     gt_im = gt_im.squeeze(0)
            # write_png(gt_im, os.path.join(self.im_out_dir, f"{batch_idx}_gt.png"))
            save_image(gt_im, os.path.join(self.im_out_dir, f"{batch_idx}_gt.png"))
            save_image(get_window(gt_im), os.path.join(self.im_out_dir, f"{batch_idx}_wgt.png"))
            torch.save(gt_im, os.path.join(self.im_out_dir, f"{batch_idx}_gt.pt"))

        # print(torch.stack(losses).shape)
        torch.save(
            torch.stack(losses).cpu(),
            os.path.join(self.im_out_dir, f"{batch_idx}_losses.pt"),
        )

        image = image.cuda()
        gt = gt.type_as(image)
        print(image.max(), image.min(), gt.max(), gt.min())
        # print(image.device, gt.device)
        # print(image.shape, gt.shape)
        print("PSNR: ", self.psnr(image, gt))
        print("SSIM: ", self.msssim(image, gt))
        # print("FID: ", self.fid((image / 255.).type(torch.uint8).repeat(1, 3, 1, 1), (gt / 255.).type(torch.uint8).repeat(1, 3, 1, 1)))
        print(
            "LPIPS: ",
            self.lpip(
                (image * 2 - 1).repeat(1, 3, 1, 1), (gt * 2 - 1).repeat(1, 3, 1, 1)
            ),
        )

        plt.subplot(1, 5, 1)
        plt.imshow(image[0].cpu().permute(1, 2, 0), vmin=0, vmax=1, cmap="gray")
        plt.subplot(1, 5, 2)
        plt.imshow(get_window(image[0]).cpu().permute(1, 2, 0), vmin=0, vmax=1, cmap="gray")
        plt.subplot(1, 5, 3)
        plt.imshow(gt[0].cpu().permute(1, 2, 0), vmin=0, vmax=1, cmap="gray")
        plt.subplot(1, 5, 4)
        plt.imshow(get_window(gt[0]).cpu().permute(1, 2, 0), vmin=0, vmax=1, cmap="gray")
        plt.subplot(1, 5, 5)
        plt.plot(torch.stack(losses).cpu())
        plt.yscale("log")
        plt.savefig(os.path.join(self.im_out_dir, f"{batch_idx}.png"))
        plt.close()

        return torch.stack(losses).mean()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        # optimizer = self.hparams.optimizer(params=self.parameters())
        # if self.hparams.scheduler is not None:
        #     scheduler = self.hparams.scheduler(optimizer=optimizer)
        #     return {
        #         "optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler": scheduler,
        #             "monitor": "val/loss",
        #             "interval": "epoch",
        #             "frequency": 1,
        #         },
        #     }
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {"optimizer": optimizer}