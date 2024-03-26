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
import torchvision

from src.models.components.guidence_modules import GenericGuidenceModule
from src.models.components.measurement_ops import PhaseRetrievalOperator, QuantizerOperator


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
        guidence_module: GenericGuidenceModule,
        pipeline: DiffusionPipeline,
        unet: torch.nn.Module,
        vqvae: torch.nn.Module,
        unet_ckpt_path: str,
        vqvae_ckpt_path: str,
        im_out_dir: str,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # self.save_hyperparameters(logger=False)

        if vqvae is not None:
            self.unet, self.vqvae = self.restore_ckpt(
                unet, vqvae, unet_ckpt_path, vqvae_ckpt_path
            )
        else:
            self.unet, self.vqvae = self.restore_ckpt(unet, None, unet_ckpt_path, None)

        if vqvae is not None:
            self.guidence_module = guidence_module(vae=self.vqvae)
        else:
            self.guidence_module = guidence_module

        if vqvae is not None:
            self.pipeline = pipeline(
                unet=self.unet, vae=self.vqvae, condition_module=self.guidence_module
            )
        else:
            self.pipeline = pipeline(
                unet=self.unet, condition_module=self.guidence_module
            )

        self.im_out_dir = im_out_dir
        os.makedirs(self.im_out_dir, exist_ok=True)
        
        self.pred_psnr = PeakSignalNoiseRatio(data_range=1)
        self.pred_msssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1)
        self.pred_fid = FrechetInceptionDistance(normalize=True)
        self.pred_lpip = LearnedPerceptualImagePatchSimilarity()
        
        self.deg_psnr = PeakSignalNoiseRatio(data_range=1)
        self.deg_msssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1)
        self.deg_fid = FrechetInceptionDistance(normalize=True)
        self.deg_lpip = LearnedPerceptualImagePatchSimilarity()

    def restore_ckpt(self, unet, vqvae, unet_ckpt_path, vqvae_ckpt_path):
        if vqvae_ckpt_path is not None:
            state_dict = torch.load(unet_ckpt_path, map_location="cpu")
            unet.load_state_dict(state_dict)
            state_dict = torch.load(vqvae_ckpt_path, map_location="cpu")
            vqvae.load_state_dict(state_dict)

            return unet, vqvae
        else:
            state_dict = torch.load(unet_ckpt_path, map_location="cpu")
            unet.load_state_dict(state_dict)
            return unet, None

    def test_step(self, batch: Any, batch_idx: int):
        degrade_op = self.guidence_module.degrade_op
        noise_op = self.guidence_module.noise_op

        measurement = degrade_op.forward(batch)
        if noise_op is not None:
            measurement = noise_op.forward(measurement).clamp(0, 1)
            
        generator = torch.Generator().manual_seed(3614)
        if isinstance(degrade_op, (PhaseRetrievalOperator, QuantizerOperator)):
            image, losses = self.pipeline(
                measurement,
                generator=generator,
                num_inference_steps=1000,
                output_type="numpy",
                return_dict=False,
            )
        else:
            image, losses = self.pipeline(
                measurement * 2 - 1,
                generator=generator,
                num_inference_steps=1000,
                output_type="numpy",
                return_dict=False,
            )
        image = image[0]

        for im in image:
            # if im.shape[0] == 1:
            #     image = image.squeeze(0)
            # write_png(im, os.path.join(self.im_out_dir, f"{batch_idx}.png"))
            torch.save(im, os.path.join(self.im_out_dir, f"{batch_idx}.pt"))

        # gt = (batch / 2 + 0.5).clamp(0, 1)
        gt = batch
        for gt_im in gt:
            # if gt_im.shape[0] == 1:
            #     gt_im = gt_im.squeeze(0)
            # write_png(gt_im, os.path.join(self.im_out_dir, f"{batch_idx}_gt.png"))
            torch.save(gt_im, os.path.join(self.im_out_dir, f"{batch_idx}_gt.pt"))

        image = image.cuda()
        gt = gt.type_as(image)
        print("PSNR: ", self.pred_psnr(image, gt))
        print("SSIM: ", self.pred_msssim(image, gt))
        # self.pred_fid.update(image, real=False)
        # self.pred_fid.update(gt, real=True)
        # self.pred_fid.update(gt, real=True)
        # print("FID: ", self.pred_fid.compute())
        print("LPIPS: ", self.pred_lpip((image * 2 - 1), (gt * 2 - 1)))
        
        # print("PSNR: ", self.deg_psnr(degrade_op.backproject(measurement), gt))
        # print("SSIM: ", self.deg_msssim(degrade_op.backproject(measurement), gt))
        # # self.deg_fid.update(degrade_op.backproject(measurement), real=False)
        # # self.deg_fid.update(gt, real=True)
        # # self.deg_fid.update(gt, real=True)
        # # print("FID: ", self.deg_fid.compute())
        # print("LPIPS: ", self.deg_lpip((degrade_op.backproject(measurement) * 2 - 1), (gt * 2 - 1)))
        
        
        plt.subplot(1, 4, 1)
        plt.imshow(image[0].cpu().permute(1, 2, 0), cmap="gray")
        # plt.axis("off")
        plt.subplot(1, 4, 2)
        plt.imshow(gt[0].cpu().permute(1, 2, 0), cmap="gray")
        # plt.axis("off")
        plt.subplot(1, 4, 3)
        plt.imshow(measurement[0].cpu().permute(1, 2, 0))
        # plt.axis("off")
        plt.subplot(1, 4, 4)
        plt.plot(torch.stack(losses).cpu())
        plt.yscale("log")
        plt.savefig(os.path.join(self.im_out_dir, f"{batch_idx}.png"))
        plt.close()
        

        return torch.stack(losses).mean

    def on_test_epoch_end(self):
        pass

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


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None)
