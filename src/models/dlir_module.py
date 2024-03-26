import copy
import os
from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import (
    PeakSignalNoiseRatio,
    MultiScaleStructuralSimilarityIndexMeasure,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from src.models.components.guidence_modules import GenericGuidenceModule
from src.models.components.lora_utils.locon import LoConModule

def get_window(data, hu_max=256, hu_min=-150):
    # max_val = 3200
    # min_val = -2048
    # data = data * (max_val - min_val) + min_val
    # data = data.clip(hu_min, hu_max)
    # data = (data - hu_min) / (hu_max - hu_min)
    return data

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
        
        self.phi_param_list = [unet, vqvae, unet_ckpt_path, vqvae_ckpt_path]

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

        self.unet_phi, _ = self.restore_ckpt(
            unet, vqvae, unet_ckpt_path, vqvae_ckpt_path
        )

        self.unet_phi = self.unet_phi.cuda()
        
        LOCON_ENABLE = False

        if LOCON_ENABLE:
            def replace_layers(model, old, new):
                for n, module in model.named_children():
                    if len(list(module.children())) > 0:
                        ## compound module, go inside it
                        replace_layers(module, old, new)

                    if isinstance(module, old):
                        ## simple module
                        # setattr(model, n, new)
                        new_model = new(lora_name=None, org_module=module, lora_dim=32)
                        new_model.apply_to()
                        new_model.cuda()
                        setattr(model, n, new_model)
                        # print(n)

            replace_layers(self.unet_phi, torch.nn.Conv2d, LoConModule)

            for param in self.unet_phi.parameters():
                param.requires_grad_(False)
    
            self.phi_parameters = []
            for k, v in self.unet_phi.named_parameters():
                if 'lora' in k:
                    print(k)
                    self.phi_parameters += v
                    v.requires_grad_(True)
        else:
            for param in self.unet_phi.parameters():
                param.requires_grad_(True)
            self.phi_parameters = self.unet_phi.parameters()
        
        self.im_out_dir = im_out_dir
        os.makedirs(self.im_out_dir, exist_ok=True)

        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.msssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1)
        self.fid = FrechetInceptionDistance()
        self.lpip = LearnedPerceptualImagePatchSimilarity()
        
    def make_unet_phi(self, unet, vqvae, unet_ckpt_path, vqvae_ckpt_path):
        unet_phi, _ = self.restore_ckpt(
            copy.deepcopy(unet), copy.deepcopy(vqvae), unet_ckpt_path, vqvae_ckpt_path
        )

        unet_phi = unet_phi.cuda()
        
        # def reset_reset_parameters(m: torch.nn.Module):
        #     if hasattr(m, "reset_parameters"):
        #         m.reset_parameters()
        
        # unet_phi.apply(reset_reset_parameters)
        
        LOCON_ENABLE = False

        if LOCON_ENABLE:
            def replace_layers(model, old, new):
                for n, module in model.named_children():
                    if len(list(module.children())) > 0:
                        ## compound module, go inside it
                        replace_layers(module, old, new)

                    if isinstance(module, old):
                        ## simple module
                        # setattr(model, n, new)
                        new_model = new(lora_name=None, org_module=module, lora_dim=2)
                        new_model.apply_to()
                        new_model.cuda()
                        setattr(model, n, new_model)
                        # print(n)

            replace_layers(unet_phi, torch.nn.Conv2d, LoConModule)

            for param in unet_phi.parameters():
                param.requires_grad_(False)
    
            phi_parameters = []
            for k, v in unet_phi.named_parameters():
                if 'lora' in k:
                    phi_parameters += v
                    v.requires_grad_(True)
        else:
            for param in unet_phi.parameters():
                param.requires_grad_(True)
            phi_parameters = unet_phi.parameters()
            
        return unet_phi, phi_parameters

    def restore_ckpt(self, unet, vqvae, unet_ckpt_path, vqvae_ckpt_path):
        if vqvae_ckpt_path is not None:
            old_state_dict = torch.load(unet_ckpt_path, map_location="cpu")
            new_state_dict = {}
            for k, v in old_state_dict["state_dict"].items():
                if k.startswith("unet."):
                    new_state_dict[k[5:]] = v
            unet.load_state_dict(new_state_dict)
            old_state_dict = torch.load(vqvae_ckpt_path, map_location="cpu")
            new_state_dict = {}
            for k, v in old_state_dict["state_dict"].items():
                if k.startswith("vae."):
                    new_state_dict[k[4:]] = v
            vqvae.load_state_dict(new_state_dict)

            return unet, vqvae
        else:
            old_state_dict = torch.load(unet_ckpt_path, map_location="cpu")
            new_state_dict = {}
            for k, v in old_state_dict["state_dict"].items():
                if k.startswith("net."):
                    new_state_dict[k[4:]] = v
            unet.load_state_dict(new_state_dict)
            return unet, None

    def test_step(self, batch: Any, batch_idx: int):
        # if os.path.exists(os.path.join(self.im_out_dir, f"{batch_idx}.png")):
        #     return torch.tensor(0.0)

        # if batch_idx != 159:
        #     if batch_idx != 99:
        #         return torch.tensor(0.0)

        degrade_op = self.guidence_module.degrade_op
        noise_op = self.guidence_module.noise_op

        # print(batch.shape)
        # batch_down = self.downsample(batch.clone())
        
        # batch = (batch / 2 + 0.5).clamp(0, 1)
        # gt = batch
        measurement = degrade_op.forward(batch)
        print(batch.max(), batch.min())
        print(measurement.max(), measurement.min())
        if noise_op is not None:
            measurement = noise_op.forward(measurement)
        
        # del unet_phi, phi_parameters
        unet_phi, phi_parameters = self.make_unet_phi(*self.phi_param_list)
        # unet_phi = None
        # phi_parameters = None

        generator = torch.Generator().manual_seed(142)
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

