from typing import Any, List

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from diffusers import SchedulerMixin, DDPMPipeline, DDIMPipeline, DiffusionPipeline
from torchvision.utils import make_grid

from src.models.components.vq_model_utils import VQLPIPSWithDiscriminator


class VAEModelLitModule(LightningModule):
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
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        lr_g_factor: float = 1.0,
        num_vq_embeddings: int = 512,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = model

        # self.opt_net = torch.compile(self.net)
        
        # loss function
        self.loss = VQLPIPSWithDiscriminator(
            disc_start=11000,
            disc_conditional=False,
            disc_in_channels=1,
            disc_weight=0.8,
            codebook_weight=1.0,
            n_classes=num_vq_embeddings,
            # perceptual_weight=0.0,
            # perceptual_loss=None
        )

        self.learning_rate = learning_rate
        self.lr_g_factor = lr_g_factor

        self.train_loss = MeanMetric()
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor, return_pred_indices=False):
        output = self.net(x)
        (_, _, ind) = output.info
        if return_pred_indices:
            return output.sample, output.emb_loss, ind
        return output.sample, output.emb_loss

    def training_step(self, batch, batch_idx):
        x = batch
        x = x.float()
        x = x.permute(0, 2, 1, 3, 4)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        g_opt, d_opt = self.optimizers()

        # if optimizer_idx == 0:
        aeloss, log_dict_ae = self.loss(
            qloss,
            x.permute(0, 2, 1, 3, 4).view(-1, 1, *x.shape[-2:]),
            xrec.permute(0, 2, 1, 3, 4).view(-1, 1, *x.shape[-2:]),
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
            predicted_indices=ind,
        )

        self.log_dict(
            log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True
        )
        # return aeloss
        self.manual_backward(aeloss)
        if (batch_idx + 1) % 8 == 0:
            g_opt.step()
            g_opt.zero_grad()

        # if optimizer_idx == 1:
        discloss, log_dict_disc = self.loss(
            qloss,
            x.permute(0, 2, 1, 3, 4).view(-1, 1, *x.shape[-2:]),
            xrec.permute(0, 2, 1, 3, 4).view(-1, 1, *x.shape[-2:]),
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        self.log_dict(
            log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True
        )
        self.manual_backward(discloss)
        if (batch_idx + 1) % 2 == 0:
            # print(d_opt.optimizer.param_groups[0]['lr'])
            d_opt.step()
            d_opt.zero_grad()

        self.log_dict({"g_loss": aeloss, "d_loss": discloss}, prog_bar=True)
        # return discloss


    def validation_step(self, batch, batch_idx):
        x = batch
        x = x.float()
        x = x.permute(0, 2, 1, 3, 4)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x.permute(0, 2, 1, 3, 4).view(-1, 1, *x.shape[-2:]), xrec.permute(0, 2, 1, 3, 4).view(-1, 1, *x.shape[-2:]), 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val",
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x.permute(0, 2, 1, 3, 4).view(-1, 1, *x.shape[-2:]), xrec.permute(0, 2, 1, 3, 4).view(-1, 1, *x.shape[-2:]), 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val",
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        del log_dict_ae[f"val/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

        if batch_idx == 0:
            original_grid_images = make_grid(x.permute(0, 2, 1, 3, 4).view(-1, 1, *x.shape[-2:]), nrow=8, normalize=True, range=(-1, 1))
            self.logger.experiment.add_image(
                "val/original",
                original_grid_images,
                self.current_epoch
            )
            reconstructed_grid_images = make_grid(xrec.permute(0, 2, 1, 3, 4).view(-1, 1, *x.shape[-2:]), nrow=8, normalize=True, range=(-1, 1))
            self.logger.experiment.add_image(
                "val/reconstructed",
                reconstructed_grid_images,
                self.current_epoch
            )
            
        # for param_group in self.optimizers()[0].optimizer.param_groups:
        #     param_group['lr'] = self.lr_g_factor*self.learning_rate
        
        # for param_group in self.optimizers()[1].optimizer.param_groups:
        #     param_group['lr'] = self.learning_rate   
        

        return self.log_dict
    
    def on_train_start(self) -> None:
        for param_group in self.optimizers()[0].optimizer.param_groups:
            param_group['lr'] = self.lr_g_factor*self.learning_rate
        
        for param_group in self.optimizers()[1].optimizer.param_groups:
            param_group['lr'] = self.learning_rate  
        return super().on_train_start()

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate

        opt_ae = torch.optim.AdamW(
            list(self.net.encoder.parameters()) +
            list(self.net.decoder.parameters()) + 
            list(self.net.quant_conv.parameters()) +
            list(self.net.quantize.parameters()) +
            list(self.net.post_quant_conv.parameters()), 
            lr=lr_g,
            betas=(0.5, 0.9)
        )
        opt_disc = torch.optim.AdamW(
            self.loss.discriminator.parameters(), 
            lr=lr_d, 
            betas=(0.5, 0.9)
        )

        return opt_ae, opt_disc

    def get_last_layer(self):
        return self.net.decoder.conv_out.weight

if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None)
