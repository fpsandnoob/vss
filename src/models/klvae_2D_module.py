from typing import Any, List

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from diffusers import SchedulerMixin, DDPMPipeline, DDIMPipeline, DiffusionPipeline
from torchvision.utils import make_grid
from transformers import get_cosine_schedule_with_warmup, Adafactor

from src.models.components.kl_model_utils import LPIPSWithDiscriminator


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
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        # self.net = torch.compile(model, mode='reduce-overhead')
        self.net = model

        # self.opt_net = torch.compile(self.net)
        
        # loss function
        self.loss = LPIPSWithDiscriminator(
            disc_start=45001,
            kl_weight=1e-5,
            disc_conditional=False,
            disc_in_channels=1,
            disc_weight=0.5,
            # perceptual_weight=0.0,
            # perceptual_loss=None
        )

        self.learning_rate = learning_rate
        self.lr_g_factor = lr_g_factor

        self.train_loss = MeanMetric()
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor):
        output = self.net(x, sample_posterior=True)
        return output.sample, output.posterior

    def training_step(self, batch, batch_idx):
        x = batch
        x = x.float()
        # x = x.permute(0, 2, 1, 3, 4)
        sample, posterior = self(x)

        g_opt, d_opt = self.optimizers()
        
        g_sch = self.lr_schedulers()

        # if optimizer_idx == 0:
        if batch_idx % 2 == 0:
            aeloss, log_dict_ae = self.loss(
                x.view(-1, 1, *x.shape[-2:]),
                sample.view(-1, 1, *x.shape[-2:]),
                posterior,
                0,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )

            self.log_dict(
                log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True
            )
            # return aeloss
            self.manual_backward(aeloss)
            
            # if (batch_idx + 1) % 2 == 0:
            g_opt.zero_grad()
            g_opt.step()
            g_sch.step()
            self.log_dict({"g_loss": aeloss, "g_lr": g_sch.get_last_lr()[0]}, prog_bar=True)

        # if optimizer_idx == 1:\
        else:
            discloss, log_dict_disc = self.loss(
                x.view(-1, 1, *x.shape[-2:]),
                sample.view(-1, 1, *x.shape[-2:]),
                posterior,
                1,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train"
            )
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True
            )
            self.manual_backward(discloss)
            # if (batch_idx + 1) % 2 == 0:
                # print(d_opt.optimizer.param_groups[0]['lr'])
            d_opt.zero_grad()
            d_opt.step()  

            self.log_dict({"d_loss": discloss, "g_lr": g_sch.get_last_lr()[0]}, prog_bar=True)
        # return discloss


    def validation_step(self, batch, batch_idx):
        x = batch
        x = x.float()
        # x = x.permute(0, 2, 1, 3, 4)
        sample, posterior = self(x)
        aeloss, log_dict_ae = self.loss(
                                        x.view(-1, 1, *x.shape[-2:]),
                                        sample.view(-1, 1, *x.shape[-2:]),
                                        posterior,
                                        0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val",
                                        )

        discloss, log_dict_disc = self.loss(
                                            x.view(-1, 1, *x.shape[-2:]),
                                            sample.view(-1, 1, *x.shape[-2:]),
                                            posterior,
                                            1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val",
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
            original_grid_images = make_grid(x.view(-1, 1, *x.shape[-2:]), nrow=8, normalize=True, value_range=(-1, 1))
            self.logger.experiment.add_image(
                "val/original",
                original_grid_images,
                self.current_epoch
            )
            reconstructed_grid_images = make_grid(sample.view(-1, 1, *x.shape[-2:]), nrow=8, normalize=True, value_range=(-1, 1))
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

        opt_ae = Adafactor(
            list(self.net.encoder.parameters()) +
            list(self.net.decoder.parameters()) + 
            list(self.net.quant_conv.parameters()) +
            list(self.net.post_quant_conv.parameters()), 
            lr=lr_g,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
            # betas=(0.5, 0.9)
        )
        opt_disc = Adafactor(
            self.loss.discriminator.parameters(), 
            lr=lr_d, 
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
            # betas=(0.5, 0.9)
        )

        # return (
        #     {
        #         "optimizer": opt_ae,
        #         "lr_scheduler": {
        #             "scheduler": get_cosine_schedule_with_warmup(opt_ae, 5000, num_training_steps=self.trainer.estimated_stepping_batches)
        #         }
        #     },
        #     opt_disc
        # )
        return [opt_ae, opt_disc], [get_cosine_schedule_with_warmup(opt_ae, 5000, num_training_steps=self.trainer.estimated_stepping_batches)]

    def get_last_layer(self):
        return self.net.decoder.conv_out.weight

if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None)
