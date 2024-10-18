from dataclasses import dataclass
import random
import time
from typing import Callable, Tuple
from diffusers.models import VQModel
from diffusers.models.autoencoders.vae import DecoderOutput, VectorQuantizer
from einops import pack, rearrange, repeat, unpack
from torch import einsum
from vector_quantize_pytorch import ResidualFSQ, ResidualVQ
from vector_quantize_pytorch.vector_quantize_pytorch import EuclideanCodebook, CosineSimCodebook, gumbel_sample
from diffusers.utils import BaseOutput
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn import cluster

from collections import namedtuple
import functools
import hashlib
import os
import requests
import torch.distributed
from tqdm import tqdm
from torchvision import models
import torch.nn.functional as F
import torch.distributed as dist
from src.utils import (
    RankedLogger,
)


log = RankedLogger(__name__, rank_zero_only=False)


class ActNorm(nn.Module):
    def __init__(
        self, num_features, logdet=False, affine=True, allow_reverse_init=False
    ):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


URL_MAP = {"vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"}

CKPT_MAP = {"vgg_lpips": "vgg.pth"}

MD5_MAP = {"vgg_lpips": "d507d7349b931f0638a25a48a722f98a"}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "taming/modules/autoencoder/lpips")
        self.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu")), strict=False
        )
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu")), strict=False
        )
        return model

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(
                outs1[kk]
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [
            spatial_average(lins[kk].model(diffs[kk]), keepdim=True)
            for kk in range(len(self.chns))
        ]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


def hinge_d_loss_with_exemplar_weights(logits_real, logits_fake, weights):
    assert weights.shape[0] == logits_real.shape[0] == logits_fake.shape[0]
    loss_real = torch.mean(F.relu(1.0 - logits_real), dim=[1, 2, 3])
    loss_fake = torch.mean(F.relu(1.0 + logits_fake), dim=[1, 2, 3])
    loss_real = (weights * loss_real).sum() / weights.sum()
    loss_fake = (weights * loss_fake).sum() / weights.sum()
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def measure_perplexity(predicted_indices, n_embed):
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use


def compute_cluster_usage(predicted_indices, n_embed):
    """
    Computes the cluster usage and cluster usage ratio.

    Args:
        predicted_indices (torch.Tensor): Tensor containing the predicted indices.
        n_embed (int): Number of embeddings.

    Returns:
        tuple: A tuple containing the cluster usage and cluster usage ratio.
    """
    unique = torch.unique(predicted_indices)
    cluster_use = len(unique)
    return cluster_use, cluster_use / n_embed


# def l1(x, y):
#     return torch.linalg.vector_norm(x-y, ord=1)


# def l2(x, y):
#     return torch.linalg.vector_norm(x-y, ord=2)


def l1(x, y):
    return torch.abs(x - y)


def l2(x, y):
    return torch.pow((x - y), 2)


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_ndf=64,
        disc_loss="hinge",
        n_classes=None,
        perceptual_loss="lpips",
        pixel_loss="l1",
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        # assert perceptual_loss in ["lpips", "clips", "dists"]
        assert pixel_loss in ["l1", "l2"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        if perceptual_loss == "lpips":
            print(f"{self.__class__.__name__}: Running with LPIPS.")
            self.perceptual_loss = LPIPS().eval()
        else:
            # raise ValueError(f"Unknown perceptual loss: >> {perceptual_loss} <<")
            self.perceptual_loss = None
        self.perceptual_weight = perceptual_weight

        if pixel_loss == "l1":
            self.pixel_loss = l1
        else:
            self.pixel_loss = l2

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm,
            ndf=disc_ndf,
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.n_classes = n_classes

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        codebook_loss,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None,
        split="train",
        predicted_indices=None,
    ):
        if codebook_loss is None:
            codebook_loss = torch.tensor([0.0]).to(inputs.device)
        # rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        rec_loss = self.pixel_loss(inputs.contiguous(), reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous()
            )
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0]).type_as(inputs)

        nll_loss = rec_loss
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous(), cond), dim=1)
                )
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(
                    nll_loss, g_loss, last_layer=last_layer
                )
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            loss = (
                nll_loss
                + d_weight * disc_factor * g_loss
                + self.codebook_weight * codebook_loss.mean()
            )

            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/p_loss".format(split): p_loss.detach().mean(),
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach().mean(),
            }
            if predicted_indices is not None:
                assert self.n_classes is not None
                with torch.no_grad():
                    perplexity, cluster_usage = measure_perplexity(
                        predicted_indices, self.n_classes
                    )
                log[f"{split}/perplexity"] = perplexity
                log[f"{split}/cluster_usage"] = cluster_usage

                perplexity_loss = 0.1 * ((self.n_classes - perplexity) / self.n_classes)
                loss = loss + perplexity_loss

                log[f"{split}/perplexity_loss"] = perplexity_loss.detach().mean()

            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat((inputs.contiguous().detach(), cond), dim=1)
                )
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous().detach(), cond), dim=1)
                )

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean(),
            }
            return d_loss, log


@dataclass
class VQEncoderOutput(BaseOutput):
    latents: torch.Tensor


@dataclass
class VQDecoderOutput(BaseOutput):
    sample: torch.Tensor
    emb_loss: torch.Tensor
    info: Tuple
    entropy: torch.Tensor = None


class ReservoirSampler(nn.Module):
    def __init__(self, num_samples=1024):
        super(ReservoirSampler, self).__init__()
        self.n = num_samples
        self.ttot = 0
        self.register_buffer("ReservoirSampler_Buffer", None)
        self.reset()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        buffer_key = prefix + "ReservoirSampler_Buffer"
        if buffer_key in state_dict:
            self.buffer = state_dict[buffer_key]
        return super(ReservoirSampler, self)._load_from_state_dict(
            state_dict, prefix, *args, **kwargs
        )

    def reset(self):
        self.i = 0
        self.buffer = None

    def add(self, samples):
        self.ttot -= time.time()
        samples = samples.detach()
        if self.buffer is None:
            self.buffer = torch.empty(self.n, samples.size(-1), device=samples.device)
        buffer = self.buffer
        if self.i < self.n:
            slots = self.n - self.i
            add_samples = samples[:slots]
            samples = samples[slots:]
            buffer[self.i : self.i + len(add_samples)] = add_samples
            self.i += len(add_samples)
            if not len(samples):
                # print(f"Res size {self.i}")
                self.ttot += time.time()
                return

        for s in samples:
            # warning, includes right end too.
            idx = random.randint(0, self.i)
            self.i += 1
            if idx < len(buffer):
                buffer[idx] = s
        self.ttot += time.time()

    def contents(self):
        return self.buffer[: self.i]

@dataclass
class MultiScaleRQOutput(BaseOutput):
    sample: torch.Tensor
    emb_loss: torch.Tensor
    info: torch.Tensor

class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw):
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)

class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'

class MultiScaleResidualQuantizer(nn.Module):
    def __init__(
        self, n_e: int, vq_embed_dim: int, beta: float, latents_scale: Tuple[int]
    ) -> None:
        super().__init__()

        self.n_e = n_e
        self.vq_embed_dim = vq_embed_dim
        self.beta = beta
        self.latents_scale = latents_scale

        self.embedding = nn.Embedding(n_e, vq_embed_dim)
        
        # self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.latents_scale), self.n_e), fill_value=0.0))
        # self.record_hit = 0
        
        self.quant_resi = PhiPartiallyShared(nn.ModuleList([Phi(vq_embed_dim, 0.5)  for _ in range(4)]))

    def forward(
        self, z: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Tuple]:
        dtype = z.dtype
        if dtype != torch.float32:
            z = z.float()
        B, C, H, W = z.size()
        
        z_detach = z.detach()
        z_residual = z_detach.clone()
        z_hat = torch.zeros_like(z_residual).type_as(z_residual)
        
        with torch.cuda.amp.autocast(enabled=False):
            mean_vq_loss: torch.Tensor = torch.tensor(0.0).to(z.device)
            # vocab_hit_v = torch.zeros(self.n_e, dtype=torch.float).to(z.device)
            scale_num = len(self.latents_scale)
            min_encoding_indices_list = []
            
            
            for scale_index, path_resolution in enumerate(self.latents_scale):
                if scale_index != scale_num - 1:
                    residual = F.interpolate(z_residual, size=(path_resolution, path_resolution), mode='area').permute(0, 2, 3, 1).reshape(-1, self.vq_embed_dim)
                else:
                    residual = z_residual.permute(0, 2, 3, 1).reshape(-1, self.vq_embed_dim)
                
                min_encoding_indices = torch.argmin(torch.cdist(residual, self.embedding.weight), dim=1)
                
                # hit_vocab = min_encoding_indices.bincount(minlength=self.n_e).float()
                # if self.training and torch.distributed.is_initialized():
                #     handler = torch.distributed.all_reduce(hit_vocab, async_op=True)
                    
                min_encoding_indices = min_encoding_indices.view(B, path_resolution, path_resolution)
                quantized_feat = self.embedding(min_encoding_indices).permute(0, 3, 1, 2).contiguous()
                if scale_index != scale_num - 1:
                    quantized_feat = F.interpolate(quantized_feat, size=(H, W), mode='bicubic')
                quantized_feat = self.quant_resi[scale_index / (scale_num - 1)](quantized_feat)
                z_hat = z_hat + quantized_feat
                z_residual -= quantized_feat
                
                min_encoding_indices_list.append(min_encoding_indices.view(B, -1))
                # if self.training and torch.distributed.is_initialized():
                #     handler.wait()
                #     if self.record_hit == 0:
                #         self.ema_vocab_hit_SV[scale_index].copy_(hit_vocab)
                #     elif self.record_hit < 100:
                #         self.ema_vocab_hit_SV[scale_index] = self.ema_vocab_hit_SV[scale_index] * 0.9 + hit_vocab * 0.1
                #     else:
                #         self.ema_vocab_hit_SV[scale_index] = self.ema_vocab_hit_SV[scale_index] * 0.99 + hit_vocab * 0.01
                #     self.record_hit += 1
                
                # vocab_hit_v.add_(hit_vocab)
                mean_vq_loss += F.mse_loss(z_hat.data, z).mul_(self.beta) + F.mse_loss(z_hat, z_detach)
            
            total_min_encoding_indices = torch.cat(min_encoding_indices_list, dim=1)
            # hit_vocab = torch.bincount(total_min_encoding_indices.view(-1), minlength=self.n_e).float() / self.n_e
            # perplexity = torch.exp(-torch.sum(hit_vocab * torch.log(hit_vocab + 1e-10)))
            # usages = torch.sum(hit_vocab > 0)
            
            mean_vq_loss = mean_vq_loss / scale_num
            z_hat = (z_hat.data - z_detach).add_(z)
            
            # margin = torch.distributed.get_world_size() * (z.numel() / z.shape[0]) / self.n_e * 0.08
            
            # usages = [(self.ema_vocab_hit_SV[scale_index] >= margin).float().mean() * 100 for scale_index in range(scale_num)]
        
        return z_hat, mean_vq_loss, total_min_encoding_indices
    
    def z_to_z_hat(self, z: torch.Tensor) -> torch.Tensor:
        z_residual = z.clone()
        z_hat = torch.zeros_like(z_residual).type_as(z_residual)
        
        scale_num = len(self.latents_scale)
        for scale_index, path_resolution in enumerate(self.latents_scale):
            if scale_index != scale_num - 1:
                residual = F.interpolate(z_residual, size=(path_resolution, path_resolution), mode='area').permute(0, 2, 3, 1).reshape(-1, self.vq_embed_dim)
            else:
                residual = z_residual.permute(0, 2, 3, 1).reshape(-1, self.vq_embed_dim)
            
            min_encoding_indices = torch.argmin(torch.cdist(residual, self.embedding.weight), dim=1)
            min_encoding_indices = min_encoding_indices.view(z.size(0), path_resolution, path_resolution)
            quantized_feat = self.embedding(min_encoding_indices).permute(0, 3, 1, 2).contiguous()
            if scale_index != scale_num - 1:
                quantized_feat = F.interpolate(quantized_feat, size=(z.size(2), z.size(3)), mode='bicubic')
            quantized_feat = self.quant_resi[scale_index / (scale_num - 1)](quantized_feat)
            z_hat = z_hat + quantized_feat
            z_residual -= quantized_feat
        
        return z_hat
    
class ResidualQuantizer(nn.Module):
    def __init__(
        self, n_e: int, vq_embed_dim: int, beta: float, latents_scale: Tuple[int]
    ) -> None:
        super().__init__()

        self.n_e = n_e
        self.vq_embed_dim = vq_embed_dim
        self.beta = beta
        self.latents_scale = latents_scale

        self.embedding = nn.Embedding(n_e, vq_embed_dim)
        
    def forward(
        self, z: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Tuple]:
        
        dtype = z.dtype
        if dtype != torch.float32:
            z = z.float()
        B, C, H, W = z.size()
        
        z_detach = z.detach()
        z_residual = z_detach.clone()
        z_hat = torch.zeros_like(z_residual).type_as(z_residual)
        
        with torch.cuda.amp.autocast(enabled=False):
            mean_vq_loss = torch.tensor(0.0).to(z.device)
            scale_num = len(self.latents_scale)
            min_encoding_indices_list = []
            
            residual = z_residual.permute(0, 2, 3, 1).reshape(-1, self.vq_embed_dim)
            
            for i in range(scale_num):
                min_encoding_indices = torch.argmin(torch.cdist(residual, self.embedding.weight), dim=1)
                
                min_encoding_indices = min_encoding_indices.view(B, z_residual.shape[-1], z_residual.shape[-1])
                quantized_feat = self.embedding(min_encoding_indices).permute(0, 3, 1, 2).contiguous()
                z_hat = z_hat + quantized_feat
                z_residual = z_residual - quantized_feat
                
                min_encoding_indices_list.append(min_encoding_indices)
                
                mean_vq_loss += F.mse_loss(z_hat.detach(), z).mul_(self.beta) + F.mse_loss(z_hat, z_detach)
                
        total_min_encoding_indices = torch.cat(min_encoding_indices_list, dim=1)
        
        mean_vq_loss = mean_vq_loss / scale_num
        z_hat = z + (z_hat - z).detach()
        
        return z_hat, mean_vq_loss, total_min_encoding_indices
    
    def get_sample(
        self, z: torch.FloatTensor
    ):
        dtype = z.dtype
        if dtype != torch.float32:
            z = z.float()
        B, C, H, W = z.size()
        
        z_detach = z.detach()
        z_residual = z_detach.clone()
        z_hat = torch.zeros_like(z_residual).type_as(z_residual)
        
        with torch.cuda.amp.autocast(enabled=False):
            # mean_vq_loss = torch.tensor(0.0).to(z.device)
            scale_num = len(self.latents_scale)
            feat_list = []
            
            residual = z_residual.permute(0, 2, 3, 1).reshape(-1, self.vq_embed_dim)
            
            for i in range(scale_num):
                feat_list.append(residual)
                min_encoding_indices = torch.argmin(torch.cdist(residual, self.embedding.weight), dim=1)
                
                min_encoding_indices = min_encoding_indices.view(B, z_residual.shape[-1], z_residual.shape[-1])
                quantized_feat = self.embedding(min_encoding_indices).permute(0, 3, 1, 2).contiguous()
                z_hat = z_hat + quantized_feat
                z_residual = z_residual - quantized_feat
        
        return torch.cat(feat_list, dim=0)

class PlainVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super(PlainVectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.init()

class VQTrainModel(VQModel):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ...,
        up_block_types: Tuple[str] = ...,
        block_out_channels: Tuple[int] = ...,
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 3,
        sample_size: int = 32,
        num_vq_embeddings: int = 256,
        norm_num_groups: int = 32,
        vq_embed_dim: int | None = None,
        scaling_factor: float = 0.18215,
        norm_type: str = "group",
        mid_block_add_attention=True,
        lookup_from_codebook=False,
        force_upcast=False,
        resrvoir_num_samples=10240,
        # resrvoir_num_samples=10240,
        reestimate_every_iters=int(600 * 20),
        reestimate_max_iters=int(600 * 200),
        # no_resstimate_before_iters=2e4,
        # reestimate_every_iters=int(10 * 1.2e3),
        # reestimate_every_iters=int(20),
        # reestimate_max_iters=int(300 * 1.2e3),
        reestimate_iter_offset=int(100),
        # reestimate_iter_offset=int(5),
    ):
        super().__init__(
            in_channels,
            out_channels,
            down_block_types,
            up_block_types,
            block_out_channels,
            layers_per_block,
            act_fn,
            latent_channels,
            sample_size,
            num_vq_embeddings,
            norm_num_groups,
            vq_embed_dim,
            scaling_factor,
            norm_type,
            mid_block_add_attention,
            lookup_from_codebook,
            force_upcast,
        )
        
        vq_embed_dim = vq_embed_dim if vq_embed_dim is not None else latent_channels
        
        # self.quantize = ResidualQuantizer(num_vq_embeddings, vq_embed_dim, beta=0.25, latents_scale=[2, 4, 4, 4])
        # self.quantize = 
        
        # self.sampler = ReservoirSampler(num_samples=resrvoir_num_samples)
        self.sampler = None
        self.reestimate_every_iters = reestimate_every_iters
        self.reestimate_max_iters = reestimate_max_iters
        # self.no_resstimate_before_iters = no_resstimate_before_iters
        self.reestimate_iter_offset = reestimate_iter_offset

    def reestimate(self, current_iter):

        if current_iter < self.reestimate_iter_offset:
            # print(1)
            return

        if (
            current_iter - self.reestimate_iter_offset
        ) % self.reestimate_every_iters != 0:
            # print(2)
            return

        if current_iter > self.reestimate_max_iters:
            # print(3)
            return

        tstart = time.time()
        num_clusters = self.quantize.embedding.weight.size(0)
        encodings = self.sampler.contents()
             
        if dist.is_initialized():
            encodings_size = encodings.numel()
            encidings_flat = torch.empty(
                1,
                encodings_size * torch.distributed.get_world_size(),
                dtype=encodings.dtype,
                device=encodings.device,
            )
            dist.all_gather_into_tensor(encidings_flat, encodings.view(-1), async_op=False)
            encodings = encidings_flat.view(-1, encodings.size(1))

        if encodings.size(0) < num_clusters:
            log.info(
                f"Reservoir size {encodings.size(0)} too small for {num_clusters} clusters"
            )
            return

        if dist.is_initialized() and dist.get_rank() != 0 :
            dist.barrier()

        if dist.is_initialized() and dist.get_rank() == 0:
            encodings = encodings.cpu().numpy()
            log.info(f"Begin reestimating")
            clustered, *_ = cluster.k_means(encodings, num_clusters)
            self.quantize.embedding.weight.data[...] = torch.from_numpy(clustered).to(
                self.quantize.embedding.weight.device
            )
            log.info(f"Reestimated in {time.time() - tstart:.2f}s")
            dist.barrier()
        else:
            encodings = encodings.cpu().numpy()
            log.info(f"Begin reestimating")
            clustered, *_ = cluster.k_means(encodings, num_clusters)
            self.quantize.embedding.weight.data[...] = torch.from_numpy(clustered).to(
                self.quantize.embedding.weight.device
            )
            log.info(f"Reestimated in {time.time() - tstart:.2f}s")
            
        if dist.is_initialized():
            dist.broadcast(self.quantize.embedding.weight.data, src=0)
            dist.barrier()

        # dist.barrier()
        self.sampler.reset()

        if current_iter == self.reestimate_max_iters:
            log.info("Disabling reestimation")
            self.sampler = None

    def _log_codeusage(self, predicted_indices):
        num_tokens = self.quantize.n_e
        code_freqs = torch.histc(
            predicted_indices.float(), bins=num_tokens, min=-0.5, max=num_tokens - 0.5
        )
        count = np.prod(predicted_indices.size())
        code_freqs /= count
        entropy = torch.distributions.Categorical(code_freqs).entropy()

        return entropy

    def decode(
        self,
        h: torch.FloatTensor,
        force_not_quantize: bool = False,
        return_dict: bool = True,
        shape=None,
    ) -> DecoderOutput | torch.FloatTensor:
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        elif self.config.lookup_from_codebook:
            quant = self.quantize.get_codebook_entry(h, shape)
        else:
            quant = h
        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(
            quant2, quant if self.config.norm_type == "spatial" else None
        )

        if not return_dict:
            return (dec,)

        entropy = self._log_codeusage(info[-1])

        return VQDecoderOutput(
            sample=dec, emb_loss=emb_loss, info=info, entropy=entropy
        )

    def forward(
        self, x: torch.FloatTensor, global_iter: int, return_dict: bool = True
    ) -> VQDecoderOutput:
        # print(global_iter)
        if self.training and self.sampler is not None:
            # print(5)
            # if dist.get_rank() == 0:
            self.reestimate(global_iter)
            # else:
            # pass

        # log.info(f"Rank {dist.get_rank()} before encode of iter {global_iter}")
        h = self.encode(x).latents
        # log.info(f"Rank {dist.get_rank()} after encode of iter {global_iter}")

        if self.training and self.sampler is not None:
            # h_flattened = h.permute(0, 2, 3, 1).reshape(-1, h.size(1))
            h_feat = self.quantize.get_sample(h)
            self.sampler.add(h_feat.detach())

        dec = self.decode(h, return_dict=return_dict)
        # log.info(f"Rank {dist.get_rank()} before output of iter {global_iter}")
        return dec

def identity(t):
    return t

class MultiScaleRQTrainModel(VQModel):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ...,
        up_block_types: Tuple[str] = ...,
        block_out_channels: Tuple[int] = ...,
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 3,
        sample_size: int = 32,
        num_vq_embeddings: int = 256,
        norm_num_groups: int = 32,
        vq_embed_dim: int | None = None,
        scaling_factor: float = 0.18215,
        norm_type: str = "group",
        mid_block_add_attention=True,
        lookup_from_codebook=False,
        force_upcast=False,
        latents_scale: Tuple[int] = (1, 2, 4, 8, 16, 32, 64),
        resrvoir_num_samples=10240,
        reestimate_every_iters=int(5e4),
        reestimate_max_iters=int(6e5),
        reestimate_iter_offset=int(500),
    ):
        super().__init__(
            in_channels,
            out_channels,
            down_block_types,
            up_block_types,
            block_out_channels,
            layers_per_block,
            act_fn,
            latent_channels,
            sample_size,
            num_vq_embeddings,
            norm_num_groups,
            vq_embed_dim,
            scaling_factor,
            norm_type,
            mid_block_add_attention,
            lookup_from_codebook,
            force_upcast,
        )

        vq_embed_dim = vq_embed_dim if vq_embed_dim is not None else latent_channels
        self.latents_scale = latents_scale
        # self.quantize = ResidualQuantizer(
        #     n_e=num_vq_embeddings,
        #     vq_embed_dim=vq_embed_dim,
        #     beta=0.25,
        #     latents_scale=latents_scale
        # )
        
        self.num_vq_embeddings =num_vq_embeddings
        
        self.quantize = ResidualVQ(
            dim = vq_embed_dim,
            num_quantizers=4,
            codebook_size =num_vq_embeddings,
            # levels = [8, 8, 5, 5, 3],
            # stochastic_sample_codes=True,
            # sample_codebook_temp= 0.1,
            shared_codebook = True,
            # kmeans_init =True,
            # kmeans_iters = 10,
            use_cosine_sim = True,
            threshold_ema_dead_code=2,
            accept_image_fmap = True,
        )
        
        # self.sampler = ReservoirSampler(num_samples=resrvoir_num_samples)
        # self.reestimate_every_iters = reestimate_every_iters
        # self.reestimate_max_iters = reestimate_max_iters
        # self.reestimate_iter_offset = reestimate_iter_offset

    # def reestimate(self, current_iter):

        # if current_iter < self.reestimate_iter_offset:
        #     return

        # if (
        #     current_iter - self.reestimate_iter_offset
        # ) % self.reestimate_every_iters != 0:
        #     return

        # if current_iter > self.reestimate_max_iters:
        #     return

        # tstart = time.time()
        # num_clusters = self.quantize.embedding.weight.size(0)
        # encodings = self.sampler.contents()

        # encodings_size = encodings.numel()
        # encidings_flat = torch.empty(
        #     1,
        #     encodings_size * torch.distributed.get_world_size(),
        #     dtype=encodings.dtype,
        #     device=encodings.device,
        # )
        # dist.all_gather_into_tensor(encidings_flat, encodings.view(-1), async_op=False)
        # encodings = encidings_flat.view(-1, encodings.size(1))

        # if encodings.size(0) < num_clusters:
        #     log.info(
        #         f"Reservoir size {encodings.size(0)} too small for {num_clusters} clusters"
        #     )
        #     return

        # log.info(f"current rank {dist.get_rank()}")
        
        # # if dist.get_rank() != 0:
        # #     dist.barrier()

        # if dist.get_rank() == 0:
        #     encodings = encodings.cpu().numpy()
        #     clustered, *_ = cluster.k_means(encodings, num_clusters)
        #     log.info("begin reestimate")
        #     self.quantize.embedding.weight.data[...] = torch.from_numpy(clustered).to(
        #         self.quantize.embedding.weight.device
        #     )
        #     log.info(f"Reestimated in {time.time() - tstart:.2f}s")
        #     dist.barrier()
        # else:
        #     dist.barrier()

        # dist.broadcast(self.quantize.embedding.weight.data, src=0)
        # dist.barrier()

        # # dist.barrier()
        # self.sampler.reset()

        # if current_iter == self.reestimate_max_iters:
        #     log.info("Disabling reestimation")
        #     self.sampler = None
    
    def decode(
        self,
        h: torch.FloatTensor,
        force_not_quantize: bool = False,
        return_dict: bool = True,
        shape=None,
    ) -> DecoderOutput | torch.FloatTensor:
        # also go through quantization layer
        b, c, hh, ww = h.shape
        
        # h = h.permute(0, 2, 3, 1).reshape(-1, hh * ww, h.size(1))
        
        if not force_not_quantize:
            quant, info, emb_loss = self.quantize(h)
        elif self.config.lookup_from_codebook:
            quant = self.quantize.get_codebook_entry(h, shape)
        else:
            quant = h
        
        # quant = quant.reshape(-1, hh, ww, c).permute(0, 3, 1, 2)
            
        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(
            quant2, quant if self.config.norm_type == "spatial" else None
        )

        if not return_dict:
            return (dec,)

        if not force_not_quantize:
            if emb_loss != None:
                return VQDecoderOutput(
                    sample=dec, emb_loss=emb_loss.mean(), info=info
                )
            else:
                return VQDecoderOutput(
                    sample=dec, emb_loss=None, info=None
                )
        else:
            return VQDecoderOutput(
                sample=dec, emb_loss=None, info=None
            )
        
        # return VQDecoderOutput(
        #     sample=dec, emb_loss=None, info=info
        # )

    def forward(
        self, x: torch.FloatTensor, global_iter: int, return_dict: bool = True
    ) -> VQDecoderOutput:
        # if self.training and self.sampler is not None:
        #     self.reestimate(global_iter)

        h = self.encode(x).latents
        
        # if self.training and self.sampler is not None:
        #     h_flattened = h.permute(0, 2, 3, 1).reshape(-1, h.size(1))
        #     self.sampler.add(h_flattened.detach())

        dec = self.decode(h, return_dict=return_dict)
        return dec

class RQTrainModel(VQModel):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ...,
        up_block_types: Tuple[str] = ...,
        block_out_channels: Tuple[int] = ...,
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 3,
        sample_size: int = 32,
        num_vq_embeddings: int = 256,
        norm_num_groups: int = 32,
        vq_embed_dim: int | None = None,
        scaling_factor: float = 0.18215,
        norm_type: str = "group",
        mid_block_add_attention=True,
        lookup_from_codebook=False,
        force_upcast=False,
        latents_scale: Tuple[int] = (1, 2, 4, 8, 16, 32, 64),
    ):
        super().__init__(
            in_channels,
            out_channels,
            down_block_types,
            up_block_types,
            block_out_channels,
            layers_per_block,
            act_fn,
            latent_channels,
            sample_size,
            num_vq_embeddings,
            norm_num_groups,
            vq_embed_dim,
            scaling_factor,
            norm_type,
            mid_block_add_attention,
            lookup_from_codebook,
            force_upcast,
        )

        vq_embed_dim = vq_embed_dim if vq_embed_dim is not None else latent_channels
        self.latents_scale = latents_scale

        self.num_vq_embeddings =num_vq_embeddings
        
        self.quantize = CustomResidualVQ(
            dim = vq_embed_dim,
            num_quantizers=4,
            codebook_size =num_vq_embeddings,
            # levels = [8, 8, 5, 5, 3],
            # stochastic_sample_codes=True,
            # sample_codebook_temp= 0.1,
            shared_codebook = True,
            # kmeans_init =True,
            # kmeans_iters = 10,
            use_cosine_sim = True,
            threshold_ema_dead_code=2,
            accept_image_fmap = True,
        )
    
    # def decode(
    #     self,
    #     h: torch.FloatTensor,
    #     force_not_quantize: bool = False,
    #     return_dict: bool = True,
    #     shape=None,
    # ) -> DecoderOutput | torch.FloatTensor:
    #     # also go through quantization layer
        
    #     if not force_not_quantize:
    #         quant, info, emb_loss = self.quantize(h)
    #     elif self.config.lookup_from_codebook:
    #         quant = self.quantize.get_codebook_entry(h, shape)
    #     else:
    #         quant = h
        
    #     quant2 = self.post_quant_conv(quant)
    #     dec = self.decoder(
    #         quant2, quant if self.config.norm_type == "spatial" else None
    #     )

    #     if not return_dict:
    #         return (dec,)

    #     if not force_not_quantize:
    #         if emb_loss != None:
    #             return VQDecoderOutput(
    #                 sample=dec, emb_loss=emb_loss.mean(), info=info
    #             )
    #         else:
    #             return VQDecoderOutput(
    #                 sample=dec, emb_loss=None, info=None
    #             )
    #     else:
    #         return VQDecoderOutput(
    #             sample=dec, emb_loss=None, info=None
    #         )
        
    def forward(
        self, x: torch.FloatTensor, global_iter: int, return_dict: bool = True
    ) -> VQDecoderOutput:

        h = self.encode(x).latents

        dec = self.decode(h, return_dict=return_dict)
        return dec

def exists(val):
    return val is not None

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum('h i d, h j d -> h i j', normed_codes, normed_codes)
    return (cosine_sim ** 2).sum() / (h * n ** 2) - (1 / n)

def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, 'h b n -> h b n d', d = dim)
    embeds = repeat(embeds, 'h c d -> h b c d', b = batch)
    return embeds.gather(2, indices)

class CustomCosineSimCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks = 1,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        reset_cluster_size = None,
        use_ddp = False,
        learnable_codebook = False,
        gumbel_sample = gumbel_sample,
        sample_codebook_temp = 1.,
        ema_update = True
    ):
        super().__init__()
        self.transform_input = l2norm

        self.ema_update = ema_update
        self.decay = decay

        if not kmeans_init:
            embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))
        else:
            embed = torch.zeros(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.eps = eps

        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

    # @torch.jit.ignore
    # def init_embed_(self, data, mask = None):
    #     if self.initted:
    #         return

    #     if exists(mask):
    #         c = data.shape[0]
    #         data = rearrange(data[mask], '(c n) d -> c n d', c = c)

    #     embed, cluster_size = kmeans(
    #         data,
    #         self.codebook_size,
    #         self.kmeans_iters,
    #         use_cosine_sim = True,
    #         sample_fn = self.sample_fn,
    #         all_reduce_fn = self.kmeans_all_reduce_fn
    #     )

    #     embed_sum = embed * rearrange(cluster_size, '... -> ... 1')
        
    #     self.embed.data.copy_(embed)
    #     self.embed_avg.data.copy_(embed_sum)
    #     self.cluster_size.data.copy_(cluster_size)
    #     self.initted.data.copy_(torch.Tensor([True]))
    
    @torch.cuda.amp.autocast(enabled = False)
    def forward(
        self,
        x,
        sample_codebook_temp = None,
        mask = None,
        freeze_codebook = False
    ):
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        dtype = x.dtype

        flatten, ps = pack_one(x, 'h * d')

        if exists(mask):
            mask = repeat(mask, 'b n -> c (b h n)', c = flatten.shape[0], h = flatten.shape[-2] // (mask.shape[0] * mask.shape[1]))

        # self.init_embed_(flatten, mask = mask)

        embed = self.embed if self.learnable_codebook else self.embed.detach()

        dist = einsum('h n d, h c d -> h n c', flatten, embed)

        embed_ind = self.gumbel_sample(dist, dim = -1, temperature = sample_codebook_temp, training = self.training)
        embed_ind = unpack_one(embed_ind, ps, 'h *')


        quantize = batched_embedding(embed_ind, embed)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

        dist = unpack_one(dist, ps, 'h * d')
        return quantize, embed_ind, dist

def custom_gumbel_sample(
    logits,
    temperature = 1.,
    stochastic = False,
    straight_through = False,
    reinmax = False,
    dim = -1,
    training = True
):
    dtype, size = logits.dtype, logits.shape[dim]
    sampling_logits = logits

    ind = sampling_logits.argmax(dim = dim)
    return ind


class CustomVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        codebook_dim = None,
        heads = 1,
        separate_codebook_per_head = False,
        decay = 0.8,
        eps = 1e-5,
        freeze_codebook = False,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        use_cosine_sim = False,
        layernorm_after_project_in = False, # proposed by @SaltyChtao here https://github.com/lucidrains/vector-quantize-pytorch/issues/26#issuecomment-1324711561
        threshold_ema_dead_code = 0,
        channel_last = True,
        accept_image_fmap = False,
        commitment_weight = 1.,
        commitment_use_cross_entropy_loss = False,
        orthogonal_reg_weight = 0.,
        orthogonal_reg_active_codes_only = False,
        orthogonal_reg_max_codes = None,
        stochastic_sample_codes = False,
        sample_codebook_temp = 1.,
        straight_through = False,
        reinmax = False,  # using reinmax for improved straight-through, assuming straight through helps at all
        sync_codebook = None,
        sync_affine_param = False,
        ema_update = True,
        learnable_codebook = False,
        in_place_codebook_optimizer = None, # Optimizer used to update the codebook embedding if using learnable_codebook
        affine_param = False,
        affine_param_batch_decay = 0.99,
        affine_param_codebook_decay = 0.9, 
        sync_update_v = 0. # the v that controls optimistic vs pessimistic update for synchronous update rule (21) https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head
        
        codebook_class = CustomCosineSimCodebook
        
        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.has_codebook_orthogonal_loss = has_codebook_orthogonal_loss
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        
        self.project_in = torch.nn.Sequential(
            nn.Linear(dim, codebook_input_dim),
            nn.LayerNorm(codebook_input_dim) if layernorm_after_project_in else None
        ) if requires_projection else nn.Identity()

        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()
        
        # codebook_class = EuclideanCodebook if not use_cosine_sim else CosineSimCodebook

        gumbel_sample_fn = functools.partial(
            custom_gumbel_sample,
            stochastic = stochastic_sample_codes,
            reinmax = reinmax,
            straight_through = straight_through
        )
        
        codebook_kwargs = dict(
            dim = codebook_dim,
            num_codebooks = heads if separate_codebook_per_head else 1,
            codebook_size = codebook_size,
            kmeans_init = kmeans_init,
            kmeans_iters = kmeans_iters,
            sync_kmeans = sync_kmeans,
            decay = decay,
            eps = eps,
            threshold_ema_dead_code = threshold_ema_dead_code,
            use_ddp = sync_codebook,
            learnable_codebook = has_codebook_orthogonal_loss or learnable_codebook,
            sample_codebook_temp = sample_codebook_temp,
            gumbel_sample = gumbel_sample_fn,
            ema_update = ema_update
        )
        
        self._codebook = codebook_class(**codebook_kwargs)
        
        self.codebook_size = codebook_size

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last
        
    
    def forward(
        self,
        x,
        indices = None,
        mask = None,
        sample_codebook_temp = None,
        freeze_codebook = False
    ):
        only_one = x.ndim == 2

        if only_one:
            assert not mask is not None
            x = rearrange(x, 'b d -> b 1 d')

        need_transpose = not self.channel_last and not self.accept_image_fmap

        # rearrange inputs

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')

        # project input

        x = self.project_in(x)

        # l2norm for cosine sim, otherwise identity

        x = self._codebook.transform_input(x)

        # codebook forward kwargs

        codebook_forward_kwargs = dict(
            sample_codebook_temp = sample_codebook_temp,
            mask = mask,
            freeze_codebook = freeze_codebook
        )

        # quantize
        quantize, _, _ = self._codebook(x, **codebook_forward_kwargs)
        
        quantize = x + (quantize - x).detach()
        
        del _


        # if self.accept_image_fmap:
        #     embed_ind = rearrange(embed_ind, 'b (h w) ... -> b h w ...', h = height, w = width)

        # if only_one:
        #     embed_ind = rearrange(embed_ind, 'b 1 ... -> b ...')

        # aggregate loss

        quantize = self.project_out(quantize)

        # rearrange quantized embeddings

        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')

        if self.accept_image_fmap:
            quantize = rearrange(quantize, 'b (h w) c -> b c h w', h = height, w = width)

        if only_one:
            quantize = rearrange(quantize, 'b 1 d -> b d')

        # if masking, only return quantized for where mask has True

        return quantize, None, None

def default(val, d):
    return val if exists(val) else d

class CustomResidualVQ(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_quantizers,
        codebook_dim = None,
        shared_codebook = False,
        heads = 1,
        quantize_dropout = False,
        quantize_dropout_cutoff_index = 0,
        quantize_dropout_multiple_of = 1,
        accept_image_fmap = False,
        **kwargs
    ):
        super().__init__()
        assert heads == 1, 'residual vq is not compatible with multi-headed codes'
        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection

        self.num_quantizers = num_quantizers

        self.accept_image_fmap = accept_image_fmap
        self.layers = nn.ModuleList([CustomVectorQuantizer(dim = codebook_dim, codebook_dim = codebook_dim, accept_image_fmap = accept_image_fmap, **kwargs) for _ in range(num_quantizers)])

        # assert all([not vq.has_projections for vq in self.layers])

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # encodec paper proposes structured dropout, believe this was set to 4

        if not shared_codebook:
            return

        first_vq, *rest_vq = self.layers
        codebook = first_vq._codebook

        for vq in rest_vq:
            vq._codebook = codebook
            
            
    def forward(
        self,
        x,
        mask = None,
        indices = None,
        return_all_codes = False,
        sample_codebook_temp = None,
        freeze_codebook = False,
        rand_quantize_dropout_fixed_seed = None
    ):
        num_quant, quant_dropout_multiple_of, return_loss, device = self.num_quantizers, self.quantize_dropout_multiple_of, exists(indices), x.device

        x = self.project_in(x)

        assert not (self.accept_image_fmap and exists(indices))

        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []

        if return_loss:
            assert not torch.any(indices == -1), 'some of the residual vq indices were dropped out. please use indices derived when the module is in eval mode to derive cross entropy loss'
            ce_losses = []

        # go through the layers

        for quantizer_index, layer in enumerate(self.layers):
        # for i in range():
            
            # if quantizer_index == 3:
            #     break
            
            layer_indices = None
            if return_loss:
                layer_indices = indices[..., quantizer_index]

            quantized, *rest = layer(
                residual,
                mask = mask,
                indices = layer_indices,
                sample_codebook_temp = sample_codebook_temp,
                freeze_codebook = freeze_codebook
            )

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            if return_loss:
                ce_loss = rest[0]
                ce_losses.append(ce_loss)
                continue

            embed_indices, loss = rest

            all_indices.append(embed_indices)
            all_losses.append(loss)

        # project out, if needed

        quantized_out = self.project_out(quantized_out)

        # whether to early return the cross entropy loss

        if return_loss:
            return quantized_out, sum(ce_losses)

        # stack all losses and indices

        if all_losses[0] != None:
            all_losses, all_indices = map(functools.partial(torch.stack, dim = -1), (all_losses, all_indices))

            ret = (quantized_out, all_indices, all_losses)
        else:
            ret = (quantized_out, all_indices, None)

        if return_all_codes:
            # whether to return all codes from all codebooks across layers
            all_codes = self.get_codes_from_indices(all_indices)

            # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)
            ret = (*ret, all_codes)

        return ret