from dataclasses import dataclass
from typing import Optional, Tuple, Union
from diffusers.models import AutoencoderKL
from diffusers.models.vae import DecoderOutput
from diffusers.utils import BaseOutput
import torch
from torch._C import Generator

@dataclass
class KLEncoderOutput(BaseOutput):
    latents: torch.Tensor

@dataclass
class KLDecoderOutput(BaseOutput):
    sample: torch.Tensor
    posterior: torch.Tensor

class KLTrainModel(AutoencoderKL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, sample: torch.FloatTensor, sample_posterior: bool = False, return_dict: bool = True, generator: Generator | None = None) -> KLDecoderOutput:
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return KLDecoderOutput(sample=dec, posterior=posterior)


if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    model = VQTrainModel(
        in_channels=1,
        out_channels=1,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(64, 128, 256),
        layers_per_block=2,
    )
    model.eval()
    x = torch.randn(1, 1, 128, 128)
    out = model(x)
    print(out.sample.shape, out.info[-1].shape)
    print(out.info[-1].dtype)