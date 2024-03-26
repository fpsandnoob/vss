from dataclasses import dataclass
from typing import Tuple, Union
from diffusers.models import VQModel
from diffusers.utils import BaseOutput
import einops
from vector_quantize_pytorch import ResidualFSQ
import torch

@dataclass
class FSQEncoderOutput(BaseOutput):
    latents: torch.Tensor

@dataclass
class FSQDecoderOutput(BaseOutput):
    sample: torch.Tensor
    indices: torch.Tensor

class FSQTrainModel(VQModel):
    def __init__(self, *args, **kwargs):
        fsq_embed_dim = kwargs.pop('fsq_embed_dim')
        levels = kwargs.pop('levels')
        num_quantizers = kwargs.pop('num_quantizers')
        super().__init__(*args, **kwargs)
        self.quantize = ResidualFSQ(
            dim=fsq_embed_dim,
            levels=levels,
            num_quantizers=num_quantizers
        )
    
    def decode(self, h: torch.Tensor, force_not_quantize: bool = False, return_dict: bool = True)  -> Union[FSQDecoderOutput, torch.Tensor]:
        if not force_not_quantize:
            width, height = h.shape[-2:]
            h = einops.rearrange(h, 'b c w h -> b (w h) c')
            quant, indices = self.quantize(h)
            quant = einops.rearrange(quant, 'b (w h) c -> b c w h', w=width, h=height)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        
        indices = indices.view(-1)

        if not return_dict:
            return (dec,indices,)

        return FSQDecoderOutput(sample=dec, indices=indices)
    
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> FSQEncoderOutput:
        h = self.encoder(x)
        h = self.quant_conv(h)

        if not return_dict:
            return (h,)

        return FSQEncoderOutput(latents=h)

    def forward(self, sample: torch.Tensor, return_dict: bool = True):
        x = sample
        h = self.encode(x).latents
        # print(h.shape)
        dec_output = self.decode(h, force_not_quantize=False, return_dict=True)

        if not return_dict:
            return (dec_output.sample, dec_output.indices,)

        return dec_output

if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    model = FSQTrainModel(
        in_channels=1,
        out_channels=1,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(64, 128, 256),
        layers_per_block=2,
        fsq_embed_dim=256,
        levels=[7, 5, 5, 5],
        num_quantizers=1,
        vq_embed_dim=256
    )
    model.eval()
    x = torch.randn(1, 1, 128, 128)
    out = model(x)
    print(out.sample.shape, out.indices.shape)
    print(out.indices)