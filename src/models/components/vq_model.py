from dataclasses import dataclass
from typing import Tuple
from diffusers.models import VQModel
from diffusers.models.vae import Decoder
from diffusers.utils import BaseOutput
import torch

@dataclass
class VQEncoderOutput(BaseOutput):
    latents: torch.Tensor

@dataclass
class VQDecoderOutput(BaseOutput):
    sample: torch.Tensor
    emb_loss: torch.Tensor
    info: Tuple
    
class PatchedVQTrainModel(VQModel):
    def __init__(self, *args, **kwargs):
        patch_size = kwargs.pop('patch_size')
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        
    def convert_image_to_patches(self, x):
        p = self.patch_size
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1) #BHWC format, bc reshape is done on last 2 axes
        x = x.reshape(B, H, W//p, C*p) #reshape from width axis to channel axis
        x = x.permute(0, 2, 1, 3) #now height & channel should be last 2 axes
        x = x.reshape(B, W//p, H//p, C*p*p) #reshape from height axis to channel axis
        return x.permute(0, 3, 2, 1) #convert to channels-first format
    
    def convert_patches_to_image(self, x):
        p = self.patch_size
        B, C, H, W = x.shape
        x = x.permute(0,3,2,1) #BWHC; from_patches starts w/ height axis, not width
        x.reshape(B, W, H*p, C//p) #reshape from channel axis to height axis
        x = x.permute(0,2,1,3) #now width & channel should be last 2 axes
        x = x.reshape(B, H*p, W*p, C//(p*p)) #reshape from channel axis to width axis
        return x.permute(0, 3, 1, 2) #convert to channels-first format
    
    def decode(self, h: torch.FloatTensor, force_not_quantize: bool = False, return_dict: bool = True):
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)

        if not return_dict:
            return (dec,emb_loss,info,)

        return VQDecoderOutput(sample=dec, emb_loss=emb_loss, info=info)

    def forward(self, sample: torch.FloatTensor, return_dict: bool = True):
        x = sample
        x = self.convert_image_to_patches(x)
        h = self.encode(x).latents
        dec_output = self.decode(h, force_not_quantize=False, return_dict=True)
        sample = self.convert_patches_to_image(dec_output.sample)
        
        if not return_dict:
            
            return (sample, dec_output.emb_loss,dec_output.info,)

        return VQDecoderOutput(sample=sample, emb_loss=dec_output.emb_loss, info=dec_output.info)

class VQTrainModel(VQModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def decode(self, h: torch.FloatTensor, force_not_quantize: bool = False, return_dict: bool = True):
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)

        if not return_dict:
            return (dec,emb_loss,info,)

        return VQDecoderOutput(sample=dec, emb_loss=emb_loss, info=info)

    def forward(self, sample: torch.FloatTensor, return_dict: bool = True):
        x = sample
        h = self.encode(x).latents
        dec_output = self.decode(h, force_not_quantize=False, return_dict=True)

        if not return_dict:
            return (dec_output.sample, dec_output.emb_loss,dec_output.info,)

        return VQDecoderOutput(sample=dec_output.sample, emb_loss=dec_output.emb_loss, info=dec_output.info)

class VQVAE2TrainModel(object):
    def __init__(self, top_model: VQModel, bottom_model:VQModel, decoder_model:Decoder) -> None:
        super().__init__()
        self.top_model = top_model
        self.bottom_model = bottom_model
        
        self.decoder_model = decoder_model
        
        self.upsample_t = torch.nn.ConvTranspose2d(
            bottom_model.quant_conv.out_channels, bottom_model.quant_conv.out_channels, 4, stride=2, padding=1
        )
        
    def encode(self, input):
        enc_b = self.bottom_model.encoder(input)
        enc_t = self.top_model.encode(enc_b)
        
        quant_t = self.top_model.quant_conv(enc_t)
        quant_t, emb_loss_t, info_t = self.top_model.quantize(quant_t)
        
        dec_t = self.top_model.decoder(self.top_model.post_quant_conv(quant_t))
        enc_b = torch.cat([dec_t, enc_b], 1)
        
        quant_b = self.bottom_model.quant_conv(enc_b)
        quant_b, emb_loss_b, info_b = self.bottom_model.quantize(quant_b)
        
        return quant_t, quant_b, emb_loss_t + emb_loss_b, info_t, info_b
    
    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b])
        dec = self.decoder_model(quant)
        
        return dec


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