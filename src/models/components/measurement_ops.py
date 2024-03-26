from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import List, Tuple
from torch.nn import functional as F
from torch import nn
import torch
import scipy
import numpy as np
import odl
from odl.contrib.torch import OperatorModule
try:
    from torch_radon import RadonFanbeam, Radon
except:
    pass
from src.data.components.mayo_proj_dataset import load_tiff_stack_with_metadata


class Op(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # generic forward
        pass

    def backproject(self, data, **kwargs):
        # generic backproject
        pass


class OpCompose(Op):
    def __init__(self, op_list: List[Op]):
        self.op_list = op_list

    def forward(self, data):
        for op in self.op_list:
            data = op.forward(data)
        return data

    def backproject(self, data, **kwargs):
        for op in self.op_list:
            data = op.backproject(data)
        return data


class LinearOp(Op):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass


class NonLinearOp(Op):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass


class NoiseOp(Op):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate b
        pass

def parallel_beam_geometry(space, num_angles=None, det_shape=None, angles=np.pi):
    r"""Create default parallel beam geometry from ``space``.

    This is intended for simple test cases where users do not need the full
    flexibility of the geometries, but simply want a geometry that works.

    This default geometry gives a fully sampled sinogram according to the
    Nyquist criterion, which in general results in a very large number of
    samples. In particular, a ``space`` that is not centered at the origin
    can result in very large detectors.

    Parameters
    ----------
    space : `DiscretizedSpace`
        Reconstruction space, the space of the volumetric data to be projected.
        Needs to be 2d or 3d.
    num_angles : int, optional
        Number of angles.
        Default: Enough to fully sample the data, see Notes.
    det_shape : int or sequence of int, optional
        Number of detector pixels.
        Default: Enough to fully sample the data, see Notes.

    Returns
    -------
    geometry : `ParallelBeamGeometry`
        If ``space`` is 2d, return a `Parallel2dGeometry`.
        If ``space`` is 3d, return a `Parallel3dAxisGeometry`.

    Examples
    --------
    Create a parallel beam geometry from a 2d space:

    >>> space = odl.uniform_discr([-1, -1], [1, 1], (20, 20))
    >>> geometry = parallel_beam_geometry(space)
    >>> geometry.angles.size
    45
    >>> geometry.detector.size
    31

    Notes
    -----
    According to [NW2001]_, pages 72--74, a function
    :math:`f : \mathbb{R}^2 \to \mathbb{R}` that has compact support

    .. math::
        \| x \| > \rho  \implies f(x) = 0,

    and is essentially bandlimited

    .. math::
       \| \xi \| > \Omega \implies \hat{f}(\xi) \approx 0,

    can be fully reconstructed from a parallel beam ray transform
    if (1) the projection angles are sampled with a spacing of
    :math:`\Delta \psi` such that

    .. math::
        \Delta \psi \leq \frac{\pi}{\rho \Omega},

    and (2) the detector is sampled with an interval :math:`\Delta s`
    that satisfies

    .. math::
        \Delta s \leq \frac{\pi}{\Omega}.

    The geometry returned by this function satisfies these conditions exactly.

    If the domain is 3-dimensional, the geometry is "separable", in that each
    slice along the z-dimension of the data is treated as independed 2d data.

    References
    ----------
    .. [NW2001] Natterer, F and Wuebbeling, F.
       *Mathematical Methods in Image Reconstruction*.
       SIAM, 2001.
       https://dx.doi.org/10.1137/1.9780898718324
    """
    # Find maximum distance from rotation axis
    corners = space.domain.corners()[:, :2]
    rho = np.max(np.linalg.norm(corners, axis=1))

    # Find default values according to Nyquist criterion.

    # We assume that the function is bandlimited by a wave along the x or y
    # axis. The highest frequency we can measure is then a standing wave with
    # period of twice the inter-node distance.
    min_side = min(space.partition.cell_sides[:2])
    omega = np.pi / min_side
    num_px_horiz = 2 * int(np.ceil(rho * omega / np.pi)) + 1

    if space.ndim == 2:
        det_min_pt = -rho
        det_max_pt = rho
        if det_shape is None:
            det_shape = num_px_horiz
    elif space.ndim == 3:
        num_px_vert = space.shape[2]
        min_h = space.domain.min_pt[2]
        max_h = space.domain.max_pt[2]
        det_min_pt = [-rho, min_h]
        det_max_pt = [rho, max_h]
        if det_shape is None:
            det_shape = [num_px_horiz, num_px_vert]

    if num_angles is None:
        num_angles = int(np.ceil(omega * rho))

    angle_partition = odl.uniform_partition(0, angles, num_angles)
    det_partition = odl.uniform_partition(det_min_pt, det_max_pt, det_shape)

    if space.ndim == 2:
        return odl.tomo.Parallel2dGeometry(angle_partition, det_partition)
    elif space.ndim == 3:
        return odl.tomo.Parallel3dAxisGeometry(angle_partition, det_partition)
    else:
        raise ValueError('``space.ndim`` must be 2 or 3.')

class ProjectionModule(OperatorModule):
    def __init__(
        self,
        num_angles: int = 64,
        det_shape: Tuple[
            int,
        ] = (513,),
        im_shape: Tuple[int, int] = (512, 512),
        angles: float = np.pi,
        cuda: bool = True,
    ):
        self.num_angles = num_angles
        self.det_shape = det_shape
        self.im_shape = im_shape
        self.cuda = cuda

        recon_space = odl.uniform_discr(
            min_pt=(-1, -1), max_pt=(1, 1), shape=self.im_shape
        )
        geometry = parallel_beam_geometry(
            recon_space, num_angles=self.num_angles, det_shape=self.det_shape, angles=angles
        )

        if self.cuda:
            ray_trafo = odl.tomo.RayTransform(recon_space, geometry, impl="astra_cuda")
        else:
            ray_trafo = odl.tomo.RayTransform(recon_space, geometry, impl="skimage")

        self.ray_trafo = ray_trafo        
        self.mu_water = 0.02
        self.epsilon = 0.0001
        nonlinear_operator = odl.ufunc_ops.exp(ray_trafo.range) * (- self.mu_water * ray_trafo)

        super(ProjectionModule, self).__init__(ray_trafo)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            out = super(ProjectionModule, self).forward(x)
            return out
        elif x.ndim == 5:
            B, C, T, H, W = x.shape
            x = x.view(B, T, H, W)
            out = super(ProjectionModule, self).forward(x)
            out = out.view(B, C, T, H, W)
            return out
              
    def __repr__(self):
        return f"ProjectionModule(num_angles={self.num_angles}, det_shape={self.det_shape}, im_shape={self.im_shape}, cuda={self.cuda})"
    
class FanBeamProjectionModule(OperatorModule):
    def __init__(
        self,
        num_angles: int = 64,
        det_shape: Tuple[
            int,
        ] = (513,),
        im_shape: Tuple[int, int] = (512, 512),
        angles: float = np.pi,
        cuda: bool = True,
    ):
        self.num_angles = num_angles
        self.det_shape = det_shape
        self.im_shape = im_shape
        self.cuda = cuda

        recon_space = odl.uniform_discr(
            min_pt=(-20, -20), max_pt=(20, 20), shape=self.im_shape,
        dtype='float32'
        )
        angle_partition = odl.uniform_partition(0, 2 * angles, num_angles)
        detector_partition = odl.uniform_partition(-60, 60, det_shape)
        geometry = odl.tomo.FanBeamGeometry(
            angle_partition, detector_partition, src_radius=40, det_radius=40)

        if self.cuda:
            ray_trafo = odl.tomo.RayTransform(recon_space, geometry, impl="astra_cuda")
        else:
            ray_trafo = odl.tomo.RayTransform(recon_space, geometry, impl="skimage")

        # self.ray_trafo = ray_trafo        
        # self.mu_water = 0.02
        # self.epsilon = 0.0001
        # nonlinear_operator = odl.ufunc_ops.exp(ray_trafo.range) * (- self.mu_water * ray_trafo)

        super(FanBeamProjectionModule, self).__init__(ray_trafo)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            out = super(FanBeamProjectionModule, self).forward(x)
            return out
        elif x.ndim == 5:
            B, C, T, H, W = x.shape
            x = x.view(B, T, H, W)
            out = super(FanBeamProjectionModule, self).forward(x)
            out = out.view(B, C, T, H, W)
            return out
        else:
            pass
              
    def __repr__(self):
        return f"ProjectionModule(num_angles={self.num_angles}, det_shape={self.det_shape}, im_shape={self.im_shape}, cuda={self.cuda})"
    
class BackProjectionModule(OperatorModule):
    def __init__(
        self,
        num_angles: int = 64,
        det_shape: Tuple[
            int,
        ] = (513,),
        im_shape: Tuple[int, int] = (512, 512),
        angles: float = np.pi,
        filter_type: str = "Ram-Lak",
        cuda: bool = True,
    ):
        self.num_angles = num_angles
        self.det_shape = det_shape
        self.im_shape = im_shape
        self.filter_type = filter_type
        self.cuda = cuda

        space = odl.uniform_discr(min_pt=(-1, -1), max_pt=(1, 1), shape=self.im_shape)
        geometry = parallel_beam_geometry(
            space, num_angles=self.num_angles, det_shape=self.det_shape, angles=angles
        )

        if self.cuda:
            ray_trafo = odl.tomo.RayTransform(space, geometry, impl="astra_cuda")
        else:
            ray_trafo = odl.tomo.RayTransform(space, geometry, impl=None)

        if self.filter_type:
            if self.filter_type != "ramp":
                fbp = odl.tomo.fbp_op(
                    ray_trafo, filter_type=self.filter_type, frequency_scaling=0.8
                )
                super(BackProjectionModule, self).__init__(fbp)
            else:
                fourier = odl.trafos.FourierTransform(ray_trafo.range, axes=[1])
                ramp_function = fourier.range.element(
                    lambda x: np.abs(x[1]) / (2 * np.pi)
                )
                ramp_filter = fourier.inverse * ramp_function * fourier
                fbp = ray_trafo.adjoint * ramp_filter

                super(BackProjectionModule, self).__init__(fbp)
        else:
            super(BackProjectionModule, self).__init__(ray_trafo.adjoint)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super(BackProjectionModule, self).forward(x)
        return out

    def __repr__(self):
        return f"BackProjectionModule(num_angles={self.num_angles}, det_shape={self.det_shape}, im_shape={self.im_shape}, filter_type={self.filter_type}, cuda={self.cuda})"

# class CTSparseViewOp(LinearOp):
#     def __init__(self, num_views, det_shape, im_shape, angles=np.pi) -> None:
#         super().__init__()
#         self.num_views = num_views
#         self.projector = ProjectionModule(
#             num_angles=num_views, det_shape=det_shape, im_shape=im_shape, angles=angles
#         )
#         self.backprojector = BackProjectionModule(
#             num_angles=num_views, det_shape=det_shape, im_shape=im_shape, angles=angles
#         )
        
#     def forward(self, data):
#         return self.projector(data).float()
    
#     def backproject(self, data):
#         return self.backprojector(data).float()
    
class CTSparseViewOp(LinearOp):
    def __init__(self, num_views, det_shape, im_shape, angles=np.pi) -> None:
        super().__init__()
        self.num_views = num_views
        self.projector = ProjectionModule(
            num_angles=num_views, det_shape=det_shape, im_shape=im_shape, angles=angles
        )
        self.backprojector = BackProjectionModule(
            num_angles=num_views, det_shape=det_shape, im_shape=im_shape, angles=angles
        )
        
    def forward(self, data):
        return self.projector(data).float()
    
    def backproject(self, data):
        return self.backprojector(data).float()
    

class CTFanBeamSparseViewODLOp(LinearOp):
    def __init__(self, num_views, det_shape, im_shape, angles=np.pi) -> None:
        super().__init__()
        self.num_views = num_views
        self.projector = ProjectionModule(
            num_angles=num_views, det_shape=det_shape, im_shape=im_shape, angles=angles
        )
        self.backprojector = BackProjectionModule(
            num_angles=num_views, det_shape=det_shape, im_shape=im_shape, angles=angles
        )
        
    def forward(self, data):
        return self.projector(data).float()
    
    def backproject(self, data):
        return self.backprojector(data).float()
    
class CTSparseViewParallelBeamOp(LinearOp):
    def __init__(self, num_views, det_count, im_size, fbp_filter) -> None:
        super().__init__()
        self.fbp_filter = fbp_filter
        
        angles = torch.linspace(0, torch.pi * 2, num_views)
        
        self.radon = Radon(
            resolution=im_size,
            angles=angles,
            det_count=det_count
        )
    
    def forward(self, data):
        return self.radon.forward(data.float())
    
    def backproject(self, data, **kwargs):
        filtered_sinogram = self.radon.filter_sinogram(data.float(), filter_name=self.fbp_filter)
        return self.radon.backprojection(filtered_sinogram)

class CTSparseViewFanBeamOp(LinearOp):
    def __init__(self, metadata_path, num_views, image_size, voxel_size, fbp_filter) -> None:
        super().__init__()
        self.metadata_path = metadata_path
        self.num_views = num_views
        self.image_size = image_size
        self.voxel_size = voxel_size
        self.fbp_filter = fbp_filter
        
        _, metadata = load_tiff_stack_with_metadata(Path(self.metadata_path))
        self.vox_scaling = 1 / self.voxel_size
        self.hu_factor = metadata['hu_factor']
        
        print("nums of views:", metadata['rotview'])
        print("voxel_size:", self.voxel_size)
        print('source_distance:', self.vox_scaling * metadata['dso'])
        print('det_distance:', self.vox_scaling * metadata['ddo'])
        print('det_spacing:', self.vox_scaling * metadata['du'])
        self.downsample_factor = int(metadata['rotview'] / num_views)
        angles = np.array(metadata['angles'])[:metadata['rotview']:self.downsample_factor] + (np.pi / 2)
        print(angles.shape)
        
        new_downsample_factor = int(metadata['rotview'] / 32)
        # print(new_downsample_factor)
        new_angles = np.array(metadata['angles'])[:metadata['rotview']:new_downsample_factor] + (np.pi / 2)
        
        self.radon = RadonFanbeam(
            self.image_size,
            angles,
            source_distance=self.vox_scaling * metadata['dso'],
            det_distance=self.vox_scaling * metadata['ddo'],
            det_count=736,
            det_spacing=self.vox_scaling * metadata['du'],
            clip_to_circle=False
        )
        
        self.gt_radon = RadonFanbeam(
            self.image_size,
            np.array(metadata['angles'])[:metadata['rotview']] + (np.pi / 2),
            # new_angles,
            source_distance=self.vox_scaling * metadata['dso'],
            det_distance=self.vox_scaling * metadata['ddo'],
            det_count=736,
            det_spacing=self.vox_scaling * metadata['du'],
            clip_to_circle=False
        )
        
    def forward(self, data):
        return self.radon.forward(data.float())
    
    def backproject(self, data, **kwargs):
        filtered_sinogram = self.radon.filter_sinogram(data.float(), filter_name=self.fbp_filter)
        return self.radon.backprojection(filtered_sinogram)
    
    def gt_backproject(self, data, **kwargs):
        filtered_sinogram = self.gt_radon.filter_sinogram(data.float(), filter_name=self.fbp_filter)
        return self.gt_radon.backprojection(filtered_sinogram).double()
        
class SuperResolutionOp(LinearOp):
    def __init__(self, scale_factor, mode="bilinear"):
        self.scale_factor = scale_factor
        self.mode = mode
        # self.downsample = nn.AdaptiveAvgPool2d((256 // scale_factor, 256 // scale_factor))
        self.downsample = nn.Upsample(scale_factor=1 / self.scale_factor, mode=mode)
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode=mode)

    def forward(self, data):
        return self.downsample(
            data
        )

    def backproject(self, x, **kwargs):
        # n, c, h, w = x.shape
        # out = torch.zeros(n, c, h, self.scale_factor, w, self.scale_factor).to(x.device) + x.view(n,c,h,1,w,1)
        # out = out.view(n, c, self.scale_factor*h, self.scale_factor*w)
        return self.upsample(
            x
        )

# class ColorizationOp(LinearOp):
#     def __init__(self) -> None:
#         super().__init__()
        
#     def forward(self, data):
#         coef = 1 / 3
#         data = data[:, 0, :, :] * coef + data[:, 1, :, :] * coef + data[:, 2, :, :] * coef
#         return data.repeat(1, 3, 1, 1)
    
#     def backproject(self, x, **kwargs):
#         x = x[:, 0, :, :]
#         coef = 1 / 3
#         base = coef ** 2 + coef ** 2 + coef ** 2
#         return torch.stack([x * coef / base, x * coef / base, x * coef / base], dim=1)

class ColorizationOp(LinearOp):
    def __init__(self) -> None:
        super().__init__()
        self.M =  torch.tensor([[5.7735014e-01, -8.1649649e-01, 4.7008697e-08],
                      [5.7735026e-01, 4.0824834e-01, 7.0710671e-01],
                      [5.7735026e-01, 4.0824822e-01, -7.0710683e-01]])
        self.invM = torch.inverse(self.M)
        
    def forward(self, data, **kwargs):
        # print(data.shape)
        # print(torch.einsum('bihw,ij->bjhw', data, self.M.to(data.device)).shape)
        # return torch.einsum('bihw,ij->bjhw', data, self.M.to(data.device))
        return torch.mean(data, dim=1, keepdim=True).repeat(1, 3, 1, 1)
    
    def backproject(self, data, **kwargs):
        return torch.einsum('bihw,ij->bjhw', data, self.M.to(data.device))

class Blurkernel(nn.Module):
    def __init__(self, blur_type="gaussian", kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size // 2),
            nn.Conv2d(
                3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3
            ),
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2, self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            k = Kernel(
                size=(self.kernel_size, self.kernel_size), intensity=self.std
            ).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k


class MotionBlurOp(LinearOp):
    def __init__(self, kernel_size, intensity, device) -> None:
        self.kernel_size = kernel_size
        self.intensity = intensity
        self.device = device
        self.conv = Blurkernel(
            blur_type="motion",
            kernel_size=self.kernel_size,
            std=self.intensity,
            device=self.device,
        )

        self.kernel = Kernel(
            size=(self.kernel_size, self.kernel_size), intensity=self.intensity
        )
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
        self.conv.to(self.device)

    def forward(self, data):
        return self.conv(data)

    def backproject(self, data, **kwargs):
        return data


class GaussianBlurOp(LinearOp):
    def __init__(self, kernel_size, std, device) -> None:
        self.kernel_size = kernel_size
        self.std = std
        self.device = device

        self.conv = Blurkernel(
            blur_type="gaussian",
            kernel_size=self.kernel_size,
            std=self.std,
            device=device,
        )

        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))
        self.conv.to(self.device)

    def forward(self, data):
        return self.conv(data)

    def backproject(self, data, **kwargs):
        return data


class CenterBoxInpaintingOP(LinearOp):
    def __init__(self, im_size, box_size):
        self.box_size = box_size
        self.im_size = im_size
        self.mask = torch.ones((1, 1, self.im_size, self.im_size))
        self.mask[
            :,
            :,
            self.im_size // 2
            - self.box_size // 2 : self.im_size // 2
            + self.box_size // 2,
            self.im_size // 2
            - self.box_size // 2 : self.im_size // 2
            + self.box_size // 2,
        ] = 0

    def forward(self, data):
        if self.mask.device != data.device:
            self.mask = self.mask.to(data.device)
        return data * self.mask

    def backproject(self, data, **kwargs):
        return data

class RandomBoxInpaintingOP(LinearOp):
    def __init__(self, im_size, mask_factor, mask_dir=None):
        mask_path = os.path.join(mask_dir, f"mask_rand_{mask_factor}.pt")
        
        self.im_size = im_size
        self.mask_factor = mask_factor
        if os.path.exists(mask_path):
            self.mask = torch.load(mask_path)
        else:
            self.mask = torch.ones(im_size, im_size)
            self.mask[torch.rand_like(self.mask) > 1 - mask_factor] = 0
            torch.save(self.mask, mask_path)
            
    def forward(self, data):
        if self.mask.device != data.device:
            self.mask = self.mask.to(data.device)
        return data * self.mask.view(1, 1, self.im_size, self.im_size)
    
    def backproject(self, data, **kwargs):
        return data
        
class NonlinearBlurOp(NonLinearOp):
    def __init__(self, opt_yaml_path, device) -> None:
        super().__init__()
        self.opt_yaml_path = opt_yaml_path
        self.device = device

        self.model = self.load_nonlinear_blur_model(
            self.opt_yaml_path, self.device
        )

    @staticmethod
    def load_nonlinear_blur_model(opt_yaml_path, device):
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard
        import yaml

        with open(opt_yaml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        model = KernelWizard(opt)
        model.eval()
        model.load_state_dict(torch.load(model_path))
        return model.to(device)

    def forward(self, data, **kwargs):
        random_kernel = torch.randn((1, 512, 2, 2), device=self.device) * 1.2
        data = (data + 1.0) / 2.0
        blurred_data = self.model.adaptKernel(data, random_kernel)
        blurred_data = ((blurred_data - 0.5) * 2.0).clamp(-1.0, 1.0)
        return blurred_data
    
class PhaseRetrievalOperator(NonLinearOp):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        if data.min() < 0:
            data = data / 2 + 0.5
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))     
        amplitude = fft2_m(padded).abs()
        return amplitude
    
    def backproject(self, data, **kwargs):
        return data

class QuantizerOperator(NonLinearOp):
    def __init__(self, num_bits, device):
        self.num_bits = num_bits
        self.device = device
        
    def forward(self, data, **kwargs):
        if data.min() >= 0:
            data = data * 2 - 1
        if self.num_bits == 1:
            return torch.sign(data)
        
        nLevels = 2**self.num_bits-1
        delta = (torch.max(data)-torch.min(data))/(nLevels+1)

        index0 = 2**self.num_bits/2 + torch.sign(data)*torch.minimum(torch.tensor(2**self.num_bits/2).to(data.device),torch.ceil(torch.abs(data)/delta)) + (1-torch.sign(data))/2
        round_index = (2*index0 - 2**self.num_bits -1)/2
        y_quantized = round_index * delta

        #we also output the lower and upper bounds of the associated measurments. 
        #Note that these bounds are not measured but only for use of dequantization later
        y_upper = round_index + 1/2
        y_lower = round_index - 1/2  

        y_lower[torch.where(y_lower==-2**self.num_bits/2)] = -1e20
        y_upper[torch.where(y_upper==2**self.num_bits/2)] = 1e20

        y_lower = y_lower*delta
        y_upper = y_upper*delta
        
        return y_quantized
    
    def backproject(self, data, **kwargs):
        return data

class GaussianNoiseOp(NoiseOp):
    def __init__(self, std, device, noise_dir=None, random=False) -> None:
        super().__init__()
        self.noise_path = os.path.join(noise_dir, f"noise_{std}.pt")
           
        self.std = std
        self.device = device
        
        self.noise = None
        self.random = random
        if os.path.exists(self.noise_path) and not random:
            self.noise = torch.load(self.noise_path)
            # print("load noise")

    def forward(self, data, **kwargs):
        if self.random:
            noise = torch.randn_like(data) * self.std
            return data + noise
        else:
            if self.noise is None:
                noise = torch.randn_like(data) * self.std
                self.noise = noise.to(data.device)
                torch.save(noise, self.noise_path)
            else:
                noise = self.noise.to(data.device)
            return data + noise

class NoNoiseOp(NoiseOp):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data, **kwargs):
        return data

class PoissonNoiseOp(NoiseOp):
    def __init__(self, rate, device) -> None:
        super().__init__()
        self.rate = rate
        self.device = device

    def forward(self, data, **kwargs):
        # data = (data + 1.0) / 2.0
        # data = data.clamp(0.0, 1.0)
        # noise = self.poisson.sample(data.shape).to(self.device)
        # noise = (noise / 255.0) / self.rate
        # data = data * noise
        # data = data * 2.0 - 1.0
        # data = data.clamp(-1.0, 1.0)
        # return data
        # data = (data + 1.0) / 2.0
        # data = data.clamp(0, 1)
        data = torch.poisson(data * self.rate) / self.rate
        # data = data * 2.0 - 1.0
        # data = data.clamp(-1, 1)
        return data
# class PoissonProjectionModule(OperatorModule):
#     def __init__(
#         self,
#         num_angles: int = 64,
#         det_shape: Tuple[
#             int,
#         ] = (513,),
#         im_shape: Tuple[int, int] = (512, 512),
#         angles: float = np.pi,
#         cuda: bool = True,
#         photons_per_pixel=1e5
#     ):
#         self.num_angles = num_angles
#         self.det_shape = det_shape
#         self.im_shape = im_shape
#         self.cuda = cuda

#         recon_space = odl.uniform_discr(
#             min_pt=(-1, -1), max_pt=(1, 1), shape=self.im_shape
#         )
#         geometry = parallel_beam_geometry(
#             recon_space, num_angles=self.num_angles, det_shape=self.det_shape, angles=angles
#         )

#         if self.cuda:
#             ray_trafo = odl.tomo.RayTransform(recon_space, geometry, impl="astra_cuda")
#         else:
#             ray_trafo = odl.tomo.RayTransform(recon_space, geometry, impl="skimage")
        
#         self.photons_per_pixel = photons_per_pixel
#         self.ray_trafo = ray_trafo        
#         self.mu_water = 0.02
#         self.epsilon = 0.0001
#         nonlinear_operator = odl.ufunc_ops.exp(ray_trafo.range) * (- self.mu_water * ray_trafo)
        
#         super(PoissonProjectionModule, self).__init__(nonlinear_operator)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.ndim == 4:
#             out = super(PoissonProjectionModule, self).forward(x)
#             out_max = torch.max(out)
#             out_min = torch.min(out)
            
#             out_norm = (out - out_min) / (out_max - out_min)
#             out_norm = torch.poisson(out_norm * self.photons_per_pixel) / self.photons_per_pixel
#             out_norm = - torch.log(self.epsilon + out_norm) / self.mu_water
#             out = out_norm * (out_max - out_min) + out_min
            
#             return out
#         elif x.ndim == 5:
#             B, C, T, H, W = x.shape
#             x = x.view(B, T, H, W)
#             out = super(PoissonProjectionModule, self).forward(x)
#             out = out.view(B, C, T, H, W)
#             return out
              
#     def __repr__(self):
#         return f"ProjectionModule(num_angles={self.num_angles}, det_shape={self.det_shape}, im_shape={self.im_shape}, cuda={self.cuda})"

# class CTSparseViewOp(LinearOp):
#     def __init__(self, num_views, det_shape, im_shape, angles=np.pi) -> None:
#         super().__init__()
#         self.num_views = num_views
#         self.projector = PoissonProjectionModule(
#             num_angles=num_views, det_shape=det_shape, im_shape=im_shape, angles=angles
#         )
#         self.backprojector = BackProjectionModule(
#             num_angles=num_views, det_shape=det_shape, im_shape=im_shape, angles=angles
#         )
        
#     def forward(self, data):
#         return self.projector(data)
    
#     def backproject(self, data):
#         return self.backprojector(data)
