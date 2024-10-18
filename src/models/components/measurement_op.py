from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import List, Tuple
import odl
from odl.contrib.torch import OperatorModule
from torch.nn import functional as F
from torch import nn
import torch
import scipy
import numpy as np
from torch_radon import RadonFanbeam, Radon

from src.models.components.utils import load_tiff_stack_with_metadata

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

class CTSparseViewOp(LinearOp):
    def __init__(self, num_views, det_shape, im_shape, angles=np.pi, filter_type='ram-lak') -> None:
        super().__init__()
        self.num_views = num_views
        self.projector = ProjectionModule(
            num_angles=num_views, det_shape=det_shape, im_shape=im_shape, angles=angles
        )
        self.backprojector = BackProjectionModule(
            num_angles=num_views, det_shape=det_shape, im_shape=im_shape, angles=angles, filter_type=filter_type
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