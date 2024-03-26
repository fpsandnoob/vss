import odl
from odl.contrib.torch import OperatorModule
from torch import nn
from typing import List, Tuple
import torch
import numpy as np
# import torch_radon

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
    
# class ProjectionModuleTorch(nn.Module):
#     def __init__(self,
#         num_angles: int = 64,
#         det_shape: Tuple[
#             int,
#         ] = (513,),
#         im_shape: Tuple[int, int] = (512, 512),
#         angles: float = np.pi,
#         cuda: bool = True,) -> None:
#         super().__init__()

#         self.radon = torch_radon.Radon(im_shape[0], torch.linspace(0, torch.pi, num_angles).to('cuda' if cuda else 'cpu'), clip_to_circle=False)

#     def forward(self, x):
#         return self.radon.forward(x)
    
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
    
# class BackProjectionModuleTorch(nn.Module):
#     def __init__(
#         self,
#         num_angles: int = 64,
#         det_shape: Tuple[
#             int,
#         ] = (513,),
#         im_shape: Tuple[int, int] = (512, 512),
#         angles: float = np.pi,
#         filter_type: str = "Ram-Lak",
#         cuda: bool = True,
#     ):

class FISTATVBackProjectionModule(OperatorModule):
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

        grad = odl.Gradient(space)

        l1_norm = odl.solvers.L1Norm(space)
        regularizer = 0.015 * odl.solvers.L1Norm(grad.range)
        

        callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())
        gamma = 0.01
        x_acc = space.zero()
        odl.solvers.accelerated_proximal_gradient(
            x_acc, f=data_discrepancy, g=regularizer, niter=50, gamma=gamma,
            callback=callback)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out = super(BackProjectionModule, self).forward(x)
        return out

    def __repr__(self):
        return f"BackProjectionModule(num_angles={self.num_angles}, det_shape={self.det_shape}, im_shape={self.im_shape}, filter_type={self.filter_type}, cuda={self.cuda})"

def FISTATVBackProjection(y, projection_module):
    space = odl.uniform_discr(min_pt=(-20, -20), max_pt=(20, 20), shape=projection_module.im_shape)
    grad = odl.Gradient(space)

    l1_norm = odl.solvers.L1Norm(space)
    regularizer = 0.05 * odl.solvers.L1Norm(grad.range) * grad
    data_discrepancy = l1_norm.translated(y)

    # callback = odl.solvers.CallbackPrintIteration()
    gamma = 0.01
    x_acc = space.zero()
    odl.solvers.accelerated_proximal_gradient(
        x_acc, f=data_discrepancy, g=regularizer, niter=200, gamma=gamma,
        callback=None)

    return torch.from_numpy(np.asarray(x_acc)).unsqueeze(0).unsqueeze(0)
