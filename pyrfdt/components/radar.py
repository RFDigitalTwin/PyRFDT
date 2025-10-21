"""
Radar component for radar visualization.
"""
from typing import Dict, Any
import torch
import numpy as np
from PIL import Image
import io
import base64
from .base import Component, component, field, button

import invertwin as rfdt
from drjit.cuda.ad import Int as IntD, Float as FloatD, Matrix4f as Matrix4fD, Array3f as Vector3fD, Array2f as Vector2fD
from drjit.cuda import Float as FloatC, Matrix4f as Matrix4fC, Array3f as Vector3fC, Array3i as Vector3iC
from drjit.cuda import PCG32 as PCG32C, UInt64 as UInt64C
import drjit as dr

import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.autograd.forward_ad as fwAD

import trimesh

class SimpleRadar:
    def __init__(self):
        self.c0 = 299792458

        self.num_tx = 3
        self.num_rx = 4
        self.fc=77e9                    # start frequency (Hz)
        self.slope = 60.012             # frequency slope (MHz/us)
        self.adc_samples = 256
        self.adc_start_time = 6
        self.sample_rate = 4400         # (ksps)
        self.idle_time = 7              # (us), durtion between chirps
        self.ramp_end_time= 65          # (us)
        self.loop_per_frame = 128
        self.frame_per_second = 10

        self.num_doppler_bins = self.loop_per_frame
        self.num_range_bins = self.adc_samples
        self.num_angle_bins = 64

        self.range_resolution = (3e8 * self.sample_rate * 1e3) / (2 * self.slope * 1e12 * self.adc_samples)
        self.max_range = (300 * self.sample_rate) / (2 * self.slope * 1e3)
        self.doppler_resolution = 3e8 / (2 * self.fc * 1e9 * (self.idle_time + self.ramp_end_time) * 1e-6 * self.num_doppler_bins * self.num_tx)
        self.max_doppler = 3e8 / (4 * self.fc * 1e9 * (self.idle_time + self.ramp_end_time) * 1e-6 * self.num_tx)


        spacing = self.c0/self.fc/2
        self.tx_loc = np.array([[0,0,0],[4*spacing,0,0],[2*spacing,spacing,0]])
        self.rx_loc = np.array([[-6*spacing,0,0],[-5*spacing,0,0],[-4*spacing,0,0],[-3*spacing,0,0]])

        self._lambda = self.c0/self.fc


    def dechirp(self, x,xref):
        return xref * torch.conj(x)
    
    def FSPL(self, distance):
        return 20*torch.log10(distance) + 20*torch.log10(torch.tensor(self.fc)) + 20*torch.log10(torch.tensor(4*np.pi/self._lambda))


    def waveform(self, t,  phi=0):
        fc = (self.fc * t + 0.5 * (self.slope * 1e12) *  t * t )
        y = torch.exp(2j * torch.pi * fc) 
        return y

    def chirp(self,distance):
        t_sample = torch.arange(0,self.adc_samples,dtype=torch.float64)/(self.sample_rate*1e3) +self.adc_start_time*1e-6
        toa = 2* distance / self.c0
        
        tx = self.waveform(t_sample)
        rx = self.waveform(t_sample-toa.view(-1,1)) 
        rx = tx * torch.conj(rx)
        rx_combined = torch.sum(rx,axis=0)
        # sig = tx * torch.conj(rx_combined)
        return rx_combined

    
    def fft(self,distance):
        sig = self.chirp(distance)
        return torch.fft.fft(sig)






@component(name="Radar")
class RadarComponent(Component):
    """Radar component for radar objects."""


    # Radar parameters
    range = field(1.0, min=0.1, max=100.0, description="Detection range")
    fov = field(45.0, min=1.0, max=180.0, description="Field of view (degrees)")

    def __init__(self, **kwargs):
        """Initialize radar component."""
        super().__init__(**kwargs)

    @button(display_name="Render", description="Render radar detection visualization")
    def render(self):
        """Render scene from radar perspective using invertwin."""

        # ===== Validation =====
        if not self._owner or not hasattr(self._owner, 'scene'):
            print("[Radar] Not attached to a scene object")
            return

        scene = self._owner.scene
        transform = self._owner.get_component('Transform')
        if not transform:
            print("[Radar] No Transform component found")
            return "No Transform component"

        # ===== Scene Setup =====
        sc = rfdt.Scene()
        sc.opts.spp = 32
        sc.opts.sppe = 0
        sc.opts.sppse = 0
        sc.opts.height = 128
        sc.opts.width = 128

        # Initialize differentiable parameter for displacement
        P = FloatD(0.)
        dr.enable_grad(P)

        # Create integrator and radar processor
        integrator = rfdt.DemoIntegrator()
        radar = SimpleRadar()

        # Add materials
        sc.add_material(rfdt.DiffuseMaterial([0.9, 0.9, 0.9]), "basic")

        # ===== Setup Radar Sensor =====
        sensor = rfdt.Radar(60, 0.000001, 10000000.)

        # Create transformation matrix with parametric displacement in X direction
        base_transform = Matrix4fD(transform.get_transformation_matrix().numpy())
        translation = Matrix4fD([[1., 0., 0., P*100],
                                 [0., 1., 0., 0.],
                                 [0., 0., 1., 0.],
                                 [0., 0., 0., 1.]])
        sensor.to_world = translation @ base_transform
        sc.add_sensor(sensor)

        print(f"[Radar] Sensor positioned at {transform.position.tolist()} with rotation {transform.rotation.tolist()}")

        # Add environment map
        try:
            sc.add_EnvironmentMap("../../public/environment.exr",
                                dr.scalar.Matrix4f([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]),
                                1.0)
        except:
            print("[Radar] Could not load environment map")

        # ===== Add Scene Objects =====
        default_scene_dir = "scenes/default"

        for obj_id, obj in scene.objects.items():
            # Skip invisible objects, transmitters, receivers, and the radar itself
            if not obj.visible:
                continue
            if "transmitter" in obj_id.lower() or "receiver" in obj_id.lower():
                continue
            if obj_id == self._owner.id:
                continue

            # Get mesh and transform components
            mesh_comp = obj.get_component('Mesh')
            obj_transform = obj.get_component('Transform')
            if not mesh_comp or not obj_transform:
                continue

            # Create and load mesh
            rfdt_mesh = rfdt.Mesh()
            matrix = Matrix4fC(obj_transform.get_transformation_matrix().numpy())

            if mesh_comp.vertices is not None and len(mesh_comp.vertices) > 0:
                # Load from vertices and faces
                vertices = Vector3fC(mesh_comp.vertices.numpy())
                faces = Vector3iC(mesh_comp.faces.numpy())
                rfdt_mesh.load_raw(vertices, faces)
            elif mesh_comp.filename is not None:
                # Load from file
                mesh_path = f"{default_scene_dir}/{mesh_comp.filename}"
                try:
                    rfdt_mesh.load(mesh_path)
                except:
                    print(f"[Radar] Could not load mesh from {mesh_path}")
                    continue
            else:
                continue

            rfdt_mesh.to_world = matrix
            sc.add_mesh(rfdt_mesh, "basic", None)

        # ===== Render Scene =====
        with dr.suspend_grad():
            sc.configure()

        print("[Radar] Rendering radar view...")
        tau = integrator.cir_diff(sc, 0)

        # ===== Process Rendered Data =====
        # Convert to 2D image for visualization
        tau2D = np.array(tau)[:,0].reshape(sc.opts.width, sc.opts.height)
        tau_tensor = torch.tensor(tau2D)
        image_array = tau_tensor.numpy()

        # Flatten and filter data
        tau_tensor = tau_tensor.flatten()
        tau_tensor[tau_tensor == 0] = 1e-6
        mask = tau_tensor > 0.1
        tau_tensor = tau_tensor[mask]

        # ===== Compute Gradients =====
        dr.set_grad(P, 1.0)
        dr.forward_to(tau)
        diff_tau = dr.grad(tau)
        diff_tau_2D = np.array(diff_tau)[:,0].reshape(sc.opts.width, sc.opts.height)
        diff_tau_tensor = torch.tensor(diff_tau_2D).flatten()[mask]

        # print(f"[Radar] Gradient sum (DrJit): {dr.sum(diff_tau)}")
        # print(f"[Radar] Gradient sum (Torch): {diff_tau_tensor.sum()}")

        # ===== Compute Radar Signal with Forward AD =====
        with fwAD.dual_level():
            dual_tau = fwAD.make_dual(tau_tensor, diff_tau_tensor)
            dual_signal = radar.chirp(dual_tau)
            sig = fwAD.unpack_dual(dual_signal).primal
            sig_grad = fwAD.unpack_dual(dual_signal).tangent

        # print(f"[Radar] Signal shape: {sig.shape}")

        # ===== Send Results to Frontend =====
        if hasattr(scene, '_server') and scene._server and hasattr(scene._server, 'results'):
            server = scene._server
            title = f"Radar View)"

            # Display range-time heatmap
            server.results.imshow(image_array, title=title)

            # Plot radar signals
            x = torch.arange(0, sig.shape[0])
            server.results.plot(x, sig.real, title="Signal Real", xlabel="Time", ylabel="Real", color="orange")
            server.results.plot(x, sig.imag, title="Signal Imag", xlabel="Time", ylabel="Imaginary", color="cyan")

            # Plot signal gradients
            server.results.plot(x, sig_grad.real, title="Derivative Real", xlabel="Time", ylabel="Real Gradient", color="red")
            server.results.plot(x, sig_grad.imag, title="Derivative Imag", xlabel="Time", ylabel="Imag Gradient", color="blue")

            # Plot FFT
            fft = torch.fft.fft(sig)
            server.results.plot(x, torch.abs(fft), title="FFT", xlabel="Frequency", ylabel="Magnitude", color="green")

            server.results.commit("Radar Results")
        else:
            print("[Radar] Server or results not available")

        return f"Rendered radar view with Range={self.range:.1f}m, FOV={self.fov:.1f}°"

    def to_dict(self) -> Dict[str, Any]:
        """Convert radar component to dictionary for serialization."""
        result = super().to_dict()
        return result

    def from_dict(self, data: Dict[str, Any]):
        """Load radar component from dictionary."""
        super().from_dict(data)
