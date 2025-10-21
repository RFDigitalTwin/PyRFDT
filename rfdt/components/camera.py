"""
Camera component with rendering capability.
"""
import torch
import numpy as np
from PIL import Image
import io
import base64
from typing import Optional

from .base import Component, ComponentFieldType, field, button, component


@component(name="Camera")
class CameraComponent(Component):
    """Camera component for rendering and capturing scenes."""

    # Camera parameters
    fov = field(60.0, min=10.0, max=120.0, description="Field of view in degrees")
    exposure = field(1.0, min=0.1, max=10.0, description="Exposure value")
    focal_distance = field(5.0, min=0.1, max=100.0, description="Focal distance for depth of field")
    aperture = field(0.05, min=0.0, max=1.0, description="Aperture size for depth of field")
    background_color = field(torch.tensor([0.2, 0.3, 0.5, 1.0]), description="Background color")

    @button(display_name="Render", description="Capture and render the camera view")
    def render(self):
        """Render the camera view and send to results panel."""
        # Access the scene through the owner object
        if not self._owner or not hasattr(self._owner, 'scene'):
            print("Camera not attached to a scene object")
            return

        scene = self._owner.scene

        # Generate a random test image for demonstration
        # In a real implementation, this would capture the actual scene
        width, height = 800, 600

        # Create a gradient image with some random noise
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)

        # Use the background color from the component
        bg_color = self.background_color
        if isinstance(bg_color, torch.Tensor):
            bg_color = bg_color.numpy()

        # Create RGB channels with gradient and noise
        r = (xx * bg_color[0] + yy * (1 - bg_color[0])) * 255
        g = (yy * bg_color[1] + xx * (1 - bg_color[1])) * 255
        b = ((xx + yy) / 2 * bg_color[2]) * 255

        # Add some random "stars" or points
        np.random.seed(int(self.fov * 100 + self.exposure * 100))
        for _ in range(50):
            px, py = np.random.randint(0, width), np.random.randint(0, height)
            brightness = np.random.uniform(0.5, 1.0)
            r[py, px] = np.minimum(255, r[py, px] + brightness * 100)
            g[py, px] = np.minimum(255, g[py, px] + brightness * 100)
            b[py, px] = np.minimum(255, b[py, px] + brightness * 100)

        # Stack into RGB image (shape should be height x width x 3)
        image_array = np.stack([r, g, b], axis=2).astype(np.uint8)

        print(f"[Camera] Image shape: {image_array.shape}, dtype: {image_array.dtype}")
        print(f"[Camera] Image value range: {image_array.min()} - {image_array.max()}")
        print(f"[Camera] Background color: {bg_color}")

        # Create PIL image
        image = Image.fromarray(image_array)

        # Convert to base64 for sending to results panel
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Send to results panel if server is available
        if hasattr(scene, '_server') and scene._server:
            server = scene._server
            if hasattr(server, 'results'):
                # Create an image visualization in the results panel
                title = f"Camera View - {self._owner.name} (FOV: {self.fov:.1f}°, Exposure: {self.exposure:.2f})"
                server.results.imshow(image_array, title=title)
                server.results.commit(f"Rendered camera view for {self._owner.name}")
            else:
                print(f"[Camera] Server doesn't have results attribute")
        else:
            print(f"[Camera] Scene doesn't have _server or _server is None")

        return f"Rendered image with FOV={self.fov:.1f}°, Exposure={self.exposure:.2f}"

    @button(display_name="Reset", description="Reset camera to default settings")
    def reset_camera(self):
        """Reset camera settings to defaults."""
        self.fov = 60.0
        self.exposure = 1.0
        self.focal_distance = 5.0
        self.aperture = 0.05
        self.background_color = torch.tensor([0.2, 0.3, 0.5, 1.0])

        # Trigger updates for all fields
        # self.update()

        return "Camera reset to default settings"