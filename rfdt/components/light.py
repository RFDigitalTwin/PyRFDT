"""
Light component for illumination.
"""
import torch
from .base import Component, component, field


@component(name="Light")
class LightComponent(Component):
    """Light component for illumination."""
    
    # String field with 'type' in name might get dropdown widget
    light_type = field("point", description="Type of light source")
    
    intensity = field(torch.tensor(1.0), min=0.0, max=10.0, step=0.1, description="Brightness")
    color = field(torch.tensor([1.0, 1.0, 1.0, 1.0]), description="Light color")
    range = field(torch.tensor(10.0), description="Maximum range of the light")
    spot_angle = field(torch.tensor(45.0), min=1.0, max=179.0, step=1.0, description="Spot light cone angle")
    cast_shadows = field(True, description="Whether this light casts shadows")