"""
Material component for rendering properties.
"""
from typing import Optional, Dict, Any
import torch
from .base import Component, component, field


@component(name="Material") 
class MaterialComponent(Component):
    """Material component for rendering properties."""
    
    # Color automatically gets color-picker widget
    color = field(torch.tensor([0.5, 0.5, 0.5, 1.0]), description="Base color")
    
    # Float with min/max automatically gets slider widget (stored as tensor)
    metalness = field(torch.tensor(0.5), min=0.0, max=1.0, step=0.001, description="How metallic the surface is")
    roughness = field(torch.tensor(0.5), min=0.0, max=1.0, step=0.001, description="How rough the surface is")
    
    # Float without min/max gets draggable-number widget (stored as tensor)
    permittivity = field(torch.tensor(1.0), description="Dielectric constant")
    


    def to_dict_for_frontend(self) -> Dict[str, Any]:
        """Convert to dictionary format for frontend compatibility."""
        # Get color as hex string for frontend
        color_hex = "#808080"  # Default gray
        if isinstance(self.color, torch.Tensor) and self.color.shape[0] >= 3:
            r = int(self.color[0] * 255)
            g = int(self.color[1] * 255)
            b = int(self.color[2] * 255)
            color_hex = f"#{r:02x}{g:02x}{b:02x}"
        elif isinstance(self.color, str):
            color_hex = self.color
        
        # Convert tensor values to float for frontend
        metalness_val = self.metalness.item() if isinstance(self.metalness, torch.Tensor) else self.metalness
        roughness_val = self.roughness.item() if isinstance(self.roughness, torch.Tensor) else self.roughness
        permittivity_val = self.permittivity.item() if isinstance(self.permittivity, torch.Tensor) else self.permittivity
        
        return {
            "color": color_hex,
            "metalness": metalness_val,
            "roughness": roughness_val,
            "permittivity": permittivity_val
        }
    
    # Keep to_legacy_dict for now since SceneObject.to_dict() still uses it
    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for frontend compatibility."""
        return self.to_dict_for_frontend()