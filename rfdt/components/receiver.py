"""
Receiver component for receiver visualization.
"""
from typing import Dict, Any
import torch
from .base import Component, component, field


@component(name="Receiver")
class ReceiverComponent(Component):
    """Receiver component for receiver objects."""

    # Receiver parameters
    sensitivity = field(-90.0, min=-120.0, max=0.0, description="Sensitivity (dBm)")
    color = field([1.0, 0.0, 0.0, 1.0], description="Receiver color (RGBA)")

    def __init__(self, **kwargs):
        """Initialize receiver component."""
        super().__init__(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receiver component to dictionary for serialization."""
        result = super().to_dict()
        return result

    def from_dict(self, data: Dict[str, Any]):
        """Load receiver component from dictionary."""
        super().from_dict(data)
