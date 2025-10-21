"""
Transmitter component for transmitter visualization.
"""
from typing import Dict, Any
import torch
from .base import Component, component, field


@component(name="Transmitter")
class TransmitterComponent(Component):
    """Transmitter component for transmitter objects."""

    # Transmitter parameters
    frequency = field(2.4e9, min=1e6, max=100e9, description="Frequency (Hz)")
    power = field(1.0, min=0.0, max=100.0, description="Transmit power (W)")
    color = field([0.0, 1.0, 0.0, 1.0], description="Transmitter color (RGBA)")

    def __init__(self, **kwargs):
        """Initialize transmitter component."""
        super().__init__(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert transmitter component to dictionary for serialization."""
        result = super().to_dict()
        return result

    def from_dict(self, data: Dict[str, Any]):
        """Load transmitter component from dictionary."""
        super().from_dict(data)
