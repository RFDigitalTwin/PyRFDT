"""
Component system for RFDT.
"""
from .base import (
    Component,
    ComponentRegistry,
    component,
    field,
    button,
    ComponentFieldType,
    ComponentFieldMeta,
    ComponentButtonMeta,
    ABCComponentMeta
)
from .transform import TransformComponent
from .material import MaterialComponent
from .mesh import MeshComponent
from .light import LightComponent
from .camera import CameraComponent
from .radar import RadarComponent
from .transmitter import TransmitterComponent
from .receiver import ReceiverComponent

__all__ = [
    # Base classes
    'Component',
    'ComponentRegistry',
    'component',
    'field',
    'button',
    'ComponentFieldType',
    'ComponentFieldMeta',
    'ComponentButtonMeta',
    'ABCComponentMeta',
    # Components
    'TransformComponent',
    'MaterialComponent',
    'MeshComponent',
    'LightComponent',
    'CameraComponent',
    'RadarComponent',
    'TransmitterComponent',
    'ReceiverComponent',
]