"""
RFDT (Radio Frequency Digital Twin) package.

This package provides the core components for 3D scene management,
object manipulation, and WebSocket-based real-time communication.
"""

from .scene_object import SceneObject
from .scene import Scene
from .messages import MessageHandler, WebSocketMessage, serialize_scene, deserialize_scene
from .server import Server
from .editor import Editor
from .factory import ObjectFactory
from .ui.result_visualizer import ResultVisualizer
from .components import Component, ComponentRegistry, component, field
from .components import (
    TransformComponent,
    MaterialComponent,
    MeshComponent,
    LightComponent
)

__all__ = [
    'SceneObject',
    'Scene',
    'MessageHandler',
    'WebSocketMessage',
    'serialize_scene',
    'deserialize_scene',
    'Server',
    'Editor',
    'Component',
    'ComponentRegistry',
    'component',
    'field',
    'ObjectFactory',
    'TransformComponent',
    'MaterialComponent',
    'MeshComponent',
    'LightComponent',
    'ResultVisualizer'
]