"""
Custom node system for node editor - allows defining nodes in Python.

This module provides a class-based system for defining custom nodes that can be
registered and used in the frontend node editor.
"""

from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass, field as dataclass_field
from abc import ABC, abstractmethod
import torch
import numpy as np


@dataclass
class PortDefinition:
    """Defines an input or output port for a node."""
    name: str
    type: str  # 'float', 'vector2', 'vector3', 'vector4', 'bool', 'string', 'color', 'texture2D', 'matrix'
    default: Any = None
    description: str = ""


@dataclass
class NodeDefinition:
    """Complete definition of a node type."""
    node_type: str  # Unique identifier for this node type
    display_name: str  # Display name in UI
    category: str  # Category for menu organization
    inputs: List[PortDefinition] = dataclass_field(default_factory=list)
    outputs: List[PortDefinition] = dataclass_field(default_factory=list)
    node_class: Type['Node'] = None  # Node class
    description: str = ""
    icon: str = ""  # Optional icon identifier
    color: str = ""  # Optional custom color
    is_builtin: bool = False  # Whether this is a built-in node (cannot be removed)


class Node(ABC):
    """
    Base class for custom nodes.

    Subclass this to create custom nodes.

    Example:
        class AddNode(Node):
            node_type = "add"
            display_name = "Add"
            category = "Math"

            inputs = [
                PortDefinition("a", "float", 0.0),
                PortDefinition("b", "float", 0.0)
            ]
            outputs = [
                PortDefinition("output", "float")
            ]

            def execute(self, a: float, b: float) -> float:
                return a + b
    """

    # Class attributes to be overridden
    node_type: str = ""
    display_name: str = ""
    category: str = "Custom"
    description: str = ""
    icon: str = ""
    color: str = ""
    is_builtin: bool = False

    # Port definitions
    inputs: List[PortDefinition] = []
    outputs: List[PortDefinition] = []

    @abstractmethod
    def execute(self, **inputs) -> Any:
        """
        Execute the node with given inputs.

        Args:
            **inputs: Input values mapped by port name

        Returns:
            Single value or tuple of values for multiple outputs
        """
        pass


class NodeRegistry:
    """Registry for all available node types."""

    def __init__(self):
        self._nodes: Dict[str, NodeDefinition] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(self, node_def: NodeDefinition) -> None:
        """Register a node definition."""
        if node_def.node_type in self._nodes:
            existing = self._nodes[node_def.node_type]
            if existing.is_builtin:
                raise ValueError(f"Cannot override built-in node type: {node_def.node_type}")

        self._nodes[node_def.node_type] = node_def

        # Update category index
        if node_def.category not in self._categories:
            self._categories[node_def.category] = []
        if node_def.node_type not in self._categories[node_def.category]:
            self._categories[node_def.category].append(node_def.node_type)

    def unregister(self, node_type: str) -> None:
        """Unregister a node type (cannot unregister built-in nodes)."""
        if node_type in self._nodes:
            if self._nodes[node_type].is_builtin:
                raise ValueError(f"Cannot unregister built-in node type: {node_type}")

            node_def = self._nodes[node_type]
            del self._nodes[node_type]

            # Update category index
            if node_def.category in self._categories:
                self._categories[node_def.category].remove(node_type)
                if not self._categories[node_def.category]:
                    del self._categories[node_def.category]

    def get(self, node_type: str) -> Optional[NodeDefinition]:
        """Get a node definition by type."""
        return self._nodes.get(node_type)

    def get_all(self) -> Dict[str, NodeDefinition]:
        """Get all registered nodes."""
        return self._nodes.copy()

    def get_by_category(self, category: str) -> List[NodeDefinition]:
        """Get all nodes in a category."""
        if category not in self._categories:
            return []
        return [self._nodes[node_type] for node_type in self._categories[category]]

    def get_categories(self) -> List[str]:
        """Get all category names."""
        return list(self._categories.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary for frontend."""
        categories = []
        for category in self._categories:
            items = []
            for node_type in self._categories[category]:
                node_def = self._nodes[node_type]
                items.append({
                    'nodeType': node_def.node_type,
                    'label': node_def.display_name,
                    'description': node_def.description,
                    'inputs': [{'name': p.name, 'type': p.type} for p in node_def.inputs],
                    'outputs': [{'name': p.name, 'type': p.type} for p in node_def.outputs],
                    'icon': node_def.icon,
                    'color': node_def.color
                })
            categories.append({
                'category': category,
                'items': items
            })
        return {'categories': categories}


# Global registry instance
_registry = NodeRegistry()


def register_node_class(cls: Type[Node]) -> Type[Node]:
    """
    Decorator for registering a class-based node.

    Example:
        @register_node_class
        class AddNode(Node):
            node_type = "add"
            display_name = "Add"
            category = "Math"

            inputs = [
                PortDefinition("a", "float", 0.0),
                PortDefinition("b", "float", 0.0)
            ]
            outputs = [PortDefinition("output", "float")]

            def execute(self, a: float, b: float) -> float:
                return a + b
    """
    if not issubclass(cls, Node):
        raise TypeError(f"{cls.__name__} must inherit from Node")

    if not cls.node_type:
        raise ValueError(f"{cls.__name__} must define node_type")

    # Get class docstring if no description provided
    description = cls.description or (cls.__doc__ or "").strip()

    # Create node definition
    node_def = NodeDefinition(
        node_type=cls.node_type,
        display_name=cls.display_name or cls.node_type,
        category=cls.category,
        inputs=list(cls.inputs),  # Make a copy
        outputs=list(cls.outputs),  # Make a copy
        node_class=cls,
        description=description,
        icon=cls.icon,
        color=cls.color,
        is_builtin=cls.is_builtin
    )

    # Register the node
    _registry.register(node_def)

    return cls


def execute_node(node_type: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a node with given inputs.

    Args:
        node_type: The type identifier of the node
        inputs: Dictionary mapping input port names to values

    Returns:
        Dictionary mapping output port names to values
    """
    node_def = _registry.get(node_type)
    if not node_def:
        raise ValueError(f"Unknown node type: {node_type}")

    if not node_def.node_class:
        raise ValueError(f"Node {node_type} has no associated class")

    # Prepare input arguments with defaults
    exec_args = {}
    for input_def in node_def.inputs:
        if input_def.name in inputs:
            exec_args[input_def.name] = inputs[input_def.name]
        elif input_def.default is not None:
            exec_args[input_def.name] = input_def.default

    # Instantiate and execute
    node_instance = node_def.node_class()
    result = node_instance.execute(**exec_args)

    # Format outputs
    outputs = {}
    if len(node_def.outputs) == 1:
        # Single output
        outputs[node_def.outputs[0].name] = result
    else:
        # Multiple outputs - result should be tuple
        if not isinstance(result, (tuple, list)):
            result = (result,)
        for i, output_def in enumerate(node_def.outputs):
            if i < len(result):
                outputs[output_def.name] = result[i]

    return outputs


def get_registry() -> NodeRegistry:
    """Get the global node registry."""
    return _registry


# Import system nodes to auto-register them
from . import system_nodes_class