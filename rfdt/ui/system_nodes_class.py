"""
Class-based system nodes - the main system nodes using class-based definitions.

This module contains all built-in nodes defined using the class-based system.
"""

import numpy as np
import torch
from typing import Union, Any, Dict
from .nodes import Node, PortDefinition, register_node_class


# ============================================================================
# BASIC MATH OPERATORS
# ============================================================================

@register_node_class
class AddNode(Node):
    """Add two numbers: a + b"""
    node_type = "add"
    display_name = "Add"
    category = "Math"
    is_builtin = True

    inputs = [
        PortDefinition("a", "float", 0.0),
        PortDefinition("b", "float", 0.0)
    ]
    outputs = [PortDefinition("output", "float")]

    def execute(self, a: float, b: float) -> float:
        return a + b


@register_node_class
class SubtractNode(Node):
    """Subtract b from a: a - b"""
    node_type = "subtract"
    display_name = "Subtract"
    category = "Math"
    is_builtin = True

    inputs = [
        PortDefinition("a", "float", 0.0),
        PortDefinition("b", "float", 0.0)
    ]
    outputs = [PortDefinition("output", "float")]

    def execute(self, a: float, b: float) -> float:
        return a - b


@register_node_class
class MultiplyNode(Node):
    """Multiply two numbers: a × b"""
    node_type = "multiply"
    display_name = "Multiply"
    category = "Math"
    is_builtin = True

    inputs = [
        PortDefinition("a", "float", 0.0),
        PortDefinition("b", "float", 0.0)
    ]
    outputs = [PortDefinition("output", "float")]

    def execute(self, a: float, b: float) -> float:
        return a * b


@register_node_class
class DivideNode(Node):
    """Divide a by b: a ÷ b"""
    node_type = "divide"
    display_name = "Divide"
    category = "Math"
    is_builtin = True

    inputs = [
        PortDefinition("a", "float", 0.0),
        PortDefinition("b", "float", 1.0)
    ]
    outputs = [PortDefinition("output", "float")]

    def execute(self, a: float, b: float) -> float:
        return a / b if b != 0 else 0.0


@register_node_class
class PowerNode(Node):
    """Raise base to the power of exponent"""
    node_type = "power"
    display_name = "Power"
    category = "Math"
    is_builtin = True

    inputs = [
        PortDefinition("base", "float", 0.0),
        PortDefinition("exponent", "float", 2.0)
    ]
    outputs = [PortDefinition("output", "float")]

    def execute(self, base: float, exponent: float) -> float:
        return float(base ** exponent)


@register_node_class
class SqrtNode(Node):
    """Square root"""
    node_type = "sqrt"
    display_name = "Sqrt"
    category = "Math"
    is_builtin = True

    inputs = [PortDefinition("x", "float", 0.0)]
    outputs = [PortDefinition("output", "float")]

    def execute(self, x: float) -> float:
        return float(np.sqrt(max(0, x)))


@register_node_class
class AbsNode(Node):
    """Absolute value"""
    node_type = "abs"
    display_name = "Abs"
    category = "Math"
    is_builtin = True

    inputs = [PortDefinition("x", "float", 0.0)]
    outputs = [PortDefinition("output", "float")]

    def execute(self, x: float) -> float:
        return abs(x)


@register_node_class
class MinNode(Node):
    """Minimum of two values"""
    node_type = "min"
    display_name = "Min"
    category = "Math"
    is_builtin = True

    inputs = [
        PortDefinition("a", "float", 0.0),
        PortDefinition("b", "float", 0.0)
    ]
    outputs = [PortDefinition("output", "float")]

    def execute(self, a: float, b: float) -> float:
        return min(a, b)


@register_node_class
class MaxNode(Node):
    """Maximum of two values"""
    node_type = "max"
    display_name = "Max"
    category = "Math"
    is_builtin = True

    inputs = [
        PortDefinition("a", "float", 0.0),
        PortDefinition("b", "float", 0.0)
    ]
    outputs = [PortDefinition("output", "float")]

    def execute(self, a: float, b: float) -> float:
        return max(a, b)


@register_node_class
class ClampNode(Node):
    """Clamp value between min and max"""
    node_type = "clamp"
    display_name = "Clamp"
    category = "Math"
    is_builtin = True

    inputs = [
        PortDefinition("value", "float", 0.5),
        PortDefinition("min_val", "float", 0.0),
        PortDefinition("max_val", "float", 1.0)
    ]
    outputs = [PortDefinition("output", "float")]

    def execute(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        return max(min_val, min(max_val, value))


@register_node_class
class LerpNode(Node):
    """Linear interpolation from a to b"""
    node_type = "lerp"
    display_name = "Lerp"
    category = "Math"
    is_builtin = True

    inputs = [
        PortDefinition("a", "float", 0.0),
        PortDefinition("b", "float", 1.0),
        PortDefinition("t", "float", 0.5)
    ]
    outputs = [PortDefinition("output", "float")]

    def execute(self, a: float, b: float, t: float) -> float:
        return a + (b - a) * t


# ============================================================================
# TRIGONOMETRY
# ============================================================================

@register_node_class
class SinNode(Node):
    """Sine of angle in radians"""
    node_type = "sin"
    display_name = "Sin"
    category = "Math"
    is_builtin = True

    inputs = [PortDefinition("angle", "float", 0.0)]
    outputs = [PortDefinition("output", "float")]

    def execute(self, angle: float) -> float:
        return float(np.sin(angle))


@register_node_class
class CosNode(Node):
    """Cosine of angle in radians"""
    node_type = "cos"
    display_name = "Cos"
    category = "Math"
    is_builtin = True

    inputs = [PortDefinition("angle", "float", 0.0)]
    outputs = [PortDefinition("output", "float")]

    def execute(self, angle: float) -> float:
        return float(np.cos(angle))


@register_node_class
class Atan2Node(Node):
    """Atan2(y, x) - angle from origin to point"""
    node_type = "atan2"
    display_name = "Atan2"
    category = "Math"
    is_builtin = True

    inputs = [
        PortDefinition("y", "float", 0.0),
        PortDefinition("x", "float", 1.0)
    ]
    outputs = [PortDefinition("output", "float")]

    def execute(self, y: float, x: float) -> float:
        return float(np.arctan2(y, x))


# ============================================================================
# LOGIC NODES
# ============================================================================

@register_node_class
class GreaterNode(Node):
    """Check if a > b"""
    node_type = "greater"
    display_name = "Greater"
    category = "Logic"
    is_builtin = True

    inputs = [
        PortDefinition("a", "float", 0.0),
        PortDefinition("b", "float", 0.0)
    ]
    outputs = [PortDefinition("output", "bool")]

    def execute(self, a: float, b: float) -> bool:
        return a > b


@register_node_class
class LessNode(Node):
    """Check if a < b"""
    node_type = "less"
    display_name = "Less"
    category = "Logic"
    is_builtin = True

    inputs = [
        PortDefinition("a", "float", 0.0),
        PortDefinition("b", "float", 0.0)
    ]
    outputs = [PortDefinition("output", "bool")]

    def execute(self, a: float, b: float) -> bool:
        return a < b


@register_node_class
class AndNode(Node):
    """Logical AND"""
    node_type = "and"
    display_name = "And"
    category = "Logic"
    is_builtin = True

    inputs = [
        PortDefinition("a", "bool", False),
        PortDefinition("b", "bool", False)
    ]
    outputs = [PortDefinition("output", "bool")]

    def execute(self, a: bool, b: bool) -> bool:
        return a and b


@register_node_class
class OrNode(Node):
    """Logical OR"""
    node_type = "or"
    display_name = "Or"
    category = "Logic"
    is_builtin = True

    inputs = [
        PortDefinition("a", "bool", False),
        PortDefinition("b", "bool", False)
    ]
    outputs = [PortDefinition("output", "bool")]

    def execute(self, a: bool, b: bool) -> bool:
        return a or b


@register_node_class
class NotNode(Node):
    """Logical NOT"""
    node_type = "not"
    display_name = "Not"
    category = "Logic"
    is_builtin = True

    inputs = [PortDefinition("a", "bool", False)]
    outputs = [PortDefinition("output", "bool")]

    def execute(self, a: bool) -> bool:
        return not a


# ============================================================================
# VECTOR/UTILITY NODES
# ============================================================================

@register_node_class
class NormalizeNode(Node):
    """Normalize vector to unit length"""
    node_type = "normalize"
    display_name = "Normalize"
    category = "Utility"
    is_builtin = True

    inputs = [PortDefinition("vec", "vector3")]
    outputs = [PortDefinition("output", "vector3")]

    def execute(self, vec) -> np.ndarray:
        length = np.linalg.norm(vec)
        return vec / length if length > 0 else vec


@register_node_class
class DotProductNode(Node):
    """Dot product of two vectors"""
    node_type = "dot"
    display_name = "Dot Product"
    category = "Utility"
    is_builtin = True

    inputs = [
        PortDefinition("a", "vector3"),
        PortDefinition("b", "vector3")
    ]
    outputs = [PortDefinition("output", "float")]

    def execute(self, a, b) -> float:
        return float(np.dot(a, b))


def register_all_class_nodes():
    """
    This function is called automatically when the module is imported.
    All nodes decorated with @register_node_class are already registered.
    """
    from .nodes import get_registry
    registry = get_registry()

    # Count class-based nodes
    class_node_count = sum(1 for node in registry.get_all().values() if node.node_class is not None)

    print(f"[Class Nodes] Registered {class_node_count} class-based nodes")
    return registry


# Auto-register when module is imported
_class_registry = register_all_class_nodes()
