"""
Enhanced component system with decorator-based registration and PyTorch support.
"""
from typing import Any, Dict, Optional, Type, Callable, List, Union, get_type_hints, get_origin, get_args
from dataclasses import dataclass, field as dataclass_field, fields
from enum import Enum
import torch
import inspect
from abc import ABC, abstractmethod
import numpy as np


class ComponentFieldType(Enum):
    """Field types supported by the component system."""
    FLOAT = "float"
    VECTOR2 = "vector2"
    VECTOR3 = "vector3"
    VECTOR4 = "vector4"
    COLOR = "color"
    BOOLEAN = "boolean"
    STRING = "string"
    MATRIX3 = "matrix3"
    MATRIX4 = "matrix4"
    TENSOR = "tensor"  # Generic tensor type


@dataclass
class ComponentFieldMeta:
    """Metadata for component fields."""
    name: str
    display_name: str
    field_type: ComponentFieldType
    default_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    ui_widget: Optional[str] = None
    description: Optional[str] = None
    readonly: bool = False


@dataclass
class ComponentButtonMeta:
    """Metadata for component buttons."""
    name: str
    display_name: str
    callback: Callable
    description: Optional[str] = None
    icon: Optional[str] = None
    style: Optional[str] = None


class ABCComponentMeta(type(ABC), type):
    """Combined metaclass for ABC and automatic component registration."""
    _registry: Dict[str, Type['Component']] = {}
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Don't register the base Component class
        if name != 'Component' and not name.startswith('_'):
            # Auto-register the component by class name first
            ABCComponentMeta._registry[name] = cls
            
            # Set component name (will be updated by decorator if needed)
            cls._component_name = kwargs.get('name', name)
            
            # Initialize class-level attributes
            cls._fields_meta = {}
            cls._buttons_meta = {}
            cls._callbacks = {}
            
            # Process field metadata from type hints will happen in __init_subclass__
            
        return cls
    
    @classmethod
    def get_registry(mcs) -> Dict[str, Type['Component']]:
        """Get all registered components."""
        return mcs._registry.copy()


class ComponentMeta(ABCComponentMeta):
    """Alias for backward compatibility."""
    pass


class Component(metaclass=ABCComponentMeta):
    """Base class for all components using PyTorch tensors."""

    _component_name: str = ""
    _fields_meta: Dict[str, ComponentFieldMeta] = {}
    _buttons_meta: Dict[str, ComponentButtonMeta] = {}
    _callbacks: Dict[str, List[Callable]] = {}
    
    def __init_subclass__(cls, **kwargs):
        """Called when a subclass is created."""
        super().__init_subclass__(**kwargs)
        # Don't process fields here - let the decorator do it
        # This avoids double-processing and field deletion issues
        pass
    
    def __init__(self, **kwargs):
        """Initialize component with default values."""
        self._values = {}
        self._owner = None  # Reference to the owning SceneObject
        self._callbacks = {}  # Instance-level callbacks
        self._button_callbacks = {}  # Instance-level button callbacks
        
        # Initialize fields with defaults
        for field_name, field_meta in self._fields_meta.items():
            value = kwargs.get(field_name, field_meta.default_value)
            self._set_field_value(field_name, value, notify=False)
    
    @classmethod
    def _process_fields(cls):
        """Process class fields and generate metadata."""
        cls._fields_meta = {}
        cls._buttons_meta = {}
        
        # Get type hints for the class (for typed fields)
        hints = get_type_hints(cls)
        
        # Process class attributes (includes Field objects)
        for attr_name in dir(cls):
            if attr_name.startswith('_') or attr_name in ['__class__', '__module__', '__dict__', '__weakref__', '__doc__']:
                continue
                
            # Get the field value
            field_val = getattr(cls, attr_name, None)
            
            # Skip methods and properties
            if callable(field_val) and not isinstance(field_val, Field):
                continue
            
            # Check if it's a Field instance from the field() function
            if isinstance(field_val, Field):
                # New field() style
                default_val = field_val.value
                field_info = field_val.metadata
                
                # Remove the Field object from the class to avoid conflicts
                delattr(cls, attr_name)
                # Try to get type hint if available
                attr_type = hints.get(attr_name, type(default_val))
            else:
                # Legacy style or plain value
                default_val = field_val
                # Check for legacy metadata (if using _fieldname_meta pattern)
                field_info = getattr(cls, f'_{attr_name}_meta', {})
                # Get type hint
                attr_type = hints.get(attr_name, type(default_val) if default_val is not None else str)
            
            # Determine field type - check default value first for better inference
            if default_val is not None and isinstance(default_val, torch.Tensor):
                # Infer type from tensor shape
                if default_val.numel() == 1:
                    field_type = ComponentFieldType.FLOAT
                elif default_val.shape[0] == 2:
                    field_type = ComponentFieldType.VECTOR2
                elif default_val.shape[0] == 3:
                    # Always treat 3D tensors as VECTOR3 (RGB colors will be handled separately)
                    field_type = ComponentFieldType.VECTOR3
                elif default_val.shape[0] == 4:
                    # Check if it's a color based on name
                    if 'color' in attr_name.lower():
                        field_type = ComponentFieldType.COLOR
                    else:
                        field_type = ComponentFieldType.VECTOR4
                else:
                    field_type = ComponentFieldType.TENSOR
            else:
                # Fall back to type hint inference
                field_type = cls._infer_field_type(attr_type)
            
            if default_val is None:
                default_val = cls._get_default_for_type(field_type)
            
            # Auto-generate display name from field name
            display_name = attr_name.replace('_', ' ').title()
            
            # Auto-infer widget based on field type and constraints
            widget = cls._infer_widget(field_type, field_info, attr_name)
            
            # Create field metadata
            cls._fields_meta[attr_name] = ComponentFieldMeta(
                name=attr_name,
                display_name=field_info.get('display_name', display_name),
                field_type=field_type,
                default_value=default_val,
                min_value=field_info.get('min', None),
                max_value=field_info.get('max', None),
                step=field_info.get('step', None),
                ui_widget=field_info.get('widget', widget),  # Use inferred widget if not specified
                description=field_info.get('description', None),
                readonly=field_info.get('readonly', False)
            )

        # Process buttons - look for methods decorated with @button
        for attr_name in dir(cls):
            if attr_name.startswith('_'):
                continue
            attr = getattr(cls, attr_name, None)
            if callable(attr) and hasattr(attr, '_is_button') and attr._is_button:
                button_meta = attr._button_meta
                cls._buttons_meta[button_meta['name']] = ComponentButtonMeta(
                    name=button_meta['name'],
                    display_name=button_meta['display_name'],
                    callback=button_meta['callback'],
                    description=button_meta.get('description'),
                    icon=button_meta.get('icon'),
                    style=button_meta.get('style')
                )
    
    @classmethod
    def _infer_widget(cls, field_type: ComponentFieldType, field_info: dict, field_name: str) -> str:
        """Automatically infer the best UI widget for a field."""
        # If widget is explicitly specified, use it
        if field_info.get('widget'):
            return field_info.get('widget')
        
        # Based on field type
        if field_type == ComponentFieldType.BOOLEAN:
            return "checkbox"
        elif field_type == ComponentFieldType.COLOR:
            return "color-picker"
        elif field_type == ComponentFieldType.STRING:
            # Check if it might be a dropdown based on name
            if 'type' in field_name.lower() or 'mode' in field_name.lower():
                return "dropdown"
            return "text-input"
        elif field_type == ComponentFieldType.FLOAT:
            # If has min/max, use slider, otherwise draggable number
            if field_info.get('min') is not None and field_info.get('max') is not None:
                return "slider"
            return "draggable-number"
        elif field_type in [ComponentFieldType.VECTOR2, ComponentFieldType.VECTOR3, ComponentFieldType.VECTOR4]:
            return "vector-input"
        elif field_type in [ComponentFieldType.MATRIX3, ComponentFieldType.MATRIX4]:
            return "matrix-input"
        else:
            return "text-input"  # Default fallback
    
    @classmethod
    def _infer_field_type(cls, python_type) -> ComponentFieldType:
        """Infer component field type from Python type hint."""
        # Handle Optional types
        origin = get_origin(python_type)
        if origin is Union:
            args = get_args(python_type)
            if len(args) == 2 and type(None) in args:
                python_type = args[0] if args[1] is type(None) else args[1]
        
        # Direct type mappings
        if python_type == float:
            return ComponentFieldType.FLOAT
        elif python_type == bool:
            return ComponentFieldType.BOOLEAN
        elif python_type == str:
            return ComponentFieldType.STRING
        elif python_type == torch.Tensor:
            return ComponentFieldType.TENSOR
        
        # Check for tensor-like types with shape hints
        if hasattr(python_type, '__origin__'):
            origin = get_origin(python_type)
            args = get_args(python_type)
            
            if origin == torch.Tensor:
                # Could parse shape from args if needed
                return ComponentFieldType.TENSOR
        
        # Default mappings for common names
        type_name = str(python_type)
        if 'Vector2' in type_name or 'Vec2' in type_name:
            return ComponentFieldType.VECTOR2
        elif 'Vector3' in type_name or 'Vec3' in type_name:
            return ComponentFieldType.VECTOR3
        elif 'Vector4' in type_name or 'Vec4' in type_name:
            return ComponentFieldType.VECTOR4
        elif 'Color' in type_name:
            return ComponentFieldType.COLOR
        elif 'Matrix3' in type_name or 'Mat3' in type_name:
            return ComponentFieldType.MATRIX3
        elif 'Matrix4' in type_name or 'Mat4' in type_name:
            return ComponentFieldType.MATRIX4
        
        # Default to tensor for unknown types
        return ComponentFieldType.TENSOR
    
    @classmethod
    def _get_default_for_type(cls, field_type: ComponentFieldType) -> Any:
        """Get default value for a field type."""
        if field_type == ComponentFieldType.FLOAT:
            return torch.tensor(0.0, dtype=torch.float32)
        elif field_type == ComponentFieldType.VECTOR2:
            return torch.zeros(2, dtype=torch.float32)
        elif field_type == ComponentFieldType.VECTOR3:
            return torch.zeros(3, dtype=torch.float32)
        elif field_type == ComponentFieldType.VECTOR4:
            return torch.zeros(4, dtype=torch.float32)
        elif field_type == ComponentFieldType.COLOR:
            return torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)  # RGBA white
        elif field_type == ComponentFieldType.BOOLEAN:
            return False
        elif field_type == ComponentFieldType.STRING:
            return ""
        elif field_type == ComponentFieldType.MATRIX3:
            return torch.eye(3, dtype=torch.float32)
        elif field_type == ComponentFieldType.MATRIX4:
            return torch.eye(4, dtype=torch.float32)
        else:
            return torch.zeros(1, dtype=torch.float32)
    
    def _set_field_value(self, field_name: str, value: Any, notify: bool = True):
        """Set a field value with type conversion to PyTorch tensor."""
        if field_name not in self._fields_meta:
            raise AttributeError(f"Component {self._component_name} has no field '{field_name}'")
        
        field_meta = self._fields_meta[field_name]
        
        if field_meta.readonly:
            raise AttributeError(f"Field '{field_name}' is readonly")
        
        # Convert to tensor if needed
        tensor_value = self._convert_to_tensor(value, field_meta.field_type)
        
        # Validate value
        if not self._validate_field_value(tensor_value, field_meta):
            raise ValueError(f"Invalid value for field '{field_name}'")
        
        # Store old value for callbacks
        old_value = self._values.get(field_name)

        # Set the value
        self._values[field_name] = tensor_value

        # Notify callbacks (always trigger if notify is True, even for first-time sets)
        if notify:
            self._trigger_callbacks(field_name, old_value, tensor_value)
    
    def _convert_to_tensor(self, value: Any, field_type: ComponentFieldType) -> Any:
        """Convert value to appropriate PyTorch tensor."""
        if field_type == ComponentFieldType.FLOAT:
            if isinstance(value, torch.Tensor):
                # Keep as tensor for consistency
                if value.numel() == 1:
                    return value.squeeze()  # Remove extra dimensions but keep as tensor
                return value[0:1]  # Keep first element as 1D tensor
            # Convert scalar to tensor
            return torch.tensor(float(value), dtype=torch.float32)
        
        elif field_type == ComponentFieldType.BOOLEAN:
            if isinstance(value, torch.Tensor):
                return bool(value.item()) if value.numel() == 1 else bool(value[0])
            return bool(value)
        
        elif field_type == ComponentFieldType.STRING:
            return str(value)
        
        elif field_type in [ComponentFieldType.VECTOR2, ComponentFieldType.VECTOR3, 
                           ComponentFieldType.VECTOR4, ComponentFieldType.COLOR]:
            expected_dim = {
                ComponentFieldType.VECTOR2: 2,
                ComponentFieldType.VECTOR3: 3,
                ComponentFieldType.VECTOR4: 4,
                ComponentFieldType.COLOR: 4  # RGBA
            }[field_type]
            
            if isinstance(value, torch.Tensor):
                if value.shape[0] != expected_dim:
                    # Special case: allow RGB (3) to be converted to RGBA (4) for color
                    if field_type == ComponentFieldType.COLOR and value.shape[0] == 3:
                        return torch.cat([value.float(), torch.tensor([1.0], dtype=torch.float32)])
                    raise ValueError(f"Expected tensor of size {expected_dim}, got {value.shape[0]}")
                return value.float()
            elif isinstance(value, (list, tuple, np.ndarray)):
                if len(value) == 3 and field_type == ComponentFieldType.COLOR:
                    # Convert RGB to RGBA
                    value = list(value) + [1.0]
                if len(value) != expected_dim:
                    raise ValueError(f"Expected {expected_dim} values, got {len(value)}")
                return torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, dict):
                # Handle dict format like {"x": 1, "y": 2, "z": 3}
                if field_type == ComponentFieldType.VECTOR2:
                    return torch.tensor([value.get('x', 0), value.get('y', 0)], dtype=torch.float32)
                elif field_type == ComponentFieldType.VECTOR3:
                    return torch.tensor([value.get('x', 0), value.get('y', 0), value.get('z', 0)], dtype=torch.float32)
                elif field_type == ComponentFieldType.VECTOR4:
                    return torch.tensor([value.get('x', 0), value.get('y', 0), 
                                        value.get('z', 0), value.get('w', 0)], dtype=torch.float32)
                elif field_type == ComponentFieldType.COLOR:
                    # Handle color as hex string or RGBA dict
                    if 'r' in value:
                        return torch.tensor([value.get('r', 1), value.get('g', 1), 
                                            value.get('b', 1), value.get('a', 1)], dtype=torch.float32)
            elif isinstance(value, str) and field_type == ComponentFieldType.COLOR:
                # Handle hex color string
                return self._hex_to_tensor(value)
        
        elif field_type in [ComponentFieldType.MATRIX3, ComponentFieldType.MATRIX4]:
            expected_size = 3 if field_type == ComponentFieldType.MATRIX3 else 4
            
            if isinstance(value, torch.Tensor):
                if value.shape != (expected_size, expected_size):
                    raise ValueError(f"Expected {expected_size}x{expected_size} matrix")
                return value.float()
            elif isinstance(value, (list, np.ndarray)):
                tensor = torch.tensor(value, dtype=torch.float32)
                if tensor.shape != (expected_size, expected_size):
                    # Try to reshape if it's a flat array
                    if tensor.numel() == expected_size * expected_size:
                        tensor = tensor.reshape(expected_size, expected_size)
                    else:
                        raise ValueError(f"Expected {expected_size}x{expected_size} matrix")
                return tensor
        
        # Default: try to convert to tensor
        if isinstance(value, torch.Tensor):
            return value.float()
        return torch.tensor(value, dtype=torch.float32)
    
    def _hex_to_tensor(self, hex_color: str) -> torch.Tensor:
        """Convert hex color string to RGBA tensor."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
            return torch.tensor([r, g, b, 1.0], dtype=torch.float32)
        elif len(hex_color) == 8:
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
            a = int(hex_color[6:8], 16) / 255.0
            return torch.tensor([r, g, b, a], dtype=torch.float32)
        else:
            return torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    
    def _validate_field_value(self, value: Any, field_meta: ComponentFieldMeta) -> bool:
        """Validate a field value against its constraints."""
        if field_meta.field_type == ComponentFieldType.FLOAT:
            if field_meta.min_value is not None and value < field_meta.min_value:
                return False
            if field_meta.max_value is not None and value > field_meta.max_value:
                return False
        return True
    
    def _trigger_callbacks(self, field_name: str, old_value: Any, new_value: Any):
        """Trigger callbacks for field changes."""
        # Component-level callbacks
        if field_name in self._callbacks:
            for callback in self._callbacks[field_name]:
                callback(self, field_name, old_value, new_value)

        # Global on_value_change callback
        if hasattr(self, 'on_value_change'):
            self.on_value_change(field_name, old_value, new_value)

        # Automatically sync to frontend when value changes
        if hasattr(self, '_owner') and self._owner:
            scene = getattr(self._owner, 'scene', None)
            if scene:
                # Get field metadata
                field_meta = self._fields_meta.get(field_name)

                # Convert value for serialization
                value = new_value
                if field_meta and field_meta.field_type == ComponentFieldType.COLOR:
                    # Convert color to hex string for frontend
                    if isinstance(value, torch.Tensor) and value.shape[0] >= 3:
                        r = int(value[0].item() * 255)
                        g = int(value[1].item() * 255)
                        b = int(value[2].item() * 255)
                        value = f"#{r:02x}{g:02x}{b:02x}"
                    elif isinstance(value, str):
                        value = value  # Already a string
                    else:
                        value = "#808080"  # Default gray
                elif hasattr(value, 'tolist'):
                    value = value.tolist()
                elif hasattr(value, 'item'):
                    value = value.item()

                scene._emit('component_updated',
                          object_id=self._owner.id,
                          component_name=self._component_name,
                          field_name=field_name,
                          value=value)
    
    def __getattr__(self, name: str) -> Any:
        """Get field value using dot notation."""
        if name.startswith('_'):
            # Access private attributes normally
            return object.__getattribute__(self, name)
        
        # Check if we have the field metadata (instance is initialized)
        if hasattr(self, '_fields_meta') and hasattr(self, '_values') and name in self._fields_meta:
            return self._values.get(name, self._fields_meta[name].default_value)
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any):
        """Set field value using dot notation."""
        if name.startswith('_') or name in ['_values', '_owner', '_callbacks', '_fields_meta', '_component_name']:
            # Set private attributes normally
            object.__setattr__(self, name, value)
        elif hasattr(self, '_fields_meta') and name in self._fields_meta:
            self._set_field_value(name, value)
        else:
            object.__setattr__(self, name, value)
    
    def add_callback(self, field_name: str, callback: Callable):
        """Add a callback for field changes."""
        if field_name not in self._callbacks:
            self._callbacks[field_name] = []
        self._callbacks[field_name].append(callback)
    
    def remove_callback(self, field_name: str, callback: Callable):
        """Remove a callback for field changes."""
        if field_name in self._callbacks:
            self._callbacks[field_name].remove(callback)
    
    def update(self, field_name: str = None):
        """Manually trigger synchronization to frontend.

        Args:
            field_name: Optional specific field to update. If None, updates all fields one by one.
        """
        if hasattr(self, '_owner') and self._owner:
            # Get the scene from the owner
            scene = getattr(self._owner, 'scene', None)
            if scene:
                if field_name:
                    # Update specific field
                    if field_name in self._values:
                        # Get field metadata
                        field_meta = self._fields_meta.get(field_name)

                        # Convert value for serialization
                        value = self._values[field_name]
                        if field_meta and field_meta.field_type == ComponentFieldType.COLOR:
                            # Convert color to hex string for frontend
                            if isinstance(value, torch.Tensor) and value.shape[0] >= 3:
                                r = int(value[0].item() * 255)
                                g = int(value[1].item() * 255)
                                b = int(value[2].item() * 255)
                                value = f"#{r:02x}{g:02x}{b:02x}"
                            elif isinstance(value, str):
                                value = value  # Already a string
                            else:
                                value = "#808080"  # Default gray
                        elif hasattr(value, 'tolist'):
                            value = value.tolist()
                        elif hasattr(value, 'item'):
                            value = value.item()

                        scene._emit('component_updated',
                                  object_id=self._owner.id,
                                  component_name=self._component_name,
                                  field_name=field_name,
                                  value=value)
                else:
                    # Send updates for all fields
                    for field_name, field_value in self._values.items():
                        # Get field metadata
                        field_meta = self._fields_meta.get(field_name)

                        # Convert value for serialization
                        value = field_value
                        if field_meta and field_meta.field_type == ComponentFieldType.COLOR:
                            # Convert color to hex string for frontend
                            if isinstance(value, torch.Tensor) and value.shape[0] >= 3:
                                r = int(value[0].item() * 255)
                                g = int(value[1].item() * 255)
                                b = int(value[2].item() * 255)
                                value = f"#{r:02x}{g:02x}{b:02x}"
                            elif isinstance(value, str):
                                value = value  # Already a string
                            else:
                                value = "#808080"  # Default gray
                        elif hasattr(value, 'tolist'):
                            value = value.tolist()
                        elif hasattr(value, 'item'):
                            value = value.item()

                        scene._emit('component_updated',
                                  object_id=self._owner.id,
                                  component_name=self._component_name,
                                  field_name=field_name,
                                  value=value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert component to dictionary for serialization."""
        result = {
            "name": self._component_name,
            "display_name": self._component_name.replace('_', ' ').title(),
            "values": {},
            "fields": [],  # Include field definitions for frontend
            "buttons": []  # Include button definitions for frontend
        }

        # Add field definitions
        for field_name, field_meta in self._fields_meta.items():
            field_def = {
                "name": field_name,
                "display_name": field_meta.display_name,
                "field_type": field_meta.field_type.value,
                "min_value": field_meta.min_value,
                "max_value": field_meta.max_value,
                "step": field_meta.step,
                "ui_widget": field_meta.ui_widget,
                "description": field_meta.description,
                "readonly": field_meta.readonly
            }
            result["fields"].append(field_def)

        # Add button definitions
        for button_name, button_meta in self._buttons_meta.items():
            button_def = {
                "name": button_name,
                "display_name": button_meta.display_name,
                "description": button_meta.description,
                "icon": button_meta.icon,
                "style": button_meta.style
            }
            result["buttons"].append(button_def)
        
        # Add values
        for field_name, value in self._values.items():
            field_meta = self._fields_meta[field_name]

            # Handle color fields specially
            if field_meta.field_type == ComponentFieldType.COLOR:
                # Convert to hex string for frontend
                if isinstance(value, torch.Tensor) and value.shape[0] >= 3:
                    r = int(value[0].item() * 255)
                    g = int(value[1].item() * 255)
                    b = int(value[2].item() * 255)
                    result["values"][field_name] = f"#{r:02x}{g:02x}{b:02x}"
                elif isinstance(value, list) and len(value) >= 3:
                    r = int(value[0] * 255)
                    g = int(value[1] * 255)
                    b = int(value[2] * 255)
                    result["values"][field_name] = f"#{r:02x}{g:02x}{b:02x}"
                elif isinstance(value, str):
                    result["values"][field_name] = value
                else:
                    result["values"][field_name] = "#808080"  # Default gray
            elif isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    result["values"][field_name] = value.item()
                else:
                    result["values"][field_name] = value.tolist()
            else:
                result["values"][field_name] = value
        
        return result
    
    def from_dict(self, data: Dict[str, Any]):
        """Load component state from dictionary."""
        values = data.get("values", {})
        for field_name, value in values.items():
            if field_name in self._fields_meta:
                self._set_field_value(field_name, value, notify=False)
    
    @classmethod
    def get_definition(cls) -> Dict[str, Any]:
        """Get component definition for frontend."""
        return {
            "name": cls._component_name,
            "display_name": cls._component_name.replace('_', ' ').title(),
            "removable": True,
            "fields": [
                {
                    "name": field_meta.name,
                    "display_name": field_meta.display_name,
                    "field_type": field_meta.field_type.value,
                    "default_value": cls._serialize_default(field_meta.default_value, field_meta.field_type),
                    "min_value": field_meta.min_value,
                    "max_value": field_meta.max_value,
                    "step": field_meta.step,
                    "ui_widget": field_meta.ui_widget,
                    "description": field_meta.description,
                    "readonly": field_meta.readonly
                }
                for field_meta in cls._fields_meta.values()
            ],
            "buttons": [
                {
                    "name": button_meta.name,
                    "display_name": button_meta.display_name,
                    "description": button_meta.description,
                    "icon": button_meta.icon,
                    "style": button_meta.style
                }
                for button_meta in cls._buttons_meta.values()
            ]
        }
    
    @classmethod
    def _serialize_default(cls, value: Any, field_type: ComponentFieldType) -> Any:
        """Serialize default value for frontend."""
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            else:
                return value.tolist()
        elif field_type == ComponentFieldType.COLOR and isinstance(value, torch.Tensor):
            # Convert to hex
            r = int(value[0] * 255)
            g = int(value[1] * 255)
            b = int(value[2] * 255)
            return f"#{r:02x}{g:02x}{b:02x}"
        return value
    
    def on_value_change(self, field_name: str, old_value: Any, new_value: Any):
        """Override this method to handle value changes."""
        pass

    def click_button(self, button_name: str) -> Any:
        """Click a button and execute its callback."""
        if button_name in self._buttons_meta:
            button_meta = self._buttons_meta[button_name]
            # Call the button's callback with self as the argument
            return button_meta.callback(self)
        else:
            raise AttributeError(f"Component {self._component_name} has no button '{button_name}'")


# Component Registry singleton
class ComponentRegistry:
    """Global registry for all components."""
    
    @classmethod
    def get_all(cls) -> List[Type[Component]]:
        """Get all registered component classes."""
        return list(ABCComponentMeta.get_registry().values())
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[Component]]:
        """Get a component class by name."""
        return ABCComponentMeta.get_registry().get(name)
    
    @classmethod
    def create_instance(cls, name: str, **kwargs) -> Optional[Component]:
        """Create an instance of a component."""
        component_class = cls.get(name)
        if component_class:
            return component_class(**kwargs)
        return None
    
    @classmethod
    def get_definitions(cls) -> List[Dict[str, Any]]:
        """Get all component definitions for frontend."""
        return [comp_cls.get_definition() for comp_cls in cls.get_all()]


def component(name: Optional[str] = None, display_name: Optional[str] = None):
    """Decorator for registering components."""
    def decorator(cls):
        # Set component metadata
        cls._component_name = name or cls.__name__
        cls._display_name = display_name or cls._component_name.replace('_', ' ').title()
        
        # Update registry with the decorated name
        if name and name != cls.__name__:
            # Remove from old name if exists
            if cls.__name__ in ABCComponentMeta._registry:
                del ABCComponentMeta._registry[cls.__name__]
            # Register with new name
            ABCComponentMeta._registry[name] = cls
        
        # Process fields after decorator has set up metadata
        if hasattr(cls, '_process_fields'):
            cls._process_fields()
        
        # Ensure the class uses our metaclass
        if not isinstance(cls, ABCComponentMeta):
            # Create a new class with our metaclass
            new_cls = ABCComponentMeta(
                cls.__name__,
                (Component,) + cls.__bases__,
                dict(cls.__dict__),
                name=cls._component_name
            )
            # Register with decorated name
            if name:
                ABCComponentMeta._registry[name] = new_cls
            # Process fields for the new class
            if hasattr(new_cls, '_process_fields'):
                new_cls._process_fields()
            return new_cls
        
        return cls
    
    return decorator


class Field:
    """Field descriptor for component fields with metadata."""
    def __init__(self, default_value, **metadata):
        self.value = default_value
        self.metadata = metadata
    
    def __repr__(self):
        return f"Field(value={self.value}, metadata={self.metadata})"


def field(default_value,
          min: Optional[float] = None,
          max: Optional[float] = None,
          step: Optional[float] = None,
          description: Optional[str] = None,
          readonly: bool = False):
    """Create a field with metadata and automatic widget inference.

    Usage:
        class MyComponent(Component):
            speed = field(1.0, min=0, max=10, description="Movement speed")
            position = field(torch.zeros(3), description="World position")
    """
    return Field(default_value, min=min, max=max, step=step,
                 description=description, readonly=readonly)


def button(display_name: Optional[str] = None,
          description: Optional[str] = None,
          icon: Optional[str] = None,
          style: Optional[str] = None):
    """Decorator to mark a method as a button.

    Usage:
        class MyComponent(Component):
            @button(display_name="Render", description="Render the camera view")
            def render(self):
                # Button click logic here
                pass
    """
    def decorator(func):
        # Store button metadata on the function
        func._is_button = True
        func._button_meta = {
            'name': func.__name__,
            'display_name': display_name or func.__name__.replace('_', ' ').title(),
            'description': description,
            'icon': icon,
            'style': style,
            'callback': func
        }
        return func
    return decorator


# Backward compatibility alias
def field_meta(display_name: Optional[str] = None,
               min: Optional[float] = None,
               max: Optional[float] = None,
               step: Optional[float] = None,
               widget: Optional[str] = None,
               description: Optional[str] = None,
               readonly: bool = False):
    """Legacy field metadata (for backward compatibility)."""
    return {
        'display_name': display_name,
        'min': min,
        'max': max,
        'step': step,
        'widget': widget,
        'description': description,
        'readonly': readonly
    }