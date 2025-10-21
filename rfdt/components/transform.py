"""
Transform component with PyTorch tensor support.
"""
from typing import List, Optional, Dict, Any
import torch
import math
from scipy.spatial.transform import Rotation
from .base import Component, component, field


@component(name="Transform")
class TransformComponent(Component):
    """Transform component with position, rotation, and scale."""
    
    # Component fields
    position = field(torch.zeros(3), description="World position")
    rotation = field(torch.zeros(3), description="Rotation in Euler angles (radians)")
    scale = field(torch.ones(3), description="Scale of the object")
    
    def __init__(self, **kwargs):
        """Initialize transform component."""
        # Handle special conversion cases for rotation
        if 'rotation' in kwargs:
            rotation_val = kwargs['rotation']
            # If it's a quaternion (4 values), convert to euler
            if isinstance(rotation_val, (list, tuple)) and len(rotation_val) == 4:
                kwargs['rotation'] = self._quaternion_to_euler_static(rotation_val)
            elif isinstance(rotation_val, torch.Tensor) and rotation_val.shape[0] == 4:
                kwargs['rotation'] = self._quaternion_to_euler_static(rotation_val.tolist())
        
        super().__init__(**kwargs)
        
        # Non-field attributes (parent-child hierarchy)
        self.parent: Optional['TransformComponent'] = None
        self.children: List['TransformComponent'] = []
    
    @staticmethod
    def _quaternion_to_euler_static(quaternion: List[float]) -> List[float]:
        """Convert quaternion to euler angles (static method for __init__)."""
        x, y, z, w = quaternion
        # Create SciPy rotation from quaternion [x, y, z, w]
        r = Rotation.from_quat([x, y, z, w])
        # Convert to euler angles with XYZ order
        euler = r.as_euler('XYZ', degrees=False)
        return euler.tolist()
    
    def _euler_to_quaternion(self, euler: Optional[torch.Tensor] = None, degrees: bool = False) -> torch.Tensor:
        """Convert euler angles (radians) to quaternion [x, y, z, w]."""
        if euler is None:
            euler = self.rotation
        
        if isinstance(euler, torch.Tensor):
            euler = euler.tolist()
        
        # Use SciPy with XYZ order
        r = Rotation.from_euler('XYZ', euler, degrees=degrees)
        quat = r.as_quat()  # Returns [x, y, z, w]
        return torch.tensor(quat, dtype=torch.float32)
    
    def _quaternion_to_euler(self, quaternion: Optional[torch.Tensor] = None, degrees: bool = False) -> torch.Tensor:
        """Convert quaternion to euler angles (radians)."""
        if quaternion is None:
            # Convert current rotation (euler) to quaternion first
            quaternion = self._euler_to_quaternion()
        
        if isinstance(quaternion, torch.Tensor):
            quaternion = quaternion.tolist()
        
        x, y, z, w = quaternion
        # Create SciPy rotation from quaternion [x, y, z, w]
        r = Rotation.from_quat([x, y, z, w])
        # Convert to euler angles with XYZ order
        euler = r.as_euler('XYZ', degrees=degrees)
        return torch.tensor(euler, dtype=torch.float32)
    
    def get_quaternion(self) -> torch.Tensor:
        """Get rotation as quaternion [x, y, z, w]."""
        return self._euler_to_quaternion()
    
    def set_quaternion(self, quaternion: List[float]):
        """Set rotation from quaternion [x, y, z, w]."""
        self.rotation = self._quaternion_to_euler(torch.tensor(quaternion, dtype=torch.float32))
    
    def get_transformation_matrix(self) -> torch.Tensor:
        """Get 4x4 transformation matrix."""
        # Create translation matrix
        T = torch.eye(4, dtype=torch.float32)
        T[0, 3] = self.position[0]
        T[1, 3] = self.position[1]
        T[2, 3] = self.position[2]
        
        # Get quaternion from euler angles
        quat = self._euler_to_quaternion()
        x, y, z, w = quat
        
        # Create rotation matrix from quaternion
        R = torch.eye(4, dtype=torch.float32)
        
        R[0, 0] = 1 - 2 * (y * y + z * z)
        R[0, 1] = 2 * (x * y - z * w)
        R[0, 2] = 2 * (x * z + y * w)
        
        R[1, 0] = 2 * (x * y + z * w)
        R[1, 1] = 1 - 2 * (x * x + z * z)
        R[1, 2] = 2 * (y * z - x * w)
        
        R[2, 0] = 2 * (x * z - y * w)
        R[2, 1] = 2 * (y * z + x * w)
        R[2, 2] = 1 - 2 * (x * x + y * y)
        
        # Create scale matrix
        S = torch.eye(4, dtype=torch.float32)
        S[0, 0] = self.scale[0]
        S[1, 1] = self.scale[1]
        S[2, 2] = self.scale[2]
        
        # Combine: T * R * S
        return T @ R @ S
    
    def set_parent(self, parent: Optional['TransformComponent']):
        """Set parent transform and update children lists."""
        if self.parent:
            self.parent.children.remove(self)
        
        self.parent = parent
        
        if parent:
            parent.children.append(self)
    
    def _trigger_callbacks(self, field_name: str, old_value, new_value):
        """Override to emit transform_updated instead of component_updated."""
        # Component-level callbacks (from base class)
        if field_name in self._callbacks:
            for callback in self._callbacks[field_name]:
                callback(self, field_name, old_value, new_value)

        # Global on_value_change callback
        if hasattr(self, 'on_value_change'):
            self.on_value_change(field_name, old_value, new_value)

        # Emit transform_updated event instead of component_updated
        if hasattr(self, '_owner') and self._owner:
            scene = getattr(self._owner, 'scene', None)
            if scene:
                # Emit transform_updated for Transform component
                scene._emit('object_updated',
                          object_id=self._owner.id,
                          object=self._owner,
                          update_type='transform')

    def to_dict(self) -> Dict[str, Any]:
        """Convert transform to dictionary for serialization."""
        # Get base dictionary from Component
        result = super().to_dict()

        # Override to ensure we return quaternion for compatibility with frontend
        quaternion = self.get_quaternion()
        result["values"]["rotation"] = quaternion.tolist()

        # Also include position and scale in the expected format
        result["values"]["position"] = self.position.tolist() if isinstance(self.position, torch.Tensor) else self.position
        result["values"]["scale"] = self.scale.tolist() if isinstance(self.scale, torch.Tensor) else self.scale

        return result