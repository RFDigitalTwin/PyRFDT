"""
Factory class for creating scene objects with predefined configurations.
"""
from typing import List, Optional, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from .scene_object import SceneObject


class ObjectFactory:
    """Factory for creating scene objects with predefined configurations."""
    
    @staticmethod
    def create_cube(position: Optional[List[float]] = None, **kwargs) -> 'SceneObject':
        """Create a cube object.
        
        Args:
            position: Optional 3D position [x, y, z]
            **kwargs: Additional SceneObject parameters
        
        Returns:
            SceneObject configured as a cube
        """
        from .scene_object import SceneObject
        obj = SceneObject(
            name=kwargs.get('name', 'Cube'),
            mesh_type='Cube',
            **{k: v for k, v in kwargs.items() if k not in ['name', 'position']}
        )
        
        # Set position if provided
        if position:
            obj["Transform"].position = torch.tensor(position, dtype=torch.float32)
        
        # Ensure mesh component generates the cube
        if "Mesh" in obj._components:
            obj._components["Mesh"]._generate_primitive()
        
        return obj
    
    @staticmethod
    def create_sphere(position: Optional[List[float]] = None, **kwargs) -> 'SceneObject':
        """Create a sphere object.
        
        Args:
            position: Optional 3D position [x, y, z]
            **kwargs: Additional SceneObject parameters
        
        Returns:
            SceneObject configured as a sphere
        """
        from .scene_object import SceneObject
        obj = SceneObject(
            name=kwargs.get('name', 'Sphere'),
            mesh_type='Sphere',
            **{k: v for k, v in kwargs.items() if k not in ['name', 'position']}
        )
        
        # Set position if provided
        if position:
            obj["Transform"].position = torch.tensor(position, dtype=torch.float32)
        
        # Ensure mesh component generates the sphere
        if "Mesh" in obj._components:
            obj._components["Mesh"]._generate_primitive()
        
        return obj
    
    @staticmethod
    def create_cylinder(position: Optional[List[float]] = None, **kwargs) -> 'SceneObject':
        """Create a cylinder object.
        
        Args:
            position: Optional 3D position [x, y, z]
            **kwargs: Additional SceneObject parameters
        
        Returns:
            SceneObject configured as a cylinder
        """
        from .scene_object import SceneObject
        obj = SceneObject(
            name=kwargs.get('name', 'Cylinder'),
            mesh_type='Cylinder',
            **{k: v for k, v in kwargs.items() if k not in ['name', 'position']}
        )
        
        # Set position if provided
        if position:
            obj["Transform"].position = torch.tensor(position, dtype=torch.float32)
        
        # Ensure mesh component generates the cylinder
        if "Mesh" in obj._components:
            obj._components["Mesh"]._generate_primitive()
        
        return obj
    
    @staticmethod
    def create_empty(position: Optional[List[float]] = None, **kwargs) -> 'SceneObject':
        """Create an empty object (Transform only).
        
        Args:
            position: Optional 3D position [x, y, z]
            **kwargs: Additional SceneObject parameters
        
        Returns:
            SceneObject with only Transform component
        """
        from .scene_object import SceneObject
        obj = SceneObject(
            name=kwargs.get('name', 'Empty'),
            mesh_type='Empty',
            **{k: v for k, v in kwargs.items() if k not in ['name', 'position']}
        )
        
        # Remove Mesh and Material components for empty objects
        obj.remove_component("Mesh")
        obj.remove_component("Material")
        
        # Set position if provided
        if position:
            obj["Transform"].position = torch.tensor(position, dtype=torch.float32)
        
        return obj
    
    @staticmethod
    def create_light(position: Optional[List[float]] = None, 
                     light_type: str = "Point", **kwargs) -> 'SceneObject':
        """Create a light object.
        
        Args:
            position: Optional 3D position [x, y, z]
            light_type: Type of light ("Point", "Directional", "Spot")
            **kwargs: Additional SceneObject parameters
        
        Returns:
            SceneObject configured as a light
        """
        from .scene_object import SceneObject
        from .components.light import LightComponent
        
        obj = SceneObject(
            name=kwargs.get('name', f'{light_type} Light'),
            mesh_type='Empty',
            **{k: v for k, v in kwargs.items() if k not in ['name', 'position', 'light_type']}
        )
        
        # Remove Mesh component for lights
        obj.remove_component("Mesh")
        
        # Add Light component
        obj.add_component("Light", light_type=light_type)
        
        # Set position if provided
        if position:
            obj["Transform"].position = torch.tensor(position, dtype=torch.float32)
        
        return obj
    
    @staticmethod
    def create_camera(position: Optional[List[float]] = None, **kwargs) -> 'SceneObject':
        """Create a camera object.
        
        Args:
            position: Optional 3D position [x, y, z]
            **kwargs: Additional SceneObject parameters
        
        Returns:
            SceneObject configured as a camera
        """
        from .scene_object import SceneObject
        obj = SceneObject(
            name=kwargs.get('name', 'Camera'),
            mesh_type='Empty',
            **{k: v for k, v in kwargs.items() if k not in ['name', 'position']}
        )
        
        # Remove Mesh component for cameras
        obj.remove_component("Mesh")
        
        # Could add Camera component here if implemented
        # obj.add_component("Camera", fov=60.0)
        
        # Set position if provided
        if position:
            obj["Transform"].position = torch.tensor(position, dtype=torch.float32)
        
        return obj