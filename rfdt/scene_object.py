"""
Enhanced SceneObject with dictionary-style component access.
"""
from typing import Any, Dict, List, Optional, Union
import torch
import uuid
from .components import Component, ComponentRegistry
from .components import TransformComponent, MaterialComponent, MeshComponent


class ComponentAccessor:
    """Helper class for dictionary-style component access."""
    
    def __init__(self, scene_object: 'SceneObject'):
        self._scene_object = scene_object
    
    def __getitem__(self, component_name: str) -> Optional[Component]:
        """Get component by name using dictionary syntax."""
        return self._scene_object.get_component(component_name)
    
    def __setitem__(self, component_name: str, component: Component):
        """Set component using dictionary syntax."""
        if isinstance(component, Component):
            self._scene_object._components[component_name] = component
            component._owner = self._scene_object
        else:
            raise TypeError(f"Expected Component instance, got {type(component)}")
    
    def __delitem__(self, component_name: str):
        """Remove component using dictionary syntax."""
        self._scene_object.remove_component(component_name)
    
    def __contains__(self, component_name: str) -> bool:
        """Check if component exists."""
        return component_name in self._scene_object._components


class SceneObject:
    """Enhanced scene object with dictionary-style component access."""
    
    def __init__(self,
                 id: Optional[str] = None,
                 name: str = "Object",
                 parent_id: Optional[str] = None,
                 mesh_type: str = "Custom",
                 mesh_path: Optional[str] = None,
                 visible: bool = True,
                 tag: Optional[str] = None):
        """Initialize scene object with components."""

        # Basic properties
        self.id = id or str(uuid.uuid4())
        self._name = name
        self._tag = tag or ""  # Initialize tag as empty string if not provided
        self.parent_id = parent_id
        self.visible = visible
        self.mesh_path = mesh_path
        self.mesh_type = mesh_type
        self.scene = None  # Will be set when added to a scene

        # Component storage
        self._components: Dict[str, Component] = {}
        
        # Component accessor for dictionary-style access
        self.components = ComponentAccessor(self)
        
        # Initialize default components
        self._init_default_components()
    
    def _init_default_components(self):
        """Initialize default components."""
        # Add Transform component (required for all objects)
        transform_comp = TransformComponent()
        transform_comp._owner = self
        self._components["Transform"] = transform_comp

        # For Radar type, only add Radar component
        if self.mesh_type == "Radar":
            from .components.radar import RadarComponent
            radar_comp = RadarComponent()
            radar_comp._owner = self
            self._components["Radar"] = radar_comp
            return

        # For Transmitter type, only add Transmitter component
        if self.mesh_type == "Transmitter":
            from .components.transmitter import TransmitterComponent
            transmitter_comp = TransmitterComponent()
            transmitter_comp._owner = self
            self._components["Transmitter"] = transmitter_comp
            return

        # For Receiver type, only add Receiver component
        if self.mesh_type == "Receiver":
            from .components.receiver import ReceiverComponent
            receiver_comp = ReceiverComponent()
            receiver_comp._owner = self
            self._components["Receiver"] = receiver_comp
            return

        # Add Material component for other types
        material_comp = MaterialComponent()
        material_comp._owner = self
        self._components["Material"] = material_comp

        # Add Mesh component
        mesh_comp = MeshComponent(
            mesh_type=self.mesh_type,
            visible=self.visible
        )
        mesh_comp._owner = self
        self._components["Mesh"] = mesh_comp

    @property
    def name(self) -> str:
        """Get object name."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Set object name and notify scene."""
        if self._name != value:
            self._name = value
            if self.scene:
                self.scene._emit('object_updated', object_id=self.id, object=self, update_type='name')

    @property
    def tag(self) -> str:
        """Get object tag."""
        return self._tag

    @tag.setter
    def tag(self, value: str):
        """Set object tag and notify scene."""
        if self._tag != value:
            self._tag = value
            if self.scene:
                self.scene._emit('object_updated', object_id=self.id, object=self, update_type='tag')
    
    def __getitem__(self, component_name: str) -> Optional[Component]:
        """Dictionary-style component access."""
        return self._components.get(component_name)
    
    def __setitem__(self, component_name: str, value: Any):
        """Dictionary-style component setting."""
        if isinstance(value, Component):
            self._components[component_name] = value
            value._owner = self
        else:
            # Try to create component if it's a dict
            if isinstance(value, dict):
                comp = ComponentRegistry.create_instance(component_name, **value)
                if comp:
                    comp._owner = self
                    self._components[component_name] = comp
    
    def add_component(self, component_name: str, **kwargs) -> bool:
        """Add a component to the object."""
        if component_name in self._components:
            return False  # Already exists
        
        component = ComponentRegistry.create_instance(component_name, **kwargs)
        if component:
            component._owner = self
            self._components[component_name] = component
            return True
        return False
    
    def get_component(self, component_name: str) -> Optional[Component]:
        """Get a component by name."""
        return self._components.get(component_name)
    
    def remove_component(self, component_name: str) -> bool:
        """Remove a component from the object."""
        if component_name in ["Transform"]:  # Don't allow removing essential components
            return False
        
        if component_name in self._components:
            del self._components[component_name]
            return True
        return False
    
    def has_component(self, component_name: str) -> bool:
        """Check if object has a component."""
        return component_name in self._components
    
    def set_component_value(self, component_name: str, field_name: str, value: Any) -> bool:
        """Set a component field value."""
        component = self._components.get(component_name)
        if component:
            try:
                setattr(component, field_name, value)
                return True
            except Exception as e:
                print(f"Error setting component value: {e}")
                return False
        return False
    
    def get_all_components(self) -> Dict[str, Component]:
        """Get all components."""
        return self._components.copy()

    def update(self, component_name: str = None):
        """Manually trigger synchronization to frontend.

        Args:
            component_name: Optional specific component to update. If None, updates all components.
        """
        if self.scene:
            if component_name:
                # Update specific component
                component = self._components.get(component_name)
                if component:
                    component.update()
            else:
                # Update all components
                for comp in self._components.values():
                    comp.update()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Get components as list
        components_list = []
        for comp_name, comp in self._components.items():
            components_list.append(comp.to_dict())
        
        # Get transform data from component
        transform_comp = self._components.get("Transform")
        transform_data = {}
        if transform_comp:
            # Get quaternion for serialization (matching frontend expectations)
            quaternion = transform_comp.get_quaternion()
            transform_data = {
                "position": transform_comp.position.tolist() if isinstance(transform_comp.position, torch.Tensor) else transform_comp.position,
                "rotation": quaternion.tolist() if isinstance(quaternion, torch.Tensor) else quaternion,
                "scale": transform_comp.scale.tolist() if isinstance(transform_comp.scale, torch.Tensor) else transform_comp.scale
            }
        
        # Get material data from component
        material_comp = self._components.get("Material")
        material_data = {}
        if material_comp:
            material_data = material_comp.to_legacy_dict()  # Use legacy format for frontend compatibility
        
        return {
            "id": self.id,
            "name": self._name,
            "tag": self._tag,
            "parent_id": self.parent_id,
            "transform": transform_data,
            "material": material_data,
            "mesh": self.mesh_type,
            "mesh_path": self.mesh_path,
            "visible": self.visible,
            "components": components_list
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Load from dictionary."""
        self.id = data.get("id", self.id)
        self._name = data.get("name", self._name)
        self._tag = data.get("tag", "")
        self.parent_id = data.get("parent_id")
        self.mesh_type = data.get("mesh", "Custom")
        self.mesh_path = data.get("mesh_path")
        self.visible = data.get("visible", True)
        
        # Load transform data into component
        if "transform" in data and "Transform" in self._components:
            transform_data = data["transform"]
            transform_comp = self._components["Transform"]
            
            if "position" in transform_data:
                transform_comp.position = torch.tensor(transform_data["position"], dtype=torch.float32)
            
            if "rotation" in transform_data:
                rotation = transform_data["rotation"]
                # If it's a quaternion (4 values), convert to euler
                if len(rotation) == 4:
                    transform_comp.set_quaternion(rotation)
                else:
                    transform_comp.rotation = torch.tensor(rotation[:3], dtype=torch.float32)
            
            if "scale" in transform_data:
                transform_comp.scale = torch.tensor(transform_data["scale"], dtype=torch.float32)
        
        # Load material data into component
        if "material" in data and "Material" in self._components:
            material_data = data["material"]
            material_comp = self._components["Material"]
            
            if "color" in material_data:
                material_comp.color = material_data["color"]
            if "metalness" in material_data:
                material_comp.metalness = material_data["metalness"]
            if "roughness" in material_data:
                material_comp.roughness = material_data["roughness"]
            if "permittivity" in material_data:
                material_comp.permittivity = material_data["permittivity"]
            if "texture" in material_data:
                material_comp.texture = material_data["texture"] if material_data["texture"] else ""
        
        # Update mesh component
        if "Mesh" in self._components:
            self._components["Mesh"].mesh_type = self.mesh_type
            self._components["Mesh"].visible = self.visible
            if self.mesh_type in ['Cube', 'Sphere', 'Cylinder']:
                self._components["Mesh"]._generate_primitive()
        
        # Load additional components
        if "components" in data:
            for comp_data in data["components"]:
                comp_name = comp_data.get("name")
                if comp_name and comp_name not in self._components:
                    comp = ComponentRegistry.create_instance(comp_name)
                    if comp:
                        comp.from_dict(comp_data)
                        comp._owner = self
                        self._components[comp_name] = comp
    
