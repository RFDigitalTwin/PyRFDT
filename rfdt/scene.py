"""
Scene containing all objects and their relationships.
"""
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .scene_object import SceneObject
from .parameters import ParameterManager
from .ui.node_editor_manager import NodeEditorManager


class Scene:
    """Scene containing all objects and their relationships."""
    
    def __init__(self):
        """Initialize empty scene."""
        self.objects: Dict[str, SceneObject] = {}
        self._transform_to_object: Dict[Any, str] = {}  # Map transforms to object IDs
        self.transmitters: List[str] = []  # List of transmitter object IDs
        self.receivers: List[str] = []  # List of receiver object IDs

        # Initialize managers
        self.parameter_manager = ParameterManager(self)
        self.node_editor_manager = NodeEditorManager(self)

        self._callbacks: Dict[str, List[callable]] = {
            'object_added': [],
            'object_removed': [],
            'object_updated': [],
            'parent_changed': [],
            'trace_requested': [],
            'node_editor_updated': [],
            'parameters_updated': [],  # New event for parameter updates
            'component_added': [],
            'component_removed': [],
            'component_updated': []
        }

        self.wt_scene = None

    # Delegate parameter access to parameter manager
    @property
    def parameters(self):
        """Get the parameters dictionary (with auto-sync on modification)."""
        return self.parameter_manager.parameters

    @parameters.setter
    def parameters(self, value):
        """Set the parameters dictionary."""
        self.parameter_manager.parameters = value

    @property
    def parameters_info(self):
        """Get parameter definitions."""
        return self.parameter_manager.parameters_info

    @parameters_info.setter
    def parameters_info(self, value):
        """Set parameter definitions."""
        self.parameter_manager.parameters_info = value

    # Delegate node editor access to node editor manager
    @property
    def node_editor(self):
        """Get node editor state."""
        return self.node_editor_manager.node_editor

    @node_editor.setter
    def node_editor(self, value):
        """Set node editor state."""
        self.node_editor_manager.node_editor = value
    
    def on(self, event: str, callback: callable) -> None:
        """Register a callback for a scene event."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def off(self, event: str, callback: callable) -> None:
        """Unregister a callback for a scene event."""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
    
    def _emit(self, event: str, **kwargs) -> None:
        """Emit an event to all registered callbacks."""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(self, **kwargs)
                except Exception as e:
                    print(f"Error in callback for {event}: {e}")
    
    def add_object(self, obj: SceneObject) -> None:
        """Add object to scene."""
        self.objects[obj.id] = obj
        # Set back-reference to scene
        obj.scene = self
        # Store transform component reference if it exists
        if "Transform" in obj._components:
            self._transform_to_object[obj["Transform"]] = obj.id

        # Check if object is a transmitter or receiver based on mesh type or name
        if 'transmitter' in obj.name.lower() or obj.mesh_type == 'Transmitter':
            if obj.id not in self.transmitters:
                self.transmitters.append(obj.id)
        elif 'receiver' in obj.name.lower() or obj.mesh_type == 'Receiver':
            if obj.id not in self.receivers:
                self.receivers.append(obj.id)

        self._emit('object_added', object=obj)
    
    def remove_object(self, object_id: str) -> None:
        """Remove object and all its children from scene."""
        if object_id not in self.objects:
            return
        
        obj = self.objects[object_id]
        
        # Remove from transmitters/receivers lists if present
        if object_id in self.transmitters:
            self.transmitters.remove(object_id)
        if object_id in self.receivers:
            self.receivers.remove(object_id)
        
        # Collect all objects to be removed (for event emission)
        removed_objects = []
        
        # Remove all children recursively
        if "Transform" not in obj._components:
            del self.objects[object_id]
            self._emit('object_removed', object_id=object_id, object=obj)
            return
        
        transform = obj["Transform"]
        children_to_remove = list(transform.children)
        for child_transform in children_to_remove:
            child_id = self._transform_to_object.get(child_transform)
            if child_id:
                self.remove_object(child_id)
        
        # Remove from parent's children list
        if transform.parent:
            transform.set_parent(None)
        
        # Remove from scene
        del self._transform_to_object[transform]
        del self.objects[object_id]
        
        # Emit event
        self._emit('object_removed', object_id=object_id, object=obj)
    
    def find_object(self, object_id: str) -> Optional[SceneObject]:
        """Find object by ID."""
        return self.objects.get(object_id)
    
    def get_object(self, object_id: str) -> Optional[SceneObject]:
        """Get object by ID (alias for find_object)."""
        return self.find_object(object_id)
    
    def add_component(self, object_id: str, component_name: str, **kwargs) -> bool:
        """Add a component to an object and emit event."""
        obj = self.find_object(object_id)
        if not obj:
            return False
        
        success = obj.add_component(component_name, **kwargs)
        if success:
            component = obj.get_component(component_name)
            if component:
                print("component added:", component.to_dict())
                self._emit('component_added', 
                          object_id=object_id, 
                          component=component.to_dict())
        return success
    
    def remove_component(self, object_id: str, component_name: str) -> bool:
        """Remove a component from an object and emit event."""
        obj = self.find_object(object_id)
        if not obj:
            return False

        success = obj.remove_component(component_name)
        if success:
            self._emit('component_removed',
                      object_id=object_id,
                      component_name=component_name)
        return success
    
    def update_component(self, object_id: str, component_name: str, field_name: str, value: Any) -> bool:
        """Update a component field and emit event."""
        obj = self.find_object(object_id)
        if not obj:
            return False
        
        success = obj.set_component_value(component_name, field_name, value)
        if success:
            self._emit('component_updated',
                      object_id=object_id,
                      component_name=component_name,
                      field_name=field_name,
                      value=value)
        return success
    
    def update_transform(self, object_id: str, position: Optional[List[float]] = None, 
                        rotation: Optional[List[float]] = None, scale: Optional[List[float]] = None) -> bool:
        """Update object transform and emit event."""
        obj = self.find_object(object_id)
        if not obj:
            return False
        
        if position is not None:
            obj["Transform"].position = torch.tensor(position, dtype=torch.float32)

        if rotation is not None:
            if len(rotation) == 4:
                # Quaternion format [x, y, z, w] - convert to Euler
                quat_tensor = torch.tensor(rotation, dtype=torch.float32)
                normalized_quat = quat_tensor / torch.norm(quat_tensor)
                # Use the set_quaternion method which handles conversion
                obj["Transform"].set_quaternion(normalized_quat.tolist())
                print("rotation set from quaternion, euler:", obj["Transform"].rotation.tolist())
            elif len(rotation) == 3:
                # Euler format [x, y, z] - keep as is (Transform expects Euler)
                obj["Transform"].rotation = torch.tensor(rotation, dtype=torch.float32)
                print("rotation euler (set directly):", obj["Transform"].rotation.tolist())
        if scale is not None:
            obj["Transform"].scale = torch.tensor(scale, dtype=torch.float32)
        print("rotation pre:", rotation, "post:", obj["Transform"].rotation.tolist())
        self._emit('object_updated', object_id=object_id, object=obj, update_type='transform')
        return True
    
    def update_material(self, object_id: str, color: Optional[str] = None, 
                       metalness: Optional[float] = None, roughness: Optional[float] = None,
                       texture: Optional[str] = None, permittivity: Optional[float] = None) -> bool:
        """Update object material and emit event."""
        obj = self.find_object(object_id)
        if not obj:
            return False
        
        if "Material" not in obj._components:
            obj.add_component("Material")
        
        material = obj["Material"]
        if color is not None:
            material.color = color
        if metalness is not None:
            material.metalness = metalness
        if roughness is not None:
            material.roughness = roughness
        if texture is not None:
            material.texture = texture
        if permittivity is not None:
            material.permittivity = permittivity
        
        self._emit('object_updated', object_id=object_id, object=obj, update_type='material')
        return True
    
    def update_visibility(self, object_id: str, visible: bool) -> bool:
        """Update object visibility and emit event."""
        obj = self.find_object(object_id)
        if not obj:
            return False
        
        obj.visible = visible
        self._emit('object_updated', object_id=object_id, object=obj, update_type='visibility')
        return True
    
    def request_trace(self, requester=None):
        """Request trace data generation."""
        self._emit('trace_requested', requester=requester)
    
    def add_transmitter(self, position: List[float] = None, rotation: List[float] = None,
                       scale: List[float] = None, name: str = None) -> str:
        """
        Create and add a new transmitter object to the scene.
        
        Args:
            position: [x, y, z] position vector
            rotation: [x, y, z] rotation in radians (euler angles)
            scale: [x, y, z] scale vector
            name: Optional custom name for the transmitter
            
        Returns:
            str: ID of the created transmitter object
        """
        import uuid
        obj_id = f"transmitter-{uuid.uuid4().hex[:8]}"
        
        # Create transmitter object
        transmitter = SceneObject(
            id=obj_id,
            name=name or f"Transmitter_{len(self.transmitters) + 1}",
            mesh_type="Transmitter",
            transform=Transform(
                position=position or [0, 0, 0],
                rotation=rotation or [0, 0, 0],
                scale=scale or [0.3, 0.3, 0.3]  # Default smaller scale for transmitters
            ),
            material=Material(
                color="#0000ff",  # Blue color for transmitters
                metalness=0.8,
                roughness=0.2
            ),
            visible=True
        )
        
        # Generate sphere mesh for transmitter
        transmitter.mesh = Mesh.create_sphere(radius=0.5)
        
        # Add to scene
        self.add_object(transmitter)
        
        # Ensure it's in the transmitters list
        if obj_id not in self.transmitters:
            self.transmitters.append(obj_id)
            
        return obj_id
    
    def add_receiver(self, position: List[float] = None, rotation: List[float] = None,
                    scale: List[float] = None, name: str = None) -> str:
        """
        Create and add a new receiver object to the scene.
        
        Args:
            position: [x, y, z] position vector
            rotation: [x, y, z] rotation in radians (euler angles)
            scale: [x, y, z] scale vector
            name: Optional custom name for the receiver
            
        Returns:
            str: ID of the created receiver object
        """
        import uuid
        obj_id = f"receiver-{uuid.uuid4().hex[:8]}"
        
        # Create receiver object
        receiver = SceneObject(
            id=obj_id,
            name=name or f"Receiver_{len(self.receivers) + 1}",
            mesh_type="Receiver",
            transform=Transform(
                position=position or [0, 0, 0],
                rotation=rotation or [0, 0, 0],
                scale=scale or [0.3, 0.3, 0.3]  # Default smaller scale for receivers
            ),
            material=Material(
                color="#ffff00",  # Yellow color for receivers
                metalness=0.8,
                roughness=0.2
            ),
            visible=True
        )
        
        # Generate sphere mesh for receiver
        receiver.mesh = Mesh.create_sphere(radius=0.5)
        
        # Add to scene
        self.add_object(receiver)
        
        # Ensure it's in the receivers list
        if obj_id not in self.receivers:
            self.receivers.append(obj_id)
            
        return obj_id
    
    def remove_transmitter(self, object_id: str) -> bool:
        """Remove an object from the transmitters list."""
        if object_id in self.transmitters:
            self.transmitters.remove(object_id)
            return True
        return False
    
    def remove_receiver(self, object_id: str) -> bool:
        """Remove an object from the receivers list."""
        if object_id in self.receivers:
            self.receivers.remove(object_id)
            return True
        return False
    
    def get_transmitter_positions(self) -> np.ndarray:
        """
        Get positions of all transmitters as a numpy array.
        
        Returns:
            np.ndarray: Nx3 array of transmitter positions in world space
        """
        positions = []
        for tx_id in self.transmitters:
            obj = self.find_object(tx_id)
            if obj:
                # Get world position (accounting for parent transforms)
                world_pos, _, _ = self._get_world_transform(obj)
                positions.append(world_pos)
        
        if positions:
            return np.array(positions, dtype=np.float32)
        else:
            return np.empty((0, 3), dtype=np.float32)
    
    def get_receiver_positions(self) -> np.ndarray:
        """
        Get positions of all receivers as a numpy array.
        
        Returns:
            np.ndarray: Nx3 array of receiver positions in world space
        """
        positions = []
        for rx_id in self.receivers:
            obj = self.find_object(rx_id)
            if obj:
                # Get world position (accounting for parent transforms)
                world_pos, _, _ = self._get_world_transform(obj)
                positions.append(world_pos)
        
        if positions:
            return np.array(positions, dtype=np.float32)
        else:
            return np.empty((0, 3), dtype=np.float32)
    
    def get_transmitter_objects(self) -> List[SceneObject]:
        """Get list of transmitter objects."""
        return [self.objects[tx_id] for tx_id in self.transmitters if tx_id in self.objects]
    
    def get_receiver_objects(self) -> List[SceneObject]:
        """Get list of receiver objects."""
        return [self.objects[rx_id] for rx_id in self.receivers if rx_id in self.objects]
    
    def set_parent(self, child_id: str, parent_id: Optional[str]) -> bool:
        """Set parent-child relationship between objects."""
        child = self.find_object(child_id)
        if not child:
            return False
        
        if "Transform" not in child._components:
            return False
        
        child_transform = child["Transform"]
        old_parent_id = None
        if child_transform.parent:
            # Find old parent ID
            for obj_id, obj in self.objects.items():
                if "Transform" in obj._components and obj["Transform"] == child_transform.parent:
                    old_parent_id = obj_id
                    break
        
        # Get world transform before changing parent
        world_pos, world_rot, world_scale = self._get_world_transform(child)
        
        if parent_id:
            parent = self.find_object(parent_id)
            if not parent:
                return False
            
            # Check for circular reference
            if self._would_create_circular_reference(child_id, parent_id):
                return False
            
            if "Transform" not in parent._components:
                return False
            child_transform.set_parent(parent["Transform"])
        else:
            child_transform.set_parent(None)
        
        # Convert world transform back to local in new parent space
        self._set_world_transform(child, world_pos, world_rot, world_scale)
        
        # Emit event
        self._emit('parent_changed', child_id=child_id, old_parent_id=old_parent_id, new_parent_id=parent_id)
        
        return True
    
    def _would_create_circular_reference(self, child_id: str, potential_parent_id: str) -> bool:
        """Check if setting parent would create circular reference."""
        current_id = potential_parent_id
        
        while current_id:
            if current_id == child_id:
                return True
            
            current_obj = self.find_object(current_id)
            if not current_obj or "Transform" not in current_obj._components:
                break
            
            current_transform = current_obj["Transform"]
            if not current_transform.parent:
                break
            
            # Find parent object ID
            parent_transform = current_transform.parent
            current_id = self._transform_to_object.get(parent_transform)
        
        return False
    
    def _get_world_transform(self, obj: SceneObject) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get world transform of an object."""
        if "Transform" not in obj._components:
            return np.zeros(3), np.zeros(3), np.ones(3)
        
        transform = obj["Transform"]
        # Start with local transform
        position = transform.position.numpy()
        rotation = transform._quaternion_to_euler()  # Convert to euler
        scale = transform.scale.numpy()
        
        # If no parent, local is world
        if not transform.parent:
            return position, np.array(rotation), scale
        
        # Get parent object
        parent_obj = None
        for scene_obj in self.objects.values():
            if "Transform" in scene_obj._components and scene_obj["Transform"] == transform.parent:
                parent_obj = scene_obj
                break
        
        if not parent_obj:
            return position, np.array(rotation), scale
        
        # Get parent's world transform
        parent_pos, parent_rot, parent_scale = self._get_world_transform(parent_obj)
        
        # Convert rotations to rotation matrices
        r_parent = Rotation.from_euler('xyz', parent_rot, degrees=False)
        r_local = Rotation.from_euler('xyz', rotation, degrees=False)
        
        # Combine transforms
        world_scale = parent_scale * scale
        r_world = r_parent * r_local
        world_rotation = r_world.as_euler('xyz', degrees=False)
        scaled_local_pos = parent_scale * position
        world_position = parent_pos + r_parent.apply(scaled_local_pos)
        
        return world_position, world_rotation, world_scale
    
    def _set_world_transform(self, obj: SceneObject, world_pos: np.ndarray, 
                            world_rot: np.ndarray, world_scale: np.ndarray) -> None:
        """Set object's local transform to achieve desired world transform."""
        if "Transform" not in obj._components:
            return
        
        transform = obj["Transform"]
        if not transform.parent:
            # No parent, world is local
            transform.position = torch.tensor(world_pos, dtype=torch.float32)
            transform.rotation = torch.tensor(world_rot, dtype=torch.float32)  # Already in Euler angles
            transform.scale = torch.tensor(world_scale, dtype=torch.float32)
            return
        
        # Get parent object
        parent_obj = None
        for scene_obj in self.objects.values():
            if "Transform" in scene_obj._components and scene_obj["Transform"] == transform.parent:
                parent_obj = scene_obj
                break
        
        if not parent_obj:
            return
        
        # Get parent's world transform
        parent_pos, parent_rot, parent_scale = self._get_world_transform(parent_obj)
        
        # Convert world to local
        r_parent = Rotation.from_euler('xyz', parent_rot, degrees=False)
        r_world = Rotation.from_euler('xyz', world_rot, degrees=False)
        
        # Local scale = world scale / parent scale
        # Avoid division by zero
        parent_scale_safe = np.where(parent_scale != 0, parent_scale, 1.0)
        local_scale = world_scale / parent_scale_safe
        
        # Local rotation = inverse(parent rotation) * world rotation
        r_local = r_parent.inv() * r_world
        local_rotation = r_local.as_euler('xyz', degrees=False)
        
        # Local position = inverse(parent rotation) * (world position - parent position) / parent scale
        world_offset = world_pos - parent_pos
        local_position = r_parent.inv().apply(world_offset) / parent_scale_safe
        
        # Update transform
        transform.position = torch.tensor(local_position, dtype=torch.float32)
        transform.rotation = torch.tensor(local_rotation, dtype=torch.float32)  # Already in Euler angles
        transform.scale = torch.tensor(local_scale, dtype=torch.float32)

    # Delegate parameter methods to parameter manager
    def update_parameters_info(self, parameters_info: List[Dict[str, Any]]) -> None:
        """Update parameter definitions and instantiate tensors."""
        self.parameter_manager.update_parameters_info(parameters_info)

    def get_parameter_value(self, param_id: str) -> Optional[torch.Tensor]:
        """Get parameter tensor by ID or name."""
        return self.parameter_manager.get_parameter_value(param_id)

    def set_parameter_value(self, param_id: str, value: Any) -> bool:
        """Set parameter value and update tensor."""
        return self.parameter_manager.set_parameter_value(param_id, value)

    def sync_parameter_values_to_frontend(self):
        """Sync current parameter tensor values back to parameters_info and emit update."""
        self.parameter_manager.sync_parameter_values()

    # Delegate node editor methods to node editor manager
    def update_node_editor(self, nodes: List[Any] = None, edges: List[Any] = None,
                          parameters: List[Any] = None, differentiableParams: List[str] = None) -> None:
        """Update node editor state and emit event."""
        self.node_editor_manager.update(nodes, edges, parameters, differentiableParams)

    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scene to dictionary for serialization."""
        objects_dict = {}
        
        for obj_id, obj in self.objects.items():
            obj_data = obj.to_dict()
            
            # Add parent_id based on transform relationships
            if "Transform" in obj._components:
                transform = obj["Transform"]
                if transform.parent:
                    parent_id = self._transform_to_object.get(transform.parent)
                    obj_data["parent_id"] = parent_id
                else:
                    obj_data["parent_id"] = None
            else:
                obj_data["parent_id"] = None
            
            objects_dict[obj_id] = obj_data
        
        return {
            "objects": objects_dict,
            "transmitters": self.transmitters,
            "receivers": self.receivers,
            "node_editor": self.node_editor_manager.get_state(),
            "parameters_info": self.parameter_manager.parameters_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Scene':
        """Create scene from dictionary."""
        scene = cls()
        
        # First pass: create all objects without parent relationships
        for obj_id, obj_data in data.get("objects", {}).items():
            # Extract components
            transform_data = obj_data.get("transform", {})
            material_data = obj_data.get("material")
            
            # Create transform
            transform = Transform(
                position=transform_data.get("position"),
                rotation=transform_data.get("rotation"),
                scale=transform_data.get("scale")
            )
            
            # Create material
            material = None
            if material_data:
                material = Material(
                    color=material_data.get("color", "#808080"),
                    texture=material_data.get("texture"),
                    metalness=material_data.get("metalness", 0.5),
                    roughness=material_data.get("roughness", 0.5),
                    permittivity=material_data.get("permittivity", 1.0)
                )
            
            # Create object
            obj = SceneObject(
                id=obj_id,
                name=obj_data.get("name", "Object"),
                mesh_type=obj_data.get("mesh", "Custom"),
                transform=transform,
                material=material,
                visible=obj_data.get("visible", True),
                model_path=obj_data.get("model_path")
            )
            
            scene.add_object(obj)
        
        # Second pass: establish parent-child relationships
        for obj_id, obj_data in data.get("objects", {}).items():
            parent_id = obj_data.get("parent_id")
            if parent_id:
                scene.set_parent(obj_id, parent_id)
        
        # Load transmitter and receiver lists if they exist
        scene.transmitters = data.get("transmitters", [])
        scene.receivers = data.get("receivers", [])
        
        # If lists don't exist, auto-detect based on object names/types (backward compatibility)
        if not scene.transmitters and not scene.receivers:
            for obj_id, obj in scene.objects.items():
                if 'transmitter' in obj.name.lower() or obj.mesh_type == 'Transmitter':
                    if obj_id not in scene.transmitters:
                        scene.transmitters.append(obj_id)
                elif 'receiver' in obj.name.lower() or obj.mesh_type == 'Receiver':
                    if obj_id not in scene.receivers:
                        scene.receivers.append(obj_id)
        
        # Load node editor state if it exists
        if "node_editor" in data:
            node_editor_data = data["node_editor"]
            scene.node_editor_manager.set_state(node_editor_data)
            # Migrate old parameters to new structure if needed
            if 'parameters' in node_editor_data:
                scene.parameter_manager.update_parameters_info(node_editor_data['parameters'])

        # Load parameters if they exist in the new format
        if "parameters_info" in data:
            scene.parameter_manager.update_parameters_info(data["parameters_info"])

        if "differentiableParams" in data.get("node_editor", {}):
            scene.parameter_manager.update_parameters_grad(data["node_editor"]["differentiableParams"])

        return scene