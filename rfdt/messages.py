"""
Message handling and serialization for WebSocket communication.
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import json
import logging
import torch
import numpy as np
from scipy.spatial.transform import Rotation

from .scene import Scene
from .scene_object import SceneObject

logger = logging.getLogger(__name__)


@dataclass
class WebSocketMessage:
    """Base class for WebSocket messages."""
    type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict())


class MessageHandler:
    """Handles WebSocket messages and scene updates."""
    
    def __init__(self, scene: Scene):
        """Initialize with a scene instance."""
        self.scene = scene
    
    async def handle_message(self, message: Dict[str, Any], websocket) -> Optional[WebSocketMessage]:
        """Route message to appropriate handler."""
        message_type = message.get('type')
        
        if message_type == 'create_object':
            return await self.handle_create_object(message)
        elif message_type == 'delete_object':
            return await self.handle_delete_object(message)
        elif message_type == 'update_transform':
            return await self.handle_update_transform(message)
        elif message_type == 'update_material':
            return await self.handle_update_material(message)
        elif message_type == 'update_visibility':
            return await self.handle_update_visibility(message)
        elif message_type == 'update_parent':
            return await self.handle_update_parent(message)
        elif message_type == 'duplicate_object':
            return await self.handle_duplicate_object(message)
        elif message_type == 'get_trace':
            return await self.handle_get_trace(message)
        elif message_type == 'update_node_editor':
            return await self.handle_update_node_editor(message)
        elif message_type == 'get_node_editor':
            return await self.handle_get_node_editor(message)
        else:
            logger.warning(f"Unknown message type: {message_type}")
            return None
    
    async def handle_create_object(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle object creation."""
        obj_data = data.get('object', {})
        
        # Extract data
        obj_id = obj_data.get('id')
        name = obj_data.get('name', 'Object')
        mesh_type = obj_data.get('mesh', 'Custom')
        transform_data = obj_data.get('transform', {})
        material_data = obj_data.get('material')
        visible = obj_data.get('visible', True)
        model_path = obj_data.get('model_path')
        parent_id = obj_data.get('parent_id')
        
        # Create transform
        transform = Transform(
            position=transform_data.get('position'),
            rotation=transform_data.get('rotation'),
            scale=transform_data.get('scale')
        )
        
        # Create material
        material = None
        if material_data:
            material = Material(
                color=material_data.get('color', '#808080'),
                texture=material_data.get('texture'),
                metalness=material_data.get('metalness', 0.5),
                roughness=material_data.get('roughness', 0.5),
                permittivity=material_data.get('permittivity', 1.0)
            )
        
        # Create object
        obj = SceneObject(
            id=obj_id,
            name=name,
            mesh_type=mesh_type,
            transform=transform,
            material=material,
            visible=visible,
            model_path=model_path
        )
        
        # Add to scene
        self.scene.add_object(obj)
        
        # Set parent if specified
        if parent_id:
            self.scene.set_parent(obj_id, parent_id)
        
        logger.info(f"Object created: {obj_id} ({name})")
        
        return {
            "type": "object_created",
            "object": self._serialize_object(obj)
        }
    
    async def handle_delete_object(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle object deletion."""
        object_id = data.get('object_id')
        
        if not object_id or object_id not in self.scene.objects:
            return []
        
        # Get all objects that will be deleted (including children)
        to_delete = [object_id]
        obj = self.scene.objects[object_id]
        
        # Recursively collect all children
        def collect_children(transform: Transform):
            for child in transform.children:
                for obj_id, obj in self.scene.objects.items():
                    if obj.transform == child:
                        to_delete.append(obj_id)
                        collect_children(child)
                        break
        
        collect_children(obj.transform)
        
        # Remove object (this will recursively remove children)
        self.scene.remove_object(object_id)
        
        logger.info(f"Objects deleted: {to_delete}")
        
        # Return deletion messages for all deleted objects
        return [{"type": "object_deleted", "object_id": obj_id} for obj_id in to_delete]
    
    async def handle_update_transform(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle transform update."""
        object_id = data.get('object_id')
        transform_data = data.get('transform')
        
        if not object_id or not transform_data:
            return None
        
        # Use scene's update method
        success = self.scene.update_transform(
            object_id,
            position=transform_data.get('position'),
            rotation=transform_data.get('rotation'),
            scale=transform_data.get('scale')
        )
        
        if not success:
            return None
        
        
        obj = self.scene.find_object(object_id)
        logger.info(f"Transform updated for {object_id}")
        
        return {
            "type": "transform_updated",
            "object_id": object_id,
            "transform": obj.transform.to_dict()
        }
    
    async def handle_update_material(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle material update."""
        object_id = data.get('object_id')
        material_data = data.get('material')
        
        if not object_id or not material_data:
            return None
        
        # Use scene's update method
        success = self.scene.update_material(
            object_id,
            color=material_data.get('color'),
            metalness=material_data.get('metalness'),
            roughness=material_data.get('roughness'),
            texture=material_data.get('texture'),
            permittivity=material_data.get('permittivity')
        )
        
        if not success:
            return None
        
        obj = self.scene.find_object(object_id)
        logger.info(f"Material updated for {object_id}")
        
        return {
            "type": "material_updated",
            "object_id": object_id,
            "material": obj.material.to_dict()
        }
    
    async def handle_update_visibility(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle visibility update."""
        object_id = data.get('object_id')
        visible = data.get('visible')
        
        if not object_id or visible is None:
            return None
        
        # Use scene's update method
        success = self.scene.update_visibility(object_id, visible)
        
        if not success:
            return None
        
        logger.info(f"Visibility updated for {object_id}: {visible}")
        
        return {
            "type": "visibility_updated",
            "object_id": object_id,
            "visible": visible
        }
    
    async def handle_update_parent(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle parent update with transform preservation."""
        object_id = data.get('object_id')
        parent_id = data.get('parent_id')
        
        if not object_id:
            return None
        
        obj = self.scene.find_object(object_id)
        if not obj:
            return None
        
        # Update parent (Scene class now handles transform preservation)
        success = self.scene.set_parent(object_id, parent_id)
        if not success:
            return {
                "type": "error",
                "message": "Cannot create circular parent reference"
            }
        
        logger.info(f"Parent updated for {object_id}: {parent_id}")
        
        return {
            "type": "object_updated",
            "object": self._serialize_object(obj)
        }
    
    async def handle_duplicate_object(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle object duplication."""
        object_id = data.get('object_id')
        
        if not object_id:
            return []
        
        original = self.scene.find_object(object_id)
        if not original:
            return []
        
        import uuid
        
        # Create new object
        new_id = f"{original.mesh_type.lower()}-{uuid.uuid4().hex[:8]}"

        print(original.transform.position.tolist())
        
        # Clone transform with exact same values
        new_transform = Transform(
            position=original.transform.position.tolist(),
            rotation=original.transform.rotation.tolist(),
            scale=original.transform.scale.tolist()
        )

        print(new_transform.position.tolist())
        
        # Clone material
        new_material = None
        if original.material:
            new_material = Material(
                color=original.material.color,
                texture=original.material.texture,
                metalness=original.material.metalness,
                roughness=original.material.roughness,
                permittivity=original.material.permittivity
            )
        
        # Create new object
        new_obj = SceneObject(
            id=new_id,
            name=f"{original.name} (Copy)",
            mesh_type=original.mesh_type,
            transform=new_transform,
            material=new_material,
            visible=original.visible,
            model_path=original.model_path
        )
        
        self.scene.add_object(new_obj)
        
        # Set parent if original had one
        orig_transform = original["Transform"] if hasattr(original, '__getitem__') else None
        if orig_transform and hasattr(orig_transform, 'parent') and orig_transform.parent:
            parent_id = None
            for pid, pobj in self.scene.objects.items():
                parent_transform = pobj["Transform"]
                if parent_transform == orig_transform.parent:
                    parent_id = pid
                    break
            if parent_id:
                self.scene.set_parent(new_id, parent_id)
        
        # Duplicate children recursively
        created_objects = [{"type": "object_created", "object": self._serialize_object(new_obj)}]
        
        # TODO: Implement recursive child duplication
        
        logger.info(f"Object duplicated: {object_id} -> {new_id}")
        
        return created_objects
    
    async def handle_get_trace(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trace data."""
        import random
        
        # Define color palette
        colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff"]
        
        # Generate 2-4 path groups
        num_groups = random.randint(2, 4)
        path_groups = []
        
        for i in range(num_groups):
            group_id = f"group{i+1}"
            color = colors[i % len(colors)]
            
            # Generate 2-5 paths per group
            num_paths = random.randint(2, 5)
            paths = []
            
            for j in range(num_paths):
                # Generate 3-10 points per path
                num_points = random.randint(3, 10)
                path = []
                
                # Random starting position
                x = random.uniform(-5, 5)
                y = random.uniform(0, 5)
                z = random.uniform(-5, 5)
                
                for k in range(num_points):
                    # Add some randomness to create curved paths
                    x += random.uniform(-1, 1)
                    y += random.uniform(-0.5, 0.5)
                    z += random.uniform(-1, 1)
                    
                    # Keep within bounds
                    x = max(-10, min(10, x))
                    y = max(0, min(10, y))
                    z = max(-10, min(10, z))
                    
                    path.append([round(x, 2), round(y, 2), round(z, 2)])
                
                paths.append(path)
            
            path_groups.append({
                "groupId": group_id,
                "color": color,
                "paths": paths
            })
        
        return {
            "type": "trace_data",
            "data": {"pathGroups": path_groups}
        }
    
    async def handle_update_node_editor(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle node editor update."""
        nodes = data.get('nodes')
        edges = data.get('edges')
        parameters = data.get('parameters')
        differentiableParams = data.get('differentiableParams')
        
        # Update scene's node editor state
        self.scene.update_node_editor(
            nodes=nodes,
            edges=edges,
            parameters=parameters,
            differentiableParams=differentiableParams
        )
        
        logger.info("Node editor updated")
        
        return {
            "type": "node_editor_updated",
            "node_editor": self.scene.node_editor
        }
    
    async def handle_get_node_editor(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get current node editor state."""
        return {
            "type": "node_editor_state",
            "node_editor": self.scene.node_editor
        }
    
    def _serialize_object(self, obj: SceneObject) -> Dict[str, Any]:
        """Serialize object for WebSocket transmission."""
        data = obj.to_dict()
        
        # Add parent_id
        transform_comp = obj["Transform"]
        if transform_comp and hasattr(transform_comp, 'parent') and transform_comp.parent:
            for obj_id, scene_obj in self.scene.objects.items():
                scene_transform = scene_obj["Transform"]
                if scene_transform == transform_comp.parent:
                    data["parent_id"] = obj_id
                    break
        
        return data
    
    def _get_world_transform(self, obj: SceneObject):
        """Get world transform of an object."""
        transform_comp = obj["Transform"]
        if not transform_comp:
            return np.zeros(3), np.zeros(3), np.ones(3)
        
        # Start with local transform
        position = transform_comp.position.numpy() if hasattr(transform_comp.position, 'numpy') else np.array(transform_comp.position)
        rotation = transform_comp.rotation.numpy() if hasattr(transform_comp.rotation, 'numpy') else np.array(transform_comp.rotation)
        scale = transform_comp.scale.numpy() if hasattr(transform_comp.scale, 'numpy') else np.array(transform_comp.scale)
        
        # If no parent, local is world
        if not hasattr(transform_comp, 'parent') or not transform_comp.parent:
            return position, rotation, scale
        
        # Get parent object
        parent_obj = None
        for scene_obj in self.scene.objects.values():
            scene_transform = scene_obj["Transform"]
            if scene_transform == transform_comp.parent:
                parent_obj = scene_obj
                break
        
        if not parent_obj:
            return position, rotation, scale
        
        # Get parent's world transform
        parent_pos, parent_rot, parent_scale = self._get_world_transform(parent_obj)
        
        # Convert rotations to rotation matrices
        r_parent = Rotation.from_euler('YXZ', parent_rot, degrees=False)
        r_local = Rotation.from_euler('YXZ', rotation, degrees=False)
        
        # Combine transforms
        world_scale = parent_scale * scale
        r_world = r_parent * r_local
        world_rotation = r_world.as_euler('YXZ', degrees=False)
        scaled_local_pos = parent_scale * position
        world_position = parent_pos + r_parent.apply(scaled_local_pos)
        
        return world_position, world_rotation, world_scale
    
    def _set_world_transform(self, obj: SceneObject, world_pos, world_rot, world_scale):
        """Set object's local transform to achieve desired world transform."""
        transform_comp = obj["Transform"]
        if not transform_comp:
            return
        
        if not hasattr(transform_comp, 'parent') or not transform_comp.parent:
            # No parent, world is local
            transform_comp.position = torch.tensor(world_pos, dtype=torch.float32)
            transform_comp.rotation = torch.tensor(world_rot, dtype=torch.float32)  # Already in Euler
            transform_comp.scale = torch.tensor(world_scale, dtype=torch.float32)
            return
        
        # Get parent object
        parent_obj = None
        for scene_obj in self.scene.objects.values():
            scene_transform = scene_obj["Transform"]
            if scene_transform == transform_comp.parent:
                parent_obj = scene_obj
                break
        
        if not parent_obj:
            return
        
        # Get parent's world transform
        parent_pos, parent_rot, parent_scale = self._get_world_transform(parent_obj)
        
        # Convert world to local
        r_parent = Rotation.from_euler('YXZ', parent_rot, degrees=False)
        r_world = Rotation.from_euler('YXZ', world_rot, degrees=False)
        
        # Local scale = world scale / parent scale
        local_scale = world_scale / parent_scale
        
        # Local rotation = inverse(parent rotation) * world rotation
        r_local = r_parent.inv() * r_world
        local_rotation = r_local.as_euler('YXZ', degrees=False)
        
        # Local position = inverse(parent rotation) * (world position - parent position) / parent scale
        world_offset = world_pos - parent_pos
        local_position = r_parent.inv().apply(world_offset) / parent_scale
        
        print("local_rotation:", local_rotation)

        # Update transform
        transform_comp.position = torch.tensor(local_position, dtype=torch.float32)
        transform_comp.rotation = torch.tensor(local_rotation, dtype=torch.float32)  # Already in Euler
        transform_comp.scale = torch.tensor(local_scale, dtype=torch.float32)


def serialize_scene(scene: Scene) -> Dict[str, Any]:
    """Serialize entire scene for transmission."""
    return scene.to_dict()


def deserialize_scene(data: Dict[str, Any]) -> Scene:
    """Deserialize scene from data."""
    return Scene.from_dict(data)