"""
Refactored FastAPI server with scene monitoring architecture.
"""
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Set, Optional, Dict, Any, List
from uuid import uuid4
import time
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pythonjsonlogger import jsonlogger
import numpy as np
import trimesh
import torch
import uvicorn

# Import our modular components
from .scene import Scene
from .scene_object import SceneObject
from .messages import MessageHandler, serialize_scene, deserialize_scene
from .editor import Editor
from .ui.result_visualizer import ResultVisualizer
from .ui.console import Console

# Configure logging
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)


class Server:
    """WebSocket server that monitors a Scene instance for changes."""
    
    def __init__(self, scene: Optional[Scene] = None, verbose: bool = False):
        """Initialize server with optional scene and verbose setting."""
        self.scene = scene
        self.verbose = verbose
        self.app = FastAPI()
        self.connected_clients: Set[WebSocket] = set()
        self.is_dirty = False
        self.last_save_time = time.time()
        self.save_interval = 30.0
        self.save_task = None
        
        # Set up directory structure
        self.scenes_dir = Path("scenes")
        self.default_scene_dir = self.scenes_dir / "default"
        self.models_dir = self.default_scene_dir / "models"
        
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Set file paths
        self.scene_file = self.default_scene_dir / "scene.json"
        
        # Configure logging based on verbose setting
        if not self.verbose:
            logger.setLevel(logging.ERROR)
        else:
            logger.setLevel(logging.INFO)
        
        # If no scene provided, load from file
        if self.scene is None:
            self.scene = self._load_default_scene()

        # Link server to scene for component access
        self.scene._server = self

        # Message handler
        self.message_handler = MessageHandler(self.scene)
        
        # Register scene callbacks
        self._register_scene_callbacks()

        # Set up routes and middleware
        self._setup_routes()

        # Trace callback (can be overridden by user)
        self.trace_callback = self._default_trace_callback

        # Result visualizer
        self.results = ResultVisualizer(server=self)

        # Console system
        self.console = Console(server=self)

        # Log available components on startup
        try:
            from .components import ComponentRegistry
            available_components = ComponentRegistry.get_all()
            component_names = [comp._component_name if hasattr(comp, '_component_name') else comp.__name__ for comp in available_components]
            print(f"[Server] Available components at startup: {component_names}")
        except Exception as e:
            print(f"[Server] Warning: Failed to load components at startup: {e}")
            logger.warning(f"Failed to load components at startup: {e}", exc_info=True)
    
    def _load_default_scene(self) -> Scene:
        """Load the default scene from JSON file."""
        # Check new location first
        if self.scene_file.exists():
            with open(self.scene_file, 'r') as f:
                data = json.load(f)
                return deserialize_scene(data)
        
        # Check old locations for backward compatibility
        for old_path in [Path("default_scene.json"), Path("scene_state.json")]:
            if old_path.exists():
                with open(old_path, 'r') as f:
                    data = json.load(f)
                    return deserialize_scene(data)
        
        # Return empty scene
        return Scene()
    
    def _register_scene_callbacks(self):
        """Register callbacks to monitor scene changes."""
        print("[Server] Registering scene callbacks...")
        self.scene.on('object_added', self._on_object_added)
        self.scene.on('object_removed', self._on_object_removed)
        self.scene.on('object_updated', self._on_object_updated)
        self.scene.on('parent_changed', self._on_parent_changed)
        self.scene.on('node_editor_updated', self._on_node_editor_updated)
        self.scene.on('parameters_updated', self._on_parameters_updated)
        self.scene.on('component_added', self._on_component_added)
        self.scene.on('component_removed', self._on_component_removed)
        self.scene.on('component_updated', self._on_component_updated)
        print(f"[Server] Callbacks registered. component_added callbacks: {len(self.scene._callbacks['component_added'])}")
    
    def _on_object_added(self, scene: Scene, object: SceneObject):
        """Handle object added event."""
        self.mark_dirty()
        obj_data = self._serialize_object(object)
        self._safe_broadcast({
            "type": "object_created",
            "object": obj_data
        })
    
    def _on_object_removed(self, scene: Scene, object_id: str, object: SceneObject):
        """Handle object removed event."""
        self.mark_dirty()
        self._safe_broadcast({
            "type": "object_deleted",
            "object_id": object_id
        })
    
    def _on_object_updated(self, scene: Scene, object_id: str, object: SceneObject, update_type: str):
        """Handle object updated event."""
        self.mark_dirty()
        
        if update_type == 'transform':
            print(f"Server: Broadcasting transform_updated for {object_id}")
            transform_comp = object["Transform"]
            if transform_comp:
                # Get the transform values in the format expected by frontend
                transform_data = transform_comp.to_dict()
                self._safe_broadcast({
                    "type": "transform_updated",
                    "object_id": object_id,
                    "transform": transform_data["values"]  # Send only the values, not the full component structure
                })
        elif update_type == 'material':
            material_comp = object["Material"]
            if material_comp:
                self._safe_broadcast({
                    "type": "material_updated",
                    "object_id": object_id,
                    "material": material_comp.to_dict()
                })
        elif update_type == 'visibility':
            self._safe_broadcast({
                "type": "visibility_updated",
                "object_id": object_id,
                "visible": object.visible
            })
        elif update_type == 'name':
            self._safe_broadcast({
                "type": "name_updated",
                "object_id": object_id,
                "name": object.name
            })
        elif update_type == 'tag':
            self._safe_broadcast({
                "type": "tag_updated",
                "object_id": object_id,
                "tag": object.tag
            })
        else:
            # Full object update
            obj_data = self._serialize_object(object)
            self._safe_broadcast({
                "type": "object_updated",
                "object": obj_data
            })
    
    def _on_parent_changed(self, scene: Scene, child_id: str, old_parent_id: Optional[str], new_parent_id: Optional[str]):
        """Handle parent changed event."""
        self.mark_dirty()
        obj = scene.find_object(child_id)
        if obj:
            obj_data = self._serialize_object(obj)
            self._safe_broadcast({
                "type": "object_updated",
                "object": obj_data
            })
    
    def _on_node_editor_updated(self, scene: Scene, node_editor: Dict[str, Any]):
        """Handle node editor update event."""
        self.mark_dirty()
        # Include parameters_info as 'parameters' for frontend compatibility
        node_editor_with_params = dict(node_editor)
        node_editor_with_params['parameters'] = scene.parameters_info
        self._safe_broadcast({
            "type": "node_editor_updated",
            "node_editor": node_editor_with_params
        })

    def _on_parameters_updated(self, scene: Scene, **kwargs):
        """Handle parameters update event."""
        self.mark_dirty()
        # Send the updated node editor state with parameters
        node_editor_with_params = dict(scene.node_editor)
        node_editor_with_params['parameters'] = scene.parameters_info
        self._safe_broadcast({
            "type": "node_editor_updated",
            "node_editor": node_editor_with_params
        })

    def _on_component_added(self, scene: Scene, **kwargs):
        """Handle component added event."""
        print(f"[Server] _on_component_added called with kwargs: {kwargs}")
        self.mark_dirty()
        self._safe_broadcast({
            "type": "component_added",
            "object_id": kwargs.get('object_id'),
            "component": kwargs.get('component')
        })
    
    def _on_component_removed(self, scene: Scene, **kwargs):
        """Handle component removed event."""
        self.mark_dirty()
        self._safe_broadcast({
            "type": "component_removed",
            "object_id": kwargs.get('object_id'),
            "component_name": kwargs.get('component_name')
        })
    
    def _on_component_updated(self, scene: Scene, **kwargs):
        """Handle component updated event."""
        if self.verbose:
            print(f"[Server] _on_component_updated called with kwargs: {kwargs}")
        self.mark_dirty()

        # Convert tensor values to regular Python types
        value = kwargs.get('value')
        if hasattr(value, 'tolist'):
            value = value.tolist()
        elif hasattr(value, 'item'):
            value = value.item()

        message = {
            "type": "component_updated",
            "object_id": kwargs.get('object_id'),
            "component_name": kwargs.get('component_name'),
            "field_name": kwargs.get('field_name'),
            "value": value
        }

        if self.verbose:
            print(f"[Server] Broadcasting component_updated: {message}")

        self._safe_broadcast(message)
    
    async def _default_trace_callback(self, scene: Scene) -> Dict[str, Any]:
        """Default trace callback implementation."""
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
        
        return {"pathGroups": path_groups}
    
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
        
        # Add mesh data if needed
        mesh_comp = obj["Mesh"]
        if mesh_comp:
            if obj.mesh_type in ["Cube", "Sphere", "Cylinder"]:
                # For primitives, get mesh data directly from component
                mesh_data = mesh_comp.get_mesh_data()
                # Calculate center for primitives
                if len(mesh_data["vertices"]) > 0:
                    vertices_array = np.array(mesh_data["vertices"])
                    center = vertices_array.mean(axis=0).tolist()
                    mesh_data["center"] = center
                data["mesh_data"] = mesh_data
            elif obj.mesh_type == "Custom" and obj.mesh_path:
                # For custom meshes, use static method to load from file
                from .components.mesh import MeshComponent
                mesh_data = MeshComponent.load_mesh_data(obj.mesh_path, self.default_scene_dir)
                if mesh_data:
                    data["mesh_data"] = mesh_data
        
        return data
    
    def mark_dirty(self):
        """Mark scene as having unsaved changes."""
        self.is_dirty = True
    
    def save_scene(self, force=False):
        """Save current scene state to disk."""
        if force or self.is_dirty:
            scene_data = serialize_scene(self.scene)
            with open(self.scene_file, 'w') as f:
                json.dump(scene_data, f, indent=2)
            logger.info("Scene saved to disk", extra={"file": str(self.scene_file)})
            self.is_dirty = False
            self.last_save_time = time.time()
    
    async def periodic_save(self):
        """Periodically save scene if dirty."""
        while True:
            await asyncio.sleep(self.save_interval)
            if self.is_dirty and (time.time() - self.last_save_time) >= self.save_interval:
                self.save_scene(force=True)
    
    def start_periodic_save(self):
        """Start the periodic save task."""
        if self.save_task is None:
            self.save_task = asyncio.create_task(self.periodic_save())
    
    def stop_periodic_save(self):
        """Stop the periodic save task."""
        if self.save_task:
            self.save_task.cancel()
            self.save_task = None
    
    async def broadcast(self, message: dict, exclude: WebSocket = None):
        """Broadcast message to all connected clients except the sender."""
        if message.get('type') == 'console_message':
            print(f"[Server] Broadcasting console_message to {len(self.connected_clients)} clients (excluding {1 if exclude else 0})")

        disconnected = set()
        for client in self.connected_clients:
            if client != exclude:
                try:
                    await client.send_json(message)
                except:
                    disconnected.add(client)

        # Clean up disconnected clients
        self.connected_clients -= disconnected

    def _safe_broadcast(self, message: dict, exclude: WebSocket = None):
        """Safely broadcast message, handling cases where there's no event loop."""
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self.broadcast(message, exclude))
        except RuntimeError:
            # No event loop running - force synchronous send
            if self.connected_clients:
                import threading

                def run_async_broadcast():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        new_loop.run_until_complete(self.broadcast(message, exclude))
                    finally:
                        new_loop.close()

                # Run in a thread to avoid blocking
                thread = threading.Thread(target=run_async_broadcast)
                thread.start()
                thread.join(timeout=1.0)  # Wait up to 1 second

                if self.verbose:
                    print(f"[Server] Sent {message.get('type')} to {len(self.connected_clients)} clients via sync mode")
            elif self.verbose:
                msg_type = message.get('type', 'unknown')
                print(f"[Server] No event loop and no clients - skipping broadcast for {msg_type}")
    
    async def handle_websocket_message(self, data: dict, websocket: WebSocket):
        """Handle incoming WebSocket message."""

        

        message_type = data.get('type')
        print(f"[Server] Handling websocket message: {message_type}")
        logger.info("Message received", extra={
            "type": message_type,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Special handling for trace request
        if message_type == 'get_trace':
            # Call user-defined trace callback
            trace_data = await self.trace_callback(self.scene)
            await websocket.send_json({
                "type": "trace_data",
                "data": trace_data
            })
            return
        
        # Use scene methods for updates
        if message_type == 'create_object':
            obj_data = data.get('object', {})
            # Extract data
            transform_data = obj_data.get('transform', {})
            material_data = obj_data.get('material')
            
            # Create object with basic properties
            obj = SceneObject(
                id=obj_data.get('id'),
                name=obj_data.get('name', 'Object'),
                mesh_type=obj_data.get('mesh', 'Custom'),
                visible=obj_data.get('visible', True),
                mesh_path=obj_data.get('model_path')
            )
            
            # Set transform data
            if transform_data:
                transform_comp = obj["Transform"]
                if transform_comp:
                    if 'position' in transform_data:
                        transform_comp.position = torch.tensor(transform_data['position'], dtype=torch.float32)
                    if 'rotation' in transform_data:
                        transform_comp.rotation = torch.tensor(transform_data['rotation'], dtype=torch.float32)
                    if 'scale' in transform_data:
                        transform_comp.scale = torch.tensor(transform_data['scale'], dtype=torch.float32)
            
            # Set material data
            if material_data:
                material_comp = obj["Material"]
                if material_comp:
                    if 'color' in material_data:
                        color_value = material_data['color']
                        # Convert hex color to RGB array if needed
                        if isinstance(color_value, str) and color_value.startswith('#'):
                            # Convert hex to RGB (0-1 range)
                            hex_color = color_value.lstrip('#')
                            r = int(hex_color[0:2], 16) / 255.0
                            g = int(hex_color[2:4], 16) / 255.0
                            b = int(hex_color[4:6], 16) / 255.0
                            material_comp.color = torch.tensor([r, g, b], dtype=torch.float32)
                        else:
                            material_comp.color = torch.tensor(color_value, dtype=torch.float32)
                    if 'metalness' in material_data:
                        material_comp.metalness = torch.tensor(material_data['metalness'], dtype=torch.float32)
                    if 'roughness' in material_data:
                        material_comp.roughness = torch.tensor(material_data['roughness'], dtype=torch.float32)
                    if 'texture' in material_data:
                        material_comp.texture = material_data['texture']
                    if 'permittivity' in material_data:
                        material_comp.permittivity = torch.tensor(material_data['permittivity'], dtype=torch.float32)
            
            # Add to scene (will trigger callback)
            self.scene.add_object(obj)
            
            # Set parent if specified
            parent_id = obj_data.get('parent_id')
            if parent_id:
                self.scene.set_parent(obj.id, parent_id)
                
        elif message_type == 'delete_object':
            object_id = data.get('object_id')
            if object_id:
                # Get all objects that will be deleted (for proper client notification)
                obj = self.scene.find_object(object_id)
                if obj:
                    # Collect all children that will be deleted
                    to_delete = [object_id]
                    def collect_children(transform_comp):
                        if hasattr(transform_comp, 'children'):
                            for child in transform_comp.children:
                                for obj_id, obj in self.scene.objects.items():
                                    child_transform = obj["Transform"]
                                    if child_transform == child:
                                        to_delete.append(obj_id)
                                        collect_children(child)
                                        break
                    
                    obj_transform = obj["Transform"]
                    if obj_transform:
                        collect_children(obj_transform)
                    
                    # Remove object (will trigger callbacks for each)
                    self.scene.remove_object(object_id)
                    
                    # Send delete messages for all children
                    for obj_id in to_delete[1:]:  # Skip first, already handled by callback
                        await self.broadcast({
                            "type": "object_deleted",
                            "object_id": obj_id
                        }, exclude=websocket)
                        
        elif message_type == 'update_transform':
            object_id = data.get('object_id')
            transform_data = data.get('transform', {})
            if object_id:
                self.scene.update_transform(
                    object_id,
                    position=transform_data.get('position'),
                    rotation=transform_data.get('rotation'),
                    scale=transform_data.get('scale')
                )
                
        elif message_type == 'update_material':
            object_id = data.get('object_id')
            material_data = data.get('material', {})
            if object_id:
                self.scene.update_material(
                    object_id,
                    color=material_data.get('color'),
                    metalness=material_data.get('metalness'),
                    roughness=material_data.get('roughness'),
                    texture=material_data.get('texture'),
                    permittivity=material_data.get('permittivity')
                )
                
        elif message_type == 'update_visibility':
            object_id = data.get('object_id')
            visible = data.get('visible')
            if object_id is not None and visible is not None:
                self.scene.update_visibility(object_id, visible)

        elif message_type == 'update_name':
            object_id = data.get('object_id')
            name = data.get('name')
            if object_id and name:
                obj = self.scene.get_object(object_id)
                if obj:
                    obj.name = name
                    # Notify all clients about the name change
                    await self.broadcast({
                        "type": "name_updated",
                        "object_id": object_id,
                        "name": name
                    }, exclude=websocket)

        elif message_type == 'update_tag':
            object_id = data.get('object_id')
            tag = data.get('tag', '')
            if object_id is not None:
                obj = self.scene.get_object(object_id)
                if obj:
                    obj.tag = tag
                    # Notify all clients about the tag change
                    await self.broadcast({
                        "type": "tag_updated",
                        "object_id": object_id,
                        "tag": tag
                    }, exclude=websocket)
                
        elif message_type == 'update_parent':
            object_id = data.get('object_id')
            parent_id = data.get('parent_id')
            if object_id:
                # Update parent (Scene class now handles transform preservation)
                success = self.scene.set_parent(object_id, parent_id)
                if not success:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Cannot create circular parent reference"
                    })
                        
        elif message_type == 'duplicate_object':
            # Use message handler for duplication logic
            result = await self.message_handler.handle_duplicate_object(data)
            if result:
                for msg in result:
                    await self.broadcast(msg, exclude=websocket)
        
        # Handle node editor messages
        elif message_type == 'update_node_editor':
            result = await self.message_handler.handle_update_node_editor(data)
            if result:
                await self.broadcast(result, exclude=websocket)
        
        elif message_type == 'get_node_editor':
            result = await self.message_handler.handle_get_node_editor(data)
            if result:
                await websocket.send_json(result)

        elif message_type == 'register_node':
            # Register a custom node type
            node_type = data.get('node_type')
            display_name = data.get('display_name', node_type)
            category = data.get('category', 'Custom')
            inputs = data.get('inputs', [])
            outputs = data.get('outputs', [])
            description = data.get('description', '')
            function_code = data.get('function', '')

            try:
                # Create a function from the provided code if any
                function = None
                if function_code:
                    exec_globals = {'np': np, 'torch': torch}
                    exec(function_code, exec_globals)
                    # Assume the function name matches the node_type
                    function = exec_globals.get(node_type)

                # Register the node
                self.scene.node_editor_manager.register_node(
                    node_type=node_type,
                    display_name=display_name,
                    category=category,
                    inputs=inputs,
                    outputs=outputs,
                    function=function,
                    description=description
                )

                await websocket.send_json({
                    "type": "node_registered",
                    "success": True,
                    "node_type": node_type
                })

                if self.verbose:
                    print(f"[Server] Registered custom node: {node_type}")

            except Exception as e:
                await websocket.send_json({
                    "type": "node_registered",
                    "success": False,
                    "error": str(e)
                })
                if self.verbose:
                    print(f"[Server] Failed to register node {node_type}: {e}")

        elif message_type == 'unregister_node':
            # Unregister a custom node type
            node_type = data.get('node_type')

            try:
                self.scene.node_editor_manager.unregister_node(node_type)

                await websocket.send_json({
                    "type": "node_unregistered",
                    "success": True,
                    "node_type": node_type
                })

                if self.verbose:
                    print(f"[Server] Unregistered node: {node_type}")

            except Exception as e:
                await websocket.send_json({
                    "type": "node_unregistered",
                    "success": False,
                    "error": str(e)
                })
                if self.verbose:
                    print(f"[Server] Failed to unregister node {node_type}: {e}")

        elif message_type == 'get_node_definitions':
            # Get all available node definitions
            definitions = self.scene.node_editor_manager.get_node_definitions()
            await websocket.send_json({
                "type": "node_definitions",
                "definitions": definitions
            })

        elif message_type == 'compile_node_graph':
            # Compile the node graph to Python code
            nodes = data.get('nodes', [])
            edges = data.get('edges', [])

            try:
                from .ui.node_compiler import NodeCompiler
                compiler = NodeCompiler()
                python_code = compiler.compile(nodes, edges)

                await websocket.send_json({
                    "type": "node_graph_compiled",
                    "success": True,
                    "code": python_code
                })

                if self.verbose:
                    print(f"[Server] Compiled node graph to Python function")
            except Exception as e:
                import traceback
                error_msg = str(e)
                error_trace = traceback.format_exc()

                await websocket.send_json({
                    "type": "node_graph_compiled",
                    "success": False,
                    "error": error_msg,
                    "trace": error_trace
                })

                if self.verbose:
                    print(f"[Server] Failed to compile node graph: {error_msg}")
                    print(error_trace)
        
        # Component system messages
        elif message_type == 'update_component':
            object_id = data.get('object_id')
            component_name = data.get('component_name')
            field_name = data.get('field_name')
            value = data.get('value')

            if object_id and component_name and field_name is not None:
                obj = self.scene.get_object(object_id)
                if obj:
                    # Temporarily remove the component_updated callback to avoid auto-broadcast
                    saved_callbacks = self.scene._callbacks.get('component_updated', []).copy()
                    self.scene._callbacks['component_updated'] = []

                    try:
                        # This will trigger _trigger_callbacks but won't broadcast since we removed the callback
                        success = obj.set_component_value(component_name, field_name, value)

                        if success:
                            # Manually broadcast to other clients (excluding sender)
                            await self.broadcast({
                                "type": "component_updated",
                                "object_id": object_id,
                                "component_name": component_name,
                                "field_name": field_name,
                                "value": value
                            }, exclude=websocket)
                    finally:
                        # Restore the callbacks
                        self.scene._callbacks['component_updated'] = saved_callbacks
        
        elif message_type == 'add_component':
            object_id = data.get('object_id')
            component_name = data.get('component_name')

            if object_id and component_name:
                obj = self.scene.get_object(object_id)
                if obj:
                    success = obj.add_component(component_name)
                    if success:
                        # Get the new component
                        new_component = obj.get_component(component_name)
                        if new_component:
                            # Broadcast update to ALL clients including the sender
                            await self.broadcast({
                                "type": "component_added",
                                "object_id": object_id,
                                "component": new_component.to_dict()
                            })  # Removed exclude=websocket so sender also gets the update

        elif message_type == 'click_component_button':
            object_id = data.get('object_id')
            component_name = data.get('component_name')
            button_name = data.get('button_name')

            if object_id and component_name and button_name:
                obj = self.scene.get_object(object_id)
                if obj:
                    component = obj.get_component(component_name)
                    if component:
                        try:
                            # Execute the button click
                            result = component.click_button(button_name)
                            # Send success response back to the client
                            await websocket.send_json({
                                "type": "button_click_result",
                                "object_id": object_id,
                                "component_name": component_name,
                                "button_name": button_name,
                                "success": True,
                                "result": str(result) if result else None
                            })
                        except Exception as e:
                            # Print error to backend console
                            import traceback
                            error_msg = f"[Button Error] {component_name}.{button_name} on {obj.name}:\n{str(e)}"
                            error_trace = traceback.format_exc()
                            print(f"\n{'='*60}")
                            print(error_msg)
                            print(error_trace)
                            print(f"{'='*60}\n")

                            # Also send to Console panel if available
                            if hasattr(self, 'console'):
                                self.console.log(error_msg, level='error')
                                self.console.log(error_trace, level='error')

                            # Send error response to client
                            await websocket.send_json({
                                "type": "button_click_result",
                                "object_id": object_id,
                                "component_name": component_name,
                                "button_name": button_name,
                                "success": False,
                                "error": str(e),
                                "traceback": error_trace
                            })
        
        elif message_type == 'remove_component':
            object_id = data.get('object_id')
            component_name = data.get('component_name')

            if object_id and component_name:
                obj = self.scene.get_object(object_id)
                if obj:
                    success = obj.remove_component(component_name)
                    if success:
                        # Broadcast update to ALL clients including the sender
                        await self.broadcast({
                            "type": "component_removed",
                            "object_id": object_id,
                            "component_name": component_name
                        })  # Removed exclude=websocket so sender also gets the update
        
        elif message_type == 'get_components':
            object_id = data.get('object_id')
            if object_id:
                obj = self.scene.get_object(object_id)
                if obj:
                    # Get all components as dictionaries
                    components = []
                    for comp_name, comp in obj.get_all_components().items():
                        components.append(comp.to_dict())
                    
                    # Send components to requesting client
                    await websocket.send_json({
                        "type": "components_updated",
                        "object_id": object_id,
                        "components": components
                    })
        
        elif message_type == 'get_component_definitions':
            try:
                # Get component definitions from the component system
                from .components import ComponentRegistry
                definitions = ComponentRegistry.get_definitions()
                print(f"[Server] Sending component definitions to client: {[d['name'] for d in definitions]}")
                await websocket.send_json({
                    "type": "component_definitions",
                    "definitions": definitions
                })
            except Exception as e:
                print(f"[Server] Error getting component definitions: {e}")
                logger.error(f"Failed to get component definitions: {e}", exc_info=True)
                # Send empty definitions to avoid blocking the client
                await websocket.send_json({
                    "type": "component_definitions",
                    "definitions": []
                })

        elif message_type == 'get_results_history':
            # Send all committed versions from ResultVisualizer
            all_versions = self.results.get_all_versions()
            await websocket.send_json({
                "type": "results_history",
                "versions": all_versions
            })
            if self.verbose:
                print(f"[Server] Sent {len(all_versions)} result versions to client")

        elif message_type == 'delete_results_version':
            # Delete a specific version from history
            version = data.get('version')
            if version is not None:
                success = self.results.delete_version(version)
                if success:
                    # Send updated history to all clients
                    all_versions = self.results.get_all_versions()
                    await self.broadcast({
                        "type": "results_history",
                        "versions": all_versions
                    })
                    if self.verbose:
                        print(f"[Server] Deleted version {version} and broadcast updated history")

        elif message_type == 'get_console_history':
            # Send console history to client
            history = self.console.get_history()
            await websocket.send_json({
                "type": "console_history",
                "messages": history
            })
            if self.verbose:
                print(f"[Server] Sent {len(history)} console messages to client")

        elif message_type == 'execute_console_command':
            # Execute a console command
            command = data.get('command', '')
            if command:
                result = await self.console.execute_command(command)
                # Send result back to client
                await websocket.send_json({
                    "type": "console_command_result",
                    "result": result
                })

        elif message_type == 'clear_console':
            # Clear console messages
            self.console.clear()
            # Broadcast clear to all clients
            await self.broadcast({
                "type": "console_cleared"
            })
    
    def _setup_routes(self):
        """Set up FastAPI routes."""
        # Enable CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @self.app.on_event("startup")
        async def startup_event():
            """Start periodic save task on startup."""
            self.start_periodic_save()
            if self.verbose:
                logger.info("Started periodic save task")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Save scene and stop periodic save task on shutdown."""
            self.save_scene(force=True)
            self.stop_periodic_save()
            if self.verbose:
                logger.info("Saved scene on shutdown")
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time scene synchronization."""
            await websocket.accept()
            self.connected_clients.add(websocket)

            print(f"[Server] Client connected. Total clients: {len(self.connected_clients)}")

            if self.verbose:
                logger.info("Client connected", extra={
                    "client_count": len(self.connected_clients)
                })
            
            # Send current scene to new client
            scene_data = serialize_scene(self.scene)
            
            # Add mesh data for custom objects and primitives
            for obj_id, obj_data in scene_data["objects"].items():
                obj = self.scene.find_object(obj_id)
                if obj:
                    mesh_comp = obj["Mesh"]
                    if mesh_comp:
                        if obj.mesh_type in ["Cube", "Sphere", "Cylinder"]:
                            # For primitives, get mesh data directly from component
                            mesh_data = mesh_comp.get_mesh_data()
                            # Calculate center for primitives
                            if len(mesh_data["vertices"]) > 0:
                                vertices_array = np.array(mesh_data["vertices"])
                                center = vertices_array.mean(axis=0).tolist()
                                mesh_data["center"] = center
                            obj_data["mesh_data"] = mesh_data
                        elif obj.mesh_type == "Custom" and obj.mesh_path:
                            # For custom meshes, use static method to load from file
                            from .components.mesh import MeshComponent
                            mesh_data = MeshComponent.load_mesh_data(obj.mesh_path, self.default_scene_dir)
                            if mesh_data:
                                obj_data["mesh_data"] = mesh_data
            
            await websocket.send_json({
                "type": "scene_update",
                "scene": scene_data
            })
            
            # Send node editor state if it exists (even if empty)
            if self.scene.node_editor:
                # Include parameters_info as 'parameters' for frontend compatibility
                node_editor_with_params = dict(self.scene.node_editor)
                node_editor_with_params['parameters'] = self.scene.parameters_info
                await websocket.send_json({
                    "type": "node_editor_state",
                    "node_editor": node_editor_with_params
                })

            # Send available node definitions
            node_definitions = self.scene.node_editor_manager.get_node_definitions()
            await websocket.send_json({
                "type": "node_definitions",
                "definitions": node_definitions
            })

            # Send results history
            all_versions = self.results.get_all_versions()
            if all_versions:
                await websocket.send_json({
                    "type": "results_history",
                    "versions": all_versions
                })
                if self.verbose:
                    print(f"[Server] Sent {len(all_versions)} result versions to new client")

            # Send console history
            console_history = self.console.get_history()
            if console_history:
                await websocket.send_json({
                    "type": "console_history",
                    "messages": console_history
                })
                if self.verbose:
                    print(f"[Server] Sent {len(console_history)} console messages to new client")

            try:
                while True:
                    data = await websocket.receive_json()
                    await self.handle_websocket_message(data, websocket)
                    
            except WebSocketDisconnect:
                self.connected_clients.remove(websocket)
                print(f"[Server] Client disconnected. Total clients: {len(self.connected_clients)}")
                if self.verbose:
                    logger.info("Client disconnected", extra={
                        "client_count": len(self.connected_clients)
                    })
                # Save scene when last client disconnects
                if len(self.connected_clients) == 0:
                    self.save_scene(force=True)
                    if self.verbose:
                        logger.info("Saved scene after last client disconnected")
            except Exception as e:
                logger.error("WebSocket error", extra={"error": str(e)})
                self.connected_clients.discard(websocket)
        
        @self.app.get("/")
        async def root():
            """Root endpoint with server info."""
            return {"message": "WebSocket Scene Server", "ws_endpoint": "/ws"}
        
        @self.app.post("/save-scene")
        async def save_scene():
            """Manually save the current scene."""
            try:
                self.save_scene(force=True)
                return {"success": True, "message": "Scene saved successfully"}
            except Exception as e:
                logger.error("Failed to save scene", extra={"error": str(e)})
                return {"success": False, "error": str(e)}
        
        @self.app.get("/scenes/{scene_name}/models/{model_file}")
        async def get_model_file(scene_name: str, model_file: str):
            """Serve model files from the models directory."""
            # Construct file path
            file_path = self.scenes_dir / scene_name / "models" / model_file
            
            # Security check
            try:
                file_path = file_path.resolve()
                models_dir = (self.scenes_dir / scene_name / "models").resolve()
                if not str(file_path).startswith(str(models_dir)):
                    return {"error": "Invalid file path"}
            except:
                return {"error": "Invalid file path"}
            
            # Check if file exists
            if not file_path.exists() or not file_path.is_file():
                return {"error": "File not found"}
            
            return FileResponse(file_path)
        
        @self.app.post("/upload-mesh")
        async def upload_mesh(
            mesh: UploadFile = File(...),
            filename: str = Form(...)
        ):
            """Handle mesh file upload and add to scene."""
            try:
                # Read file content
                content = await mesh.read()
                
                # Generate unique filename
                object_id = f"mesh-{uuid4().hex[:8]}"
                file_extension = os.path.splitext(filename)[1]
                saved_filename = f"{object_id}{file_extension}"
                file_path = self.models_dir / saved_filename
                
                # Save the file
                with open(file_path, 'wb') as f:
                    f.write(content)
                
                # Load mesh using trimesh
                try:
                    mesh_data = trimesh.load(file_path)
                    
                    # Convert scene to mesh if needed
                    if isinstance(mesh_data, trimesh.Scene):
                        meshes = list(mesh_data.geometry.values())
                        if meshes:
                            mesh_data = meshes[0]
                        else:
                            return {"error": "No mesh found in file"}
                    
                    # Calculate normalization scale
                    bounds = mesh_data.bounds
                    scale = 2.0 / max(bounds[1] - bounds[0])
                    
                    # Create object name
                    object_name = os.path.splitext(filename)[0]
                    
                    # Create new scene object
                    new_object = SceneObject(
                        id=object_id,
                        name=object_name,
                        mesh_type="Custom",
                        visible=True,
                        mesh_path=f"models/{saved_filename}"
                    )

                    # Load mesh data into MeshComponent
                    mesh_comp = new_object["Mesh"]
                    if mesh_comp:
                        mesh_comp.load_from_file(f"models/{saved_filename}", self.default_scene_dir)

                    # Set transform with scale
                    transform_comp = new_object["Transform"]
                    if transform_comp:
                        transform_comp.position = torch.zeros(3)
                        transform_comp.rotation = torch.zeros(3)
                        transform_comp.scale = torch.tensor([scale, scale, scale], dtype=torch.float32)

                    # Set default material
                    material_comp = new_object["Material"]
                    if material_comp:
                        material_comp.color = torch.tensor([0.5, 0.5, 0.5, 1.0], dtype=torch.float32)
                        material_comp.metalness = torch.tensor(0.5, dtype=torch.float32)
                        material_comp.roughness = torch.tensor(0.5, dtype=torch.float32)
                        material_comp.permittivity = torch.tensor(1.0, dtype=torch.float32)
                    
                    # Add to scene (will trigger broadcast)
                    self.scene.add_object(new_object)
                    
                    logger.info("Mesh uploaded", extra={
                        "file": filename,
                        "object_id": object_id,
                        "saved_as": saved_filename,
                        "vertices": len(mesh_data.vertices),
                        "faces": len(mesh_data.faces)
                    })
                    
                    return {
                        "success": True,
                        "object_id": object_id,
                        "model_path": f"models/{saved_filename}",
                        "vertices": len(mesh_data.vertices),
                        "faces": len(mesh_data.faces)
                    }
                    
                except Exception as e:
                    logger.error("Failed to load mesh", extra={"error": str(e)})
                    # Clean up file
                    if file_path.exists():
                        file_path.unlink()
                    return {"error": f"Failed to load mesh: {str(e)}"}
                    
            except Exception as e:
                logger.error("Upload error", extra={"error": str(e)})
                return {"error": f"Upload failed: {str(e)}"}
    
    def start(self, host: str = "0.0.0.0", port: int = 8000, threaded: bool = True, start_editor: bool = True):
        """Start the server.
        
        Args:
            host: Host to bind to (default: "0.0.0.0")
            port: Port to bind to (default: 8000)
            threaded: If True, runs server in a separate daemon thread (default: False)
        """
        def run_server():
            log_level = "info" if self.verbose else "error"
            uvicorn.run(
                self.app, 
                host=host, 
                port=port, 
                log_level=log_level,
                access_log=self.verbose,
                log_config=None
            )
        
        if threaded:
            import threading
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            if self.verbose:
                logger.info(f"Server started in background thread on {host}:{port}")
        else:
            print(f"Server started on {host}:{port}")
            run_server()
            
        if start_editor:
            self.start_editor()

    def start_editor(self):
        self.editor = Editor(verbose=self.verbose)
        self.editor.start()


# Backward compatibility - create a default server instance
if __name__ == "__main__":
    server = Server()
    server.start(threaded=False)