"""
Mesh component with vertex and face data.
"""
from typing import Optional, Dict, Any
from pathlib import Path
import torch
import numpy as np
import math
import trimesh
import logging
from .base import Component, component, field

logger = logging.getLogger(__name__)


@component(name="Mesh")
class MeshComponent(Component):
    """Mesh component for geometry."""
    
    # Component fields - only vertices is exposed as a field
    vertices = field(torch.empty((0, 3), dtype=torch.float32), description="Vertex positions (Nx3)")
    
    # UI fields  
    mesh_type = field("Custom", description="Type of mesh")
    visible = field(True, description="Whether the object is visible")

    def __init__(self, **kwargs):
        """Initialize mesh component."""
        super().__init__(**kwargs)
        
        # Internal (non-field) data
        self.faces = torch.empty((0, 3), dtype=torch.long)  # Face indices
        self.filename = None  # Source file path
        
        # If mesh_type is a primitive, generate the mesh
        if self.mesh_type in ['Cube', 'Sphere', 'Cylinder']:
            self._generate_primitive()
    
    def _generate_primitive(self):
        """Generate primitive mesh based on mesh_type."""
        if self.mesh_type == 'Cube':
            self._create_cube()
        elif self.mesh_type == 'Sphere':
            self._create_sphere()
        elif self.mesh_type == 'Cylinder':
            self._create_cylinder()
    
    def _create_cube(self, size: float = 1.0):
        """Generate cube mesh with given size."""
        half_size = size / 2
        
        # Define 8 vertices of a cube
        vertices = np.array([
            [-half_size, -half_size, -half_size],  # 0
            [ half_size, -half_size, -half_size],  # 1
            [ half_size,  half_size, -half_size],  # 2
            [-half_size,  half_size, -half_size],  # 3
            [-half_size, -half_size,  half_size],  # 4
            [ half_size, -half_size,  half_size],  # 5
            [ half_size,  half_size,  half_size],  # 6
            [-half_size,  half_size,  half_size],  # 7
        ], dtype=np.float32)
        
        faces = np.array([
            # Front (-z)
            [0, 2, 1], [0, 3, 2],
            # Back (+z)
            [4, 5, 6], [4, 6, 7],
            # Left (-x)
            [0, 7, 3], [0, 4, 7],
            # Right (+x)
            [1, 2, 6], [1, 6, 5],
            # Top (+y)
            [3, 7, 6], [3, 6, 2],
            # Bottom (-y)
            [0, 1, 5], [0, 5, 4]
        ], dtype=np.int64)
        
        self.vertices = torch.from_numpy(vertices)
        self.faces = torch.from_numpy(faces)
    
    def _create_sphere(self, radius: float = 0.5, subdivisions: int = 2):
        """Generate sphere mesh using trimesh icosphere."""
        # 使用 trimesh 内置的正二十面体细分球体生成器
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)

        # 转为 torch tensor 以保持接口一致
        self.vertices = torch.from_numpy(mesh.vertices.astype('float32'))
        self.faces = torch.from_numpy(mesh.faces.astype('int64'))
    
    def _create_cylinder(self, radius: float = 0.5, height: float = 1.0, segments: int = 32):
        """Generate cylinder mesh."""
        vertices = []
        faces = []
        
        # Generate vertices for top and bottom circles
        for y in [-height/2, height/2]:
            for i in range(segments):
                angle = 2 * math.pi * i / segments
                x = radius * math.cos(angle)
                z = radius * math.sin(angle)
                vertices.append([x, y, z])
        
        # Add center vertices for caps
        bottom_center = len(vertices)
        vertices.append([0, -height/2, 0])
        top_center = len(vertices)
        vertices.append([0, height/2, 0])
        
        # Create side faces
        for i in range(segments):
            next_i = (i + 1) % segments
            
            # Bottom vertex indices
            b1 = i
            b2 = next_i
            
            # Top vertex indices
            t1 = i + segments
            t2 = next_i + segments
            
            # Two triangles for the side
            faces.extend([
                [b1, t1, t2],
                [b1, t2, b2]
            ])
            
            # Bottom cap triangle
            faces.append([bottom_center, b2, b1])
            
            # Top cap triangle
            faces.append([top_center, t1, t2])
        
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int64)
        
        self.vertices = torch.from_numpy(vertices)
        self.faces = torch.from_numpy(faces)
    
    def load_from_file(self, file_path: str, base_dir: Optional[Path] = None) -> bool:
        """
        Load mesh data from file.
        
        Args:
            file_path: Path to the mesh file (relative or absolute)
            base_dir: Base directory for relative paths
            
        Returns:
            True if loading succeeded, False otherwise
        """
        try:
            if base_dir is None:
                base_dir = Path.cwd()
            
            full_path = base_dir / file_path
            if not full_path.exists():
                logger.warning(f"Mesh file not found: {full_path}")
                return False
                
            mesh_data = trimesh.load(full_path)
            
            # Convert to simple format
            if isinstance(mesh_data, trimesh.Scene):
                meshes = list(mesh_data.geometry.values())
                if meshes:
                    mesh_data = meshes[0]
                else:
                    logger.warning(f"No meshes found in scene file: {full_path}")
                    return False
            
            # Store mesh data
            self.vertices = torch.from_numpy(mesh_data.vertices.astype(np.float32))
            self.faces = torch.from_numpy(mesh_data.faces.astype(np.int64))
            self.filename = file_path
            self.mesh_type = "Custom"
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load mesh from {file_path}: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert mesh component to dictionary for serialization."""
        result = super().to_dict()
        
        # Add internal mesh data
        result["faces"] = self.faces.tolist() if isinstance(self.faces, torch.Tensor) else self.faces
        result["filename"] = self.filename
        
        return result
    
    def from_dict(self, data: Dict[str, Any]):
        """Load mesh component from dictionary."""
        super().from_dict(data)
        
        # Load internal mesh data
        if "faces" in data:
            self.faces = torch.tensor(data["faces"], dtype=torch.long)
        if "filename" in data:
            self.filename = data["filename"]
        
        # Generate primitive if needed
        if self.mesh_type in ['Cube', 'Sphere', 'Cylinder'] and self.vertices.shape[0] == 0:
            self._generate_primitive()
    
    def get_mesh_data(self) -> Dict[str, Any]:
        """Get mesh data for rendering."""
        return {
            "vertices": self.vertices.tolist() if isinstance(self.vertices, torch.Tensor) else self.vertices,
            "faces": self.faces.tolist() if isinstance(self.faces, torch.Tensor) else self.faces,
            "type": self.mesh_type,
            "filename": self.filename
        }
    
    @staticmethod
    def create_cube(size: float = 1.0) -> 'MeshComponent':
        """Create a cube mesh component."""
        mesh = MeshComponent(mesh_type='Cube')
        mesh._create_cube(size)
        return mesh
    
    @staticmethod
    def create_sphere(radius: float = 0.5, subdivisions: int = 2) -> 'MeshComponent':
        """Create a sphere mesh component."""
        mesh = MeshComponent(mesh_type='Sphere')
        mesh._create_sphere(radius, subdivisions)
        return mesh
    
    @staticmethod
    def create_cylinder(radius: float = 0.5, height: float = 1.0, segments: int = 32) -> 'MeshComponent':
        """Create a cylinder mesh component."""
        mesh = MeshComponent(mesh_type='Cylinder')
        mesh._create_cylinder(radius, height, segments)
        return mesh

    @staticmethod
    def load_mesh_data(mesh_path: str, base_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Load mesh data from file for transmission to frontend.

        Args:
            mesh_path: Relative path to mesh file (e.g., "models/file.obj")
            base_dir: Base directory containing the mesh file

        Returns:
            Dictionary with vertices, faces, and center, or None if loading fails
        """
        try:
            full_path = base_dir / mesh_path
            if not full_path.exists():
                logger.warning(f"Mesh file not found: {full_path}")
                return None

            mesh_data = trimesh.load(full_path)

            # Convert to simple format if needed
            if isinstance(mesh_data, trimesh.Scene):
                meshes = list(mesh_data.geometry.values())
                if meshes:
                    mesh_data = meshes[0]
                else:
                    logger.warning(f"No meshes found in scene file: {full_path}")
                    return None

            # Calculate center for centering in frontend
            center = mesh_data.centroid.tolist()

            return {
                "vertices": mesh_data.vertices.astype(np.float32).tolist(),
                "faces": mesh_data.faces.astype(np.int64).tolist(),
                "center": center
            }

        except Exception as e:
            logger.error(f"Failed to load mesh data from {mesh_path}: {e}")
            return None