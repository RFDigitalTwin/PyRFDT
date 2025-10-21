"""
Node editor management for scene - handles node graph state.
"""
from typing import List, Dict, Any, Optional


class NodeEditorManager:
    """Manages node editor state including nodes, edges, and their compilation."""

    def __init__(self, scene):
        """Initialize node editor manager.

        Args:
            scene: Reference to the parent scene object for event emission
        """
        self.scene = scene
        # Node editor state (without parameters)
        self.node_editor = {
            'nodes': [],
            'edges': [],
            'differentiableParams': []  # List of parameter IDs that require gradients
        }
        # Import and initialize node registry
        from .nodes import get_registry
        self.node_registry = get_registry()

    def update(self, nodes: List[Any] = None, edges: List[Any] = None,
               parameters: List[Any] = None, differentiableParams: List[str] = None) -> None:
        """Update node editor state and emit event."""
        if nodes is not None:
            self.node_editor['nodes'] = nodes
        if edges is not None:
            self.node_editor['edges'] = edges

        # Handle parameters separately through parameter manager
        if parameters is not None:
            self.scene.parameter_manager.update_parameters_info(parameters)

        if differentiableParams is not None:
            self.node_editor['differentiableParams'] = differentiableParams
            self.scene.parameter_manager.update_parameters_grad(differentiableParams)

        self.scene._emit('node_editor_updated', node_editor=self.node_editor)

    def get_state(self) -> Dict[str, Any]:
        """Get current node editor state."""
        return self.node_editor

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set node editor state from saved data."""
        if 'nodes' in state:
            self.node_editor['nodes'] = state['nodes']
        if 'edges' in state:
            self.node_editor['edges'] = state['edges']
        if 'differentiableParams' in state:
            self.node_editor['differentiableParams'] = state['differentiableParams']

        # Migrate old parameters to parameter manager if needed
        if 'parameters' in state:
            self.scene.parameter_manager.update_parameters_info(state['parameters'])
            # Remove from node_editor as it's now in parameter_manager
            del state['parameters']

    def get_node_definitions(self) -> Dict[str, Any]:
        """Get all available node definitions for the frontend."""
        return self.node_registry.to_dict()

    def register_node(self, node_type: str, display_name: str, category: str,
                      inputs: List[Dict], outputs: List[Dict],
                      function: Any = None, description: str = "") -> None:
        """Register a custom node type.

        Args:
            node_type: Unique identifier for the node type
            display_name: Display name in UI
            category: Category for menu organization
            inputs: List of input port definitions
            outputs: List of output port definitions
            function: Optional function to execute
            description: Optional description
        """
        from .nodes import NodeDefinition, PortDefinition

        # Convert input/output dicts to PortDefinitions
        input_ports = [
            PortDefinition(
                name=i.get('name', f'input{idx}'),
                type=i.get('type', 'float'),
                default=i.get('default'),
                description=i.get('description', '')
            )
            for idx, i in enumerate(inputs)
        ]

        output_ports = [
            PortDefinition(
                name=o.get('name', f'output{idx}'),
                type=o.get('type', 'float'),
                description=o.get('description', '')
            )
            for idx, o in enumerate(outputs)
        ]

        node_def = NodeDefinition(
            node_type=node_type,
            display_name=display_name,
            category=category,
            inputs=input_ports,
            outputs=output_ports,
            function=function,
            description=description,
            is_builtin=False
        )

        self.node_registry.register(node_def)

        # Emit event to update frontend
        self.scene._emit('node_definitions_updated', definitions=self.get_node_definitions())

    def unregister_node(self, node_type: str) -> None:
        """Unregister a custom node type (cannot unregister built-in nodes)."""
        self.node_registry.unregister(node_type)

        # Emit event to update frontend
        self.scene._emit('node_definitions_updated', definitions=self.get_node_definitions())