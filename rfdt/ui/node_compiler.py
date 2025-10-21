"""
Node Editor Compiler - Compiles node graph to Python functions
"""

from typing import Dict, List, Any, Set, Optional, Tuple
import json
from dataclasses import dataclass
from collections import defaultdict
from .nodes import get_registry, execute_node


@dataclass
class CompiledNode:
    """Represents a node that has been analyzed for compilation"""
    id: str
    type: str
    data: Dict[str, Any]
    inputs: List[Tuple[str, str]]  # List of (source_node_id, source_handle)
    outputs: List[str]  # List of handle names
    variable_name: str  # Variable name in generated code


class NodeCompiler:
    """Compiles a node graph to a Python function"""

    def __init__(self):
        self.nodes: Dict[str, Any] = {}
        self.edges: List[Dict[str, Any]] = []
        self.compiled_nodes: Dict[str, CompiledNode] = {}
        self.topological_order: List[str] = []

    def compile(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> str:
        """
        Main compilation method

        Args:
            nodes: List of nodes from the node editor
            edges: List of edges/connections from the node editor

        Returns:
            Generated Python function as string
        """
        self.nodes = {node['id']: node for node in nodes}
        self.edges = edges

        # Analyze the graph
        self._analyze_nodes()

        # Perform topological sort
        self._topological_sort()

        # Generate Python code
        return self._generate_python_code()

    def _analyze_nodes(self):
        """Analyze nodes and create CompiledNode instances"""
        for node_id, node in self.nodes.items():
            # Find inputs for this node
            inputs = []
            for edge in self.edges:
                if edge['target'] == node_id:
                    inputs.append((edge['source'], edge['sourceHandle']))

            # Get outputs from node data
            outputs = []
            if node['data'].get('outputs'):
                outputs = [f"output-{i}" for i in range(len(node['data']['outputs']))]

            # Create variable name based on node type and id
            var_name = self._generate_variable_name(node)

            self.compiled_nodes[node_id] = CompiledNode(
                id=node_id,
                type=node['type'],
                data=node['data'],
                inputs=inputs,
                outputs=outputs,
                variable_name=var_name
            )

    def _generate_variable_name(self, node: Dict[str, Any]) -> str:
        """Generate a Python variable name for a node"""
        node_type = node['type']
        node_id = node['id'].replace('-', '_')

        # Special handling for different node types
        if node_type == 'shader':
            if node['data'].get('nodeType') == 'parameter':
                # Parameter nodes use their label as variable name
                label = node['data'].get('label', 'param').lower()
                return label.replace(' ', '_').replace('-', '_')
            elif node['data'].get('nodeType') == 'constant':
                return f"const_{node_id}"
            elif node['data'].get('nodeType') == 'math':
                op = node['data'].get('label', 'op').lower()
                return f"{op}_{node_id}"
        elif node_type == 'object':
            return f"obj_{node['data'].get('label', node_id).replace(' ', '_')}"
        elif node_type == 'objectGroup':
            return f"group_{node_id}"

        return f"node_{node_id}"

    def _topological_sort(self):
        """Perform topological sort on the graph"""
        # Build adjacency list
        graph = defaultdict(list)
        in_degree = defaultdict(int)

        # Initialize all nodes
        for node_id in self.nodes:
            in_degree[node_id] = 0

        # Build the graph
        for edge in self.edges:
            source = edge['source']
            target = edge['target']
            graph[source].append(target)
            in_degree[target] += 1

        # Find nodes with no dependencies
        queue = [node_id for node_id in self.nodes if in_degree[node_id] == 0]
        self.topological_order = []

        while queue:
            node = queue.pop(0)
            self.topological_order.append(node)

            # Process neighbors
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(self.topological_order) != len(self.nodes):
            raise ValueError("Graph contains cycles, cannot compile")

    def _generate_python_code(self) -> str:
        """Generate the Python function code"""
        lines = []

        # Function signature (hyperparameters kept for backward compatibility)
        lines.append("def apply_node_values(scene, hyperparameters=None):")
        lines.append("    import numpy as np")
        lines.append("    ")

        # Generate code for each node in topological order
        for node_id in self.topological_order:
            node = self.compiled_nodes[node_id]
            code = self._generate_node_code(node)
            if code:
                lines.extend(["    " + line for line in code.split('\n') if line])
                lines.append("")  # Add blank line between nodes

        return '\n'.join(lines)

    def _generate_node_code(self, node: CompiledNode) -> str:
        """Generate code for a specific node"""
        lines = []

        if node.type == 'shader':
            lines.append(self._generate_shader_node_code(node))
        elif node.type == 'object':
            lines.append(self._generate_object_node_code(node))
        elif node.type == 'objectGroup':
            lines.append(self._generate_object_group_node_code(node))

        return '\n'.join(lines)

    def _generate_shader_node_code(self, node: CompiledNode) -> str:
        """Generate code for shader nodes (math, constants, parameters)"""
        node_type = node.data.get('nodeType')

        # Check if this is a custom registered node
        registry = get_registry()
        if registry.get(node_type):
            return self._generate_custom_node_code(node)

        if node_type == 'parameter':
            # Use parameterId to reference the actual parameter in scene.parameters
            param_id = node.data.get('parameterId')
            # The label is the parameter name
            param_name = node.data.get('label', 'param')
            param_type = node.data.get('outputs', [{}])[0].get('type', 'float')

            if param_id:
                # Access parameter from scene.parameters using ID or name
                # get_parameter_value now supports both ID and name lookup
                if param_type == 'float':
                    return f"{node.variable_name} = scene.get_parameter_value('{param_id}').item() if scene.get_parameter_value('{param_id}') is not None else 0.0"
                elif param_type in ['vector2', 'vector3', 'vector4']:
                    return f"{node.variable_name} = scene.get_parameter_value('{param_id}').numpy() if scene.get_parameter_value('{param_id}') is not None else np.zeros({int(param_type[-1])})"
                elif param_type == 'bool':
                    return f"{node.variable_name} = scene.get_parameter_value('{param_id}').item() if scene.get_parameter_value('{param_id}') is not None else False"
            else:
                # Fallback: try using parameter name directly
                if param_type == 'float':
                    return f"{node.variable_name} = scene.get_parameter_value('{param_name}').item() if scene.get_parameter_value('{param_name}') is not None else 0.0"
                elif param_type in ['vector2', 'vector3', 'vector4']:
                    return f"{node.variable_name} = scene.get_parameter_value('{param_name}').numpy() if scene.get_parameter_value('{param_name}') is not None else np.zeros({int(param_type[-1])})"
                elif param_type == 'bool':
                    return f"{node.variable_name} = scene.get_parameter_value('{param_name}').item() if scene.get_parameter_value('{param_name}') is not None else False"

        elif node_type == 'constant':
            value = node.data.get('value', 0)
            return f"{node.variable_name} = {repr(value)}"

        elif node_type == 'math':
            operation = node.data.get('label', 'Add').lower()

            if len(node.inputs) >= 2:
                input_a = self._get_input_variable(node.inputs[0])
                input_b = self._get_input_variable(node.inputs[1])

                if operation == 'add':
                    return f"{node.variable_name} = {input_a} + {input_b}"
                elif operation == 'subtract':
                    return f"{node.variable_name} = {input_a} - {input_b}"
                elif operation == 'multiply':
                    return f"{node.variable_name} = {input_a} * {input_b}"
                elif operation == 'divide':
                    return f"{node.variable_name} = {input_a} / {input_b} if {input_b} != 0 else 0"
                elif operation == 'power':
                    return f"{node.variable_name} = {input_a} ** {input_b}"
            elif len(node.inputs) == 1:
                input_val = self._get_input_variable(node.inputs[0])

                if operation == 'sqrt':
                    return f"{node.variable_name} = np.sqrt({input_val})"
                elif operation == 'abs':
                    return f"{node.variable_name} = abs({input_val})"

        return ""

    def _generate_object_node_code(self, node: CompiledNode) -> str:
        """Generate code for object nodes"""
        lines = []
        object_name = node.data.get('label', 'Object')

        lines.append(f"obj = scene.get_object_by_name('{object_name}')")
        lines.append("if obj:")

        for edge in self.edges:
            if edge['target'] == node.id:
                target_handle = edge['targetHandle']
                source_var = self._get_input_variable((edge['source'], edge['sourceHandle']))

                parts = target_handle.split('-')
                if len(parts) >= 3:
                    component_name = parts[1]
                    field_name = '-'.join(parts[2:])

                    if component_name.lower() == 'transform':
                        lines.append(f"    obj.transform.{field_name} = {source_var}")
                    else:
                        lines.append(f"    comp = obj.get_component('{component_name}')")
                        lines.append(f"    if comp:")
                        lines.append(f"        comp.{field_name} = {source_var}")

        return '\n'.join(lines)

    def _generate_object_group_node_code(self, node: CompiledNode) -> str:
        """Generate code for object group nodes"""
        lines = []
        filter_by = node.data.get('filterBy', 'name')
        filter_value = node.data.get('filterValue', '')

        if filter_by == 'name':
            lines.append(f"import re")
            lines.append(f"pattern = re.compile(r'{filter_value.replace('*', '.*')}')")
            lines.append(f"matching_objects = [obj for obj in scene.objects if pattern.match(obj.name)]")
        else:
            lines.append(f"matching_objects = [obj for obj in scene.objects if obj.tag == '{filter_value}']")

        lines.append("for obj in matching_objects:")

        for edge in self.edges:
            if edge['target'] == node.id:
                target_handle = edge['targetHandle']
                source_var = self._get_input_variable((edge['source'], edge['sourceHandle']))

                parts = target_handle.split('-')
                if len(parts) >= 3:
                    component_name = parts[1]
                    field_name = '-'.join(parts[2:])

                    if component_name.lower() == 'transform':
                        lines.append(f"    obj.transform.{field_name} = {source_var}")
                    else:
                        lines.append(f"    comp = obj.get_component('{component_name}')")
                        lines.append(f"    if comp:")
                        lines.append(f"        comp.{field_name} = {source_var}")

        return '\n'.join(lines)

    def _generate_custom_node_code(self, node: CompiledNode) -> str:
        """Generate code for custom registered nodes"""
        node_type = node.data.get('nodeType')
        lines = []

        registry = get_registry()
        node_def = registry.get(node_type)

        if not node_def:
            return f"# Warning: Unknown node type: {node_type}"

        # Collect input values
        input_args = []
        for i, (source_node_id, source_handle) in enumerate(node.inputs):
            input_var = self._get_input_variable((source_node_id, source_handle))
            input_args.append(input_var)

        # Generate function call
        if input_args or node_def.inputs:
            lines.append(f"# Execute node: {node_type}")
            lines.append(f"from rfdt.ui.nodes import execute_node")
            lines.append(f"inputs = {{")

            # Map inputs to port names
            if node_def:
                for i, input_port in enumerate(node_def.inputs):
                    if i < len(input_args):
                        lines.append(f"    '{input_port.name}': {input_args[i]},")

            lines.append(f"}}")
            lines.append(f"result = execute_node('{node_type}', inputs)")

            # Extract outputs
            if node_def and len(node_def.outputs) == 1:
                lines.append(f"{node.variable_name} = result['{node_def.outputs[0].name}']")
            else:
                lines.append(f"{node.variable_name} = result")
        else:
            # Node with no inputs (e.g., constant nodes)
            lines.append(f"from rfdt.ui.nodes import execute_node")
            output_name = node_def.outputs[0].name if node_def.outputs else 'output'
            lines.append(f"{node.variable_name} = execute_node('{node_type}', {{}})['{output_name}']")

        return '\n'.join(lines)

    def _get_input_variable(self, input_spec: Tuple[str, str]) -> str:
        """Get the variable name for an input connection"""
        source_node_id, source_handle = input_spec
        if source_node_id in self.compiled_nodes:
            return self.compiled_nodes[source_node_id].variable_name
        return "None"