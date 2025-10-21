"""
Parameter management for scene - handles hyperparameters and their tensors.
"""
from typing import Dict, List, Any, Optional
import torch


class ParametersDict(dict):
    """A dict wrapper that notifies the scene when parameters are modified."""

    def __init__(self, manager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._manager = manager

    def __setitem__(self, key, value):
        """Override setitem to trigger sync when a parameter is modified."""
        # If value is not a tensor, convert it
        if not isinstance(value, torch.Tensor):
            # Try to preserve dtype and grad from existing tensor
            if key in self:
                existing = self[key]
                if existing.dtype == torch.bool:
                    value = torch.tensor(bool(value), dtype=torch.bool)
                else:
                    value = torch.tensor(value, dtype=existing.dtype, requires_grad=existing.requires_grad)
            else:
                value = torch.tensor(value)

        super().__setitem__(key, value)
        # Auto-sync to frontend when value is changed
        self._manager.sync_parameter_values()

    def update(self, *args, **kwargs):
        """Override update to trigger sync."""
        super().update(*args, **kwargs)
        self._manager.sync_parameter_values()


class ParameterManager:
    """Manages parameters and their PyTorch tensor representations."""

    def __init__(self, scene):
        """Initialize parameter manager.

        Args:
            scene: Reference to the parent scene object for event emission
        """
        self.scene = scene
        self._parameters: Dict[str, torch.Tensor] = {}  # Actual parameter values (PyTorch tensors)
        self.parameters_info: List[Dict[str, Any]] = []  # Parameter definitions (type, name, etc.)

    @property
    def parameters(self):
        """Get the parameters dictionary (with auto-sync on modification)."""
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        """Set the parameters dictionary."""
        self._parameters = value

    def update_parameters_info(self, parameters_info: List[Dict[str, Any]]) -> None:
        """Update parameter definitions and instantiate tensors."""
        self.parameters_info = parameters_info
        self._instantiate_parameters()
        self.scene._emit('parameters_updated', parameters_info=self.parameters_info)

    def _instantiate_parameters(self) -> None:
        """Create PyTorch tensors from parameter definitions."""
        new_parameters = {}

        for param_info in self.parameters_info:
            param_id = param_info['id']
            param_name = param_info.get('name', param_id)  # Use name as key, fallback to id
            param_type = param_info['type']
            param_value = param_info.get('value', 0)

            # Check if we need gradient tracking
            # First check the param_info itself, then fall back to differentiableParams list for backward compatibility
            requires_grad = param_info.get('differentiable', False) or param_id in self.scene.node_editor.get('differentiableParams', [])

            # Always create/update tensor with new value from param_info
            # Create new tensor based on type
            if param_type == 'float':
                tensor_value = float(param_value) if param_value is not None else 0.0
                new_parameters[param_name] = torch.tensor(tensor_value, dtype=torch.float32, requires_grad=requires_grad)
            elif param_type == 'vector2':
                if isinstance(param_value, str):
                    # Parse string like "(0, 0)"
                    values = [float(x.strip()) for x in param_value.strip('()').split(',')]
                elif isinstance(param_value, (list, tuple)):
                    values = list(param_value)
                else:
                    values = [0.0, 0.0]
                new_parameters[param_name] = torch.tensor(values[:2], dtype=torch.float32, requires_grad=requires_grad)
            elif param_type == 'vector3':
                if isinstance(param_value, str):
                    # Parse string like "(0, 0, 0)"
                    values = [float(x.strip()) for x in param_value.strip('()').split(',')]
                elif isinstance(param_value, (list, tuple)):
                    values = list(param_value)
                else:
                    values = [0.0, 0.0, 0.0]
                new_parameters[param_name] = torch.tensor(values[:3], dtype=torch.float32, requires_grad=requires_grad)
            elif param_type == 'vector4':
                if isinstance(param_value, str):
                    # Parse string like "(0, 0, 0, 0)"
                    values = [float(x.strip()) for x in param_value.strip('()').split(',')]
                elif isinstance(param_value, (list, tuple)):
                    values = list(param_value)
                else:
                    values = [0.0, 0.0, 0.0, 0.0]
                new_parameters[param_name] = torch.tensor(values[:4], dtype=torch.float32, requires_grad=requires_grad)
            elif param_type == 'bool':
                bool_value = bool(param_value) if param_value is not None else False
                new_parameters[param_name] = torch.tensor(bool_value, dtype=torch.bool)  # bool doesn't support requires_grad

        self._parameters = ParametersDict(self, new_parameters)

    def update_parameters_grad(self, differentiableParams: List[str]) -> None:
        """Update requires_grad for parameters based on differentiableParams list."""
        # Update both the parameters_info and the actual tensors
        differentiable_set = set(differentiableParams)

        for param_info in self.parameters_info:
            param_id = param_info['id']
            param_name = param_info.get('name', param_id)

            # Update the differentiable field in param_info
            param_info['differentiable'] = param_id in differentiable_set

            # Update the tensor if it exists
            if param_name in self.parameters:
                tensor = self.parameters[param_name]
                should_have_grad = param_id in differentiable_set

                if should_have_grad and not tensor.requires_grad:
                    # Enable gradient
                    self.parameters[param_name] = tensor.detach().requires_grad_(True)
                elif not should_have_grad and tensor.requires_grad:
                    # Disable gradient
                    self.parameters[param_name] = tensor.detach()

    def get_parameter_value(self, param_id: str) -> Optional[torch.Tensor]:
        """Get parameter tensor by ID or name."""
        # First try direct lookup by name
        if param_id in self.parameters:
            return self.parameters[param_id]

        # If not found, try to find by ID and get the name
        for param_info in self.parameters_info:
            if param_info['id'] == param_id:
                param_name = param_info.get('name', param_id)
                return self.parameters.get(param_name)

        return None

    def set_parameter_value(self, param_id: str, value: Any) -> bool:
        """Set parameter value and update tensor."""
        # Find parameter info
        param_info = None
        param_name = None
        for info in self.parameters_info:
            if info['id'] == param_id or info.get('name') == param_id:
                param_info = info
                param_name = info.get('name', info['id'])
                break

        if not param_info:
            return False

        # Update value in info
        param_info['value'] = value

        # Re-instantiate this specific parameter
        param_type = param_info['type']
        param_id_for_grad = param_info['id']  # Use original ID for differentiableParams check
        requires_grad = param_id_for_grad in self.scene.node_editor.get('differentiableParams', [])

        if param_type == 'float':
            tensor_value = float(value) if value is not None else 0.0
            self.parameters[param_name] = torch.tensor(tensor_value, dtype=torch.float32, requires_grad=requires_grad)
        elif param_type == 'vector2':
            if isinstance(value, str):
                values = [float(x.strip()) for x in value.strip('()').split(',')]
            elif isinstance(value, (list, tuple)):
                values = list(value)
            else:
                values = [0.0, 0.0]
            self.parameters[param_name] = torch.tensor(values[:2], dtype=torch.float32, requires_grad=requires_grad)
        elif param_type == 'vector3':
            if isinstance(value, str):
                values = [float(x.strip()) for x in value.strip('()').split(',')]
            elif isinstance(value, (list, tuple)):
                values = list(value)
            else:
                values = [0.0, 0.0, 0.0]
            self.parameters[param_name] = torch.tensor(values[:3], dtype=torch.float32, requires_grad=requires_grad)
        elif param_type == 'vector4':
            if isinstance(value, str):
                values = [float(x.strip()) for x in value.strip('()').split(',')]
            elif isinstance(value, (list, tuple)):
                values = list(value)
            else:
                values = [0.0, 0.0, 0.0, 0.0]
            self.parameters[param_name] = torch.tensor(values[:4], dtype=torch.float32, requires_grad=requires_grad)
        elif param_type == 'bool':
            bool_value = bool(value) if value is not None else False
            self.parameters[param_name] = torch.tensor(bool_value, dtype=torch.bool, requires_grad=requires_grad)

        self.scene._emit('parameters_updated', param_id=param_id, value=value)
        return True

    def sync_parameter_values(self):
        """Sync current parameter tensor values back to parameters_info and emit update."""
        # Also sync differentiableParams list to parameters_info for compatibility
        differentiable_set = set(self.scene.node_editor.get('differentiableParams', []))

        for param_info in self.parameters_info:
            param_name = param_info.get('name', param_info['id'])
            if param_name in self.parameters:
                tensor = self.parameters[param_name]
                # Convert tensor value to appropriate format for param_info
                if tensor.dtype == torch.bool:
                    param_info['value'] = tensor.item()
                elif len(tensor.shape) == 0:
                    # Scalar
                    param_info['value'] = tensor.item()
                else:
                    # Vector
                    param_info['value'] = tensor.tolist()

                # Update differentiable status directly in param_info
                param_info['differentiable'] = tensor.requires_grad

                # Also update the differentiableParams list for backward compatibility
                if tensor.requires_grad:
                    differentiable_set.add(param_info['id'])
                else:
                    differentiable_set.discard(param_info['id'])

        # Update the differentiableParams list
        self.scene.node_editor['differentiableParams'] = list(differentiable_set)

        # Emit update event
        self.scene._emit('parameters_updated', parameters_info=self.parameters_info)

    def _check_tensor_type_match(self, tensor: torch.Tensor, param_type: str) -> bool:
        """Check if tensor matches expected parameter type."""
        if param_type == 'float':
            return tensor.numel() == 1 and tensor.dtype in [torch.float32, torch.float64]
        elif param_type == 'vector2':
            return tensor.shape == (2,) and tensor.dtype in [torch.float32, torch.float64]
        elif param_type == 'vector3':
            return tensor.shape == (3,) and tensor.dtype in [torch.float32, torch.float64]
        elif param_type == 'vector4':
            return tensor.shape == (4,) and tensor.dtype in [torch.float32, torch.float64]
        elif param_type == 'bool':
            return tensor.numel() == 1 and tensor.dtype == torch.bool
        return False