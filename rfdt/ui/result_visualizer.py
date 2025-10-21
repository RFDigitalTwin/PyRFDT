"""
Result visualization system for displaying plots and data in the frontend.
Supports matplotlib-like API with automatic Torch/NumPy conversion.
"""
import json
import hashlib
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class PlotData:
    """Container for a single plot's data."""
    plot_type: str  # 'line', 'bar', 'scatter', 'imshow', 'table'
    data: Dict[str, Any]
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    legend: Optional[List[str]] = None
    xticks: Optional[Dict[str, Any]] = None  # {'values': [], 'labels': []}
    yticks: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ResultVisualizer:
    """
    Matplotlib-like interface for creating visualizations in the frontend.
    Supports line plots, bar charts, scatter plots, heatmaps, and tables.
    """

    def __init__(self, server=None):
        """Initialize the visualizer with optional server reference."""
        self.server = server
        self.plots: List[PlotData] = []
        self.version = 0
        self.committed_versions: Dict[int, Dict[str, Any]] = {}  # Store full version info

    @staticmethod
    def _generate_color_from_text(text: str) -> str:
        """
        Generate a consistent color from text using hash.

        Args:
            text: Input text to generate color from

        Returns:
            Hex color string
        """
        # Create hash from text
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()

        # Use first 6 characters for color, ensure it's not too dark
        # by setting minimum values for RGB components
        r = max(80, int(hash_hex[0:2], 16))
        g = max(80, int(hash_hex[2:4], 16))
        b = max(80, int(hash_hex[4:6], 16))

        # Ensure the color is not too bright either
        r = min(200, r)
        g = min(200, g)
        b = min(200, b)

        return f'#{r:02x}{g:02x}{b:02x}'

    def _to_list(self, data: Any) -> Any:
        """Convert tensor/array data to Python lists."""
        if HAS_TORCH and isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().tolist()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif hasattr(data, '__iter__') and not isinstance(data, (str, dict)):
            # Try to convert any iterable to list
            try:
                result = []
                for item in data:
                    if HAS_TORCH and isinstance(item, torch.Tensor):
                        result.append(item.detach().cpu().numpy().tolist())
                    elif isinstance(item, np.ndarray):
                        result.append(item.tolist())
                    else:
                        result.append(item)
                return result
            except:
                return list(data)
        return data

    def _process_data(self, *args, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Process input arguments for plotting functions."""
        processed_args = []
        for arg in args:
            processed_args.append(self._to_list(arg))

        processed_kwargs = {}
        for key, value in kwargs.items():
            if key in ['label', 'labels', 'color', 'alpha', 'linewidth', 'linestyle',
                       'marker', 'markersize', 's', 'c', 'cmap']:
                # These are styling parameters we might want to keep
                if key in ['label', 'labels', 'color']:
                    processed_kwargs[key] = value
            elif key in ['xlabel', 'ylabel', 'title']:
                processed_kwargs[key] = value
            else:
                processed_kwargs[key] = self._to_list(value)

        return processed_args, processed_kwargs

    def plot(self, *args, **kwargs) -> 'ResultVisualizer':
        """
        Create a line plot similar to matplotlib.pyplot.plot.

        Examples:
            server.results.plot(x, y)
            server.results.plot(y)  # x will be range(len(y))
            server.results.plot(x, y, label="Series 1")
        """
        processed_args, processed_kwargs = self._process_data(*args, **kwargs)

        # Parse arguments matplotlib-style
        if len(processed_args) == 1:
            # Only y values provided
            y_data = processed_args[0]
            x_data = list(range(len(y_data)))
        elif len(processed_args) >= 2:
            # x and y values provided
            x_data = processed_args[0]
            y_data = processed_args[1]
        else:
            raise ValueError("plot() requires at least one argument")

        # Prepare data for frontend
        plot_data = {
            'x': x_data,
            'y': y_data,
        }

        if 'label' in processed_kwargs:
            plot_data['label'] = processed_kwargs['label']

        # Handle color parameter - supports both hex colors and named colors (red, blue, etc.)
        if 'color' in kwargs:
            plot_data['color'] = kwargs['color']

        # Create plot entry
        plot = PlotData(
            plot_type='line',
            data=plot_data,
            title=processed_kwargs.get('title'),
            xlabel=processed_kwargs.get('xlabel'),
            ylabel=processed_kwargs.get('ylabel'),
            legend=[processed_kwargs['label']] if 'label' in processed_kwargs else None
        )

        self.plots.append(plot)
        return self

    def imshow(self, data, **kwargs) -> 'ResultVisualizer':
        """
        Display an image or 2D array as a heatmap.

        Examples:
            server.results.imshow(image_array)
            server.results.imshow(tensor, title="Heatmap")
        """
        processed_data = self._to_list(data)

        # Ensure it's 2D
        if isinstance(processed_data, list) and len(processed_data) > 0:
            if not isinstance(processed_data[0], list):
                # Convert 1D to 2D
                processed_data = [processed_data]

        plot = PlotData(
            plot_type='imshow',
            data={'values': processed_data},
            title=kwargs.get('title'),
            xlabel=kwargs.get('xlabel'),
            ylabel=kwargs.get('ylabel')
        )

        self.plots.append(plot)
        return self

    def bar(self, *args, **kwargs) -> 'ResultVisualizer':
        """
        Create a bar chart.

        Examples:
            server.results.bar(x, heights)
            server.results.bar(heights)  # x will be range(len(heights))
            server.results.bar(categories, values, xlabel="Category")
        """
        processed_args, processed_kwargs = self._process_data(*args, **kwargs)

        # Parse arguments
        if len(processed_args) == 1:
            # Only heights provided
            heights = processed_args[0]
            x = list(range(len(heights)))
            labels = None
        elif len(processed_args) >= 2:
            # x (or labels) and heights provided
            x = processed_args[0]
            heights = processed_args[1]
            # If x contains strings, use them as labels
            if isinstance(x[0], str):
                labels = x
                x = list(range(len(x)))
            else:
                labels = None
        else:
            raise ValueError("bar() requires at least one argument")

        plot_data = {
            'x': x,
            'heights': heights,
        }

        if labels:
            plot_data['labels'] = labels

        plot = PlotData(
            plot_type='bar',
            data=plot_data,
            title=kwargs.get('title'),
            xlabel=kwargs.get('xlabel'),
            ylabel=kwargs.get('ylabel')
        )

        # Handle tick labels if provided
        if 'xticks' in kwargs:
            plot.xticks = kwargs['xticks']
        elif labels:
            plot.xticks = {'values': x, 'labels': labels}

        self.plots.append(plot)
        return self

    def scatter(self, x, y, **kwargs) -> 'ResultVisualizer':
        """
        Create a scatter plot.

        Examples:
            server.results.scatter(x, y)
            server.results.scatter(x, y, label="Data points")
        """
        processed_x = self._to_list(x)
        processed_y = self._to_list(y)

        plot_data = {
            'x': processed_x,
            'y': processed_y,
        }

        if 'label' in kwargs:
            plot_data['label'] = kwargs['label']

        # Handle size parameter (matplotlib uses 's')
        if 's' in kwargs:
            plot_data['size'] = self._to_list(kwargs['s'])
        elif 'size' in kwargs:
            plot_data['size'] = self._to_list(kwargs['size'])

        # Handle color parameter (matplotlib uses 'c')
        if 'c' in kwargs:
            plot_data['color'] = self._to_list(kwargs['c'])
        elif 'color' in kwargs:
            plot_data['color'] = self._to_list(kwargs['color'])

        plot = PlotData(
            plot_type='scatter',
            data=plot_data,
            title=kwargs.get('title'),
            xlabel=kwargs.get('xlabel'),
            ylabel=kwargs.get('ylabel'),
            legend=[kwargs['label']] if 'label' in kwargs else None
        )

        self.plots.append(plot)
        return self

    def table(self, data, **kwargs) -> 'ResultVisualizer':
        """
        Display data in a table format.

        Examples:
            server.results.table([[1,2,3], [4,5,6]], columns=['A', 'B', 'C'])
            server.results.table({'col1': [1,2,3], 'col2': [4,5,6]})
        """
        processed_data = self._to_list(data)

        # Handle different data formats
        if isinstance(data, dict):
            # Dictionary format: {'col1': [...], 'col2': [...]}
            columns = list(data.keys())
            # Convert to row format
            rows = []
            max_len = max(len(v) for v in data.values())
            for i in range(max_len):
                row = []
                for col in columns:
                    col_data = self._to_list(data[col])
                    if i < len(col_data):
                        row.append(col_data[i])
                    else:
                        row.append(None)
                rows.append(row)
            processed_data = rows
        else:
            # List format: [[row1], [row2], ...]
            columns = kwargs.get('columns')
            if columns is None and len(processed_data) > 0:
                # Auto-generate column names
                if isinstance(processed_data[0], (list, tuple)):
                    columns = [f'Col{i+1}' for i in range(len(processed_data[0]))]
                else:
                    columns = ['Value']

        plot = PlotData(
            plot_type='table',
            data={
                'rows': processed_data,
                'columns': columns
            },
            title=kwargs.get('title')
        )

        self.plots.append(plot)
        return self

    def clear(self) -> 'ResultVisualizer':
        """Clear all pending plots."""
        self.plots.clear()
        return self

    def commit(self, message: Optional[str] = None, tag: Optional[str] = None,
              tag_color: Optional[str] = None, clear_after: bool = True) -> None:
        """
        Commit the current plots to be displayed in the frontend.

        Args:
            message: Optional custom message for this version (default: "v{version}")
            tag: Optional tag to label this version
            tag_color: Optional color for the tag (default: auto-generated from tag text)
            clear_after: If True, clear the plots after committing (default: True)

        Examples:
            server.results.commit()  # Simple commit with default "v1" message
            server.results.commit(message="Initial Results")  # Custom message
            server.results.commit(tag="baseline")  # Add a tag
            server.results.commit(message="Experiment A", tag="test", tag_color="#ff6600")
        """
        if not self.plots:
            print("[ResultVisualizer] No plots to commit")
            return

        # Increment version
        self.version += 1

        # Generate default message if not provided
        if message is None:
            message = f"v{self.version}"

        # Generate tag color if tag is provided but color is not
        if tag and not tag_color:
            tag_color = self._generate_color_from_text(tag)

        # Prepare data for frontend
        plots_data = []
        for plot in self.plots:
            plot_dict = {
                'type': plot.plot_type,
                'data': plot.data,
                'timestamp': plot.timestamp,
            }

            # Add optional fields
            if plot.title:
                plot_dict['title'] = plot.title
            if plot.xlabel:
                plot_dict['xlabel'] = plot.xlabel
            if plot.ylabel:
                plot_dict['ylabel'] = plot.ylabel
            if plot.legend:
                plot_dict['legend'] = plot.legend
            if plot.xticks:
                plot_dict['xticks'] = plot.xticks
            if plot.yticks:
                plot_dict['yticks'] = plot.yticks

            plots_data.append(plot_dict)

        # Store this version with full info
        timestamp = datetime.now().isoformat()
        version_data = {
            'version': self.version,
            'message': message,
            'plots': plots_data,
            'timestamp': timestamp
        }

        # Add tag info if provided
        if tag:
            version_data['tag'] = tag
            version_data['tag_color'] = tag_color

        self.committed_versions[self.version] = version_data

        # Send to frontend via WebSocket - only send current version, not history
        if self.server:
            ws_message = {
                'type': 'results_update',
                'version': self.version,
                'message': message,
                'plots': plots_data,
                'timestamp': timestamp
            }

            # Add tag info to WebSocket message
            if tag:
                ws_message['tag'] = tag
                ws_message['tag_color'] = tag_color

            # Broadcast to all connected clients
            if hasattr(self.server, '_safe_broadcast'):
                self.server._safe_broadcast(ws_message)
                if self.server.verbose:
                    print(f"[ResultVisualizer] Committed version {self.version} ({message}) with {len(self.plots)} plots")
            else:
                print("[ResultVisualizer] Warning: Server doesn't have _safe_broadcast method")
        else:
            print("[ResultVisualizer] Warning: No server connected, plots not sent to frontend")

        # Clear plots if requested
        if clear_after:
            self.clear()

    def get_version(self, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get plots from a specific version.

        Args:
            version: Version number to retrieve. If None, returns current pending plots.

        Returns:
            Dict with version info, or None if version doesn't exist.
        """
        if version is None:
            return {'version': 0, 'plots': list(self.plots), 'timestamp': datetime.now().isoformat()}
        return self.committed_versions.get(version)

    def get_all_versions(self) -> List[Dict[str, Any]]:
        """
        Get all committed versions.

        Returns:
            List of all version dictionaries.
        """
        return list(self.committed_versions.values())

    def delete_version(self, version: int) -> bool:
        """
        Delete a specific version from history.

        Args:
            version: Version number to delete.

        Returns:
            True if version was deleted, False if version doesn't exist.
        """
        if version in self.committed_versions:
            del self.committed_versions[version]
            if self.server and self.server.verbose:
                print(f"[ResultVisualizer] Deleted version {version}")
            return True
        return False

    def set_title(self, title: str) -> 'ResultVisualizer':
        """Set title for the last added plot."""
        if self.plots:
            self.plots[-1].title = title
        return self

    def set_xlabel(self, label: str) -> 'ResultVisualizer':
        """Set x-axis label for the last added plot."""
        if self.plots:
            self.plots[-1].xlabel = label
        return self

    def set_ylabel(self, label: str) -> 'ResultVisualizer':
        """Set y-axis label for the last added plot."""
        if self.plots:
            self.plots[-1].ylabel = label
        return self

    def set_xticks(self, values: List, labels: Optional[List[str]] = None) -> 'ResultVisualizer':
        """
        Set x-axis tick positions and labels.

        Args:
            values: Tick positions
            labels: Tick labels (optional, defaults to values)
        """
        if self.plots:
            values = self._to_list(values)
            if labels is None:
                labels = [str(v) for v in values]
            self.plots[-1].xticks = {'values': values, 'labels': labels}
        return self

    def set_yticks(self, values: List, labels: Optional[List[str]] = None) -> 'ResultVisualizer':
        """
        Set y-axis tick positions and labels.

        Args:
            values: Tick positions
            labels: Tick labels (optional, defaults to values)
        """
        if self.plots:
            values = self._to_list(values)
            if labels is None:
                labels = [str(v) for v in values]
            self.plots[-1].yticks = {'values': values, 'labels': labels}
        return self