import numpy as np
from PIL import Image
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata


class HeatmapBuilder:
    """
    Builds a confidence heatmap from a grid of evaluated positions.
    Creates a smooth 2D color gradient visualization using matplotlib colormaps.
    """

    def __init__(self, map_width, map_height, grid_stride,
                 resolution_scale=0.175):
        """
        Initialize the heatmap builder.

        Args:
            map_width: Width of the map in pixels
            map_height: Height of the map in pixels
            grid_stride: Stride between grid points (for nearest-neighbor interpolation)
            resolution_scale: Scale factor for heatmap resolution (default: 0.25 = 1/4 resolution)
                             Lower values = faster computation but less detail
        """
        self.map_width = map_width
        self.map_height = map_height
        self.grid_stride = grid_stride
        self.resolution_scale = resolution_scale
        self.heatmap_width = int(map_width * resolution_scale)
        self.heatmap_height = int(map_height * resolution_scale)

    def build_heatmap(
        self,
        grid_positions,
        grid_confidences,
        colormap_name='jet',
        sigma=10.0,
        alpha_base=100,
        alpha_scale=155,
        threshold=0.001,
        invert_values=False
    ):
        """
        Build a heatmap image from grid positions and confidence values.

        Args:
            grid_positions: List of (x, y) tuples for evaluated positions
            grid_confidences: List of confidence values (0-1) for each position
            colormap_name: Name of matplotlib colormap to use (default: 'jet')
            sigma: Gaussian smoothing sigma value (default: 10.0)
            alpha_base: Base alpha value for transparency (default: 100)
            alpha_scale: Additional alpha scaling factor (default: 155)
            threshold: Minimum confidence threshold for visibility (default: 0.001)
            invert_values: If True, invert the values (for energy: low = good)

        Returns:
            PIL Image in RGBA mode, or None if no data
        """
        if not grid_positions or not grid_confidences:
            return None

        if invert_values:
            min_val = min(grid_confidences)
            max_val = max(grid_confidences)
            if max_val != min_val:
                grid_confidences = [(max_val - val) / (max_val - min_val)
                                    for val in grid_confidences]
            else:
                grid_confidences = [0.5 for _ in grid_confidences]
            max_confidence = 1.0
        else:
            max_confidence = max(grid_confidences) if grid_confidences else 1.0

        confidence_grid = self._populate_grid_interpolated(
            grid_positions, grid_confidences, max_confidence)

        scaled_sigma = sigma * self.resolution_scale
        smoothed_grid = gaussian_filter(confidence_grid, sigma=scaled_sigma)

        heatmap_image = self._apply_colormap(
            smoothed_grid,
            colormap_name,
            alpha_base,
            alpha_scale,
            threshold
        )

        if self.resolution_scale < 1.0:
            heatmap_image = heatmap_image.resize(
                (self.map_width, self.map_height),
                Image.Resampling.LANCZOS
            )

        return heatmap_image

    def build_averaged_heatmap(
        self,
        grid_positions_by_angle,
        grid_confidences_by_angle,
        colormap_name='jet',
        sigma=10.0,
        alpha_base=100,
        alpha_scale=155,
        threshold=0.001,
        invert_values=False
    ):
        """
        Build an averaged heatmap across all angles.

        Args:
            grid_positions_by_angle: Dict mapping angles to lists of (x, y) tuples
            grid_confidences_by_angle: Dict mapping angles to lists of confidence values
            colormap_name: Name of matplotlib colormap to use (default: 'jet')
            sigma: Gaussian smoothing sigma value (default: 10.0)
            alpha_base: Base alpha value for transparency (default: 100)
            alpha_scale: Additional alpha scaling factor (default: 155)
            threshold: Minimum confidence threshold for visibility (default: 0.001)
            invert_values: If True, invert the values (for energy: low = good)

        Returns:
            PIL Image in RGBA mode, or None if no data
        """
        averaged_positions, averaged_confidences = self._compute_averaged_data(
            grid_positions_by_angle, grid_confidences_by_angle)

        if not averaged_positions or not averaged_confidences:
            return None

        return self.build_heatmap(
            averaged_positions,
            averaged_confidences,
            colormap_name,
            sigma,
            alpha_base,
            alpha_scale,
            threshold,
            invert_values
        )

    def _compute_averaged_data(
            self, grid_positions_by_angle, grid_confidences_by_angle):
        """
        Compute averaged confidence values across all angles for each position.

        Args:
            grid_positions_by_angle: Dict mapping angles to lists of (x, y) tuples
            grid_confidences_by_angle: Dict mapping angles to lists of confidence values

        Returns:
            Tuple of (positions, averaged_confidences) where positions is a list
            of (x, y) tuples and averaged_confidences is a list of averaged values
        """
        position_confidences = {}

        for angle in grid_positions_by_angle.keys():
            positions = grid_positions_by_angle.get(angle, [])
            confidences = grid_confidences_by_angle.get(angle, [])

            for pos, conf in zip(positions, confidences):
                if pos not in position_confidences:
                    position_confidences[pos] = []
                position_confidences[pos].append(conf)

        averaged_positions = []
        averaged_confidences = []

        for pos, conf_list in position_confidences.items():
            averaged_positions.append(pos)
            averaged_confidences.append(np.mean(conf_list))

        return averaged_positions, averaged_confidences

    def _populate_grid_interpolated(
            self, grid_positions, grid_confidences, max_confidence):
        """
        Populate the confidence grid using smooth interpolation at reduced resolution.

        Args:
            grid_positions: List of (x, y) tuples
            grid_confidences: List of confidence values
            max_confidence: Maximum confidence for normalization

        Returns:
            2D numpy array with interpolated values
        """
        points = np.array(grid_positions) * self.resolution_scale
        values = np.array(grid_confidences) / max_confidence

        grid_y, grid_x = np.mgrid[0:self.heatmap_height, 0:self.heatmap_width]

        confidence_grid = griddata(
            points, values, (grid_x, grid_y),
            method='linear',
            fill_value=0.0
        )

        return confidence_grid

    def _apply_colormap(
        self,
        smoothed_grid,
        colormap_name,
        alpha_base,
        alpha_scale,
        threshold
    ):
        """
        Apply matplotlib colormap to the smoothed grid and create RGBA image.

        Args:
            smoothed_grid: 2D numpy array of confidence values
            colormap_name: Name of matplotlib colormap
            alpha_base: Base alpha value
            alpha_scale: Additional alpha scaling
            threshold: Minimum value threshold

        Returns:
            PIL Image in RGBA mode
        """
        colormap = cm.get_cmap(colormap_name)

        heatmap_rgba = colormap(smoothed_grid)

        heatmap_colored = (heatmap_rgba * 255).astype(np.uint8)

        if threshold is not None:
            mask = smoothed_grid > threshold
        else:
            mask = smoothed_grid > 0.0
        alpha_values = np.clip(
            alpha_base + smoothed_grid * alpha_scale, 0, 255
        ).astype(np.uint8)
        heatmap_colored[:, :, 3] = np.where(mask, alpha_values, 0)

        heatmap_image = Image.fromarray(heatmap_colored, mode='RGBA')
        return heatmap_image
