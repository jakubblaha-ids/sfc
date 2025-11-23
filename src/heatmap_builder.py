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
        threshold=0.001
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

        Returns:
            PIL Image in RGBA mode, or None if no data
        """
        if not grid_positions or not grid_confidences:
            return None

        # Get max confidence for normalization
        max_confidence = max(grid_confidences) if grid_confidences else 1.0

        # Use scipy's griddata for smooth interpolation at lower resolution
        confidence_grid = self._populate_grid_interpolated(
            grid_positions, grid_confidences, max_confidence)

        # Apply Gaussian smoothing for even smoother gradient
        # Scale sigma proportionally to the resolution
        scaled_sigma = sigma * self.resolution_scale
        smoothed_grid = gaussian_filter(confidence_grid, sigma=scaled_sigma)

        # Apply colormap and create RGBA image at low resolution
        heatmap_image = self._apply_colormap(
            smoothed_grid,
            colormap_name,
            alpha_base,
            alpha_scale,
            threshold
        )

        # Upscale the heatmap to full resolution using smooth interpolation
        if self.resolution_scale < 1.0:
            heatmap_image = heatmap_image.resize(
                (self.map_width, self.map_height),
                Image.Resampling.LANCZOS
            )

        return heatmap_image

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
        # Extract x, y coordinates and normalized confidence values
        # Scale positions to match the lower resolution grid
        points = np.array(grid_positions) * self.resolution_scale
        values = np.array(grid_confidences) / max_confidence

        # Create grid of points where we want to interpolate (at lower resolution)
        grid_y, grid_x = np.mgrid[0:self.heatmap_height, 0:self.heatmap_width]

        # Use linear interpolation (much faster than cubic and still smooth after Gaussian blur)
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
        # Use matplotlib's colormap to convert values to colors
        # 'jet': blue (low) -> cyan -> green -> yellow -> red (high)
        # Other good options: 'viridis', 'plasma', 'inferno', 'hot', 'coolwarm'
        colormap = cm.get_cmap(colormap_name)

        # Apply colormap to the smoothed grid
        # This returns RGBA values in range [0, 1]
        heatmap_rgba = colormap(smoothed_grid)

        # Convert to 8-bit RGBA (0-255)
        heatmap_colored = (heatmap_rgba * 255).astype(np.uint8)

        # Adjust alpha channel based on confidence values
        # Make it more visible with higher base alpha and stronger scaling
        mask = smoothed_grid > threshold
        # Base alpha + scaled alpha based on confidence
        alpha_values = np.clip(
            alpha_base + smoothed_grid * alpha_scale, 0, 255
        ).astype(np.uint8)
        heatmap_colored[:, :, 3] = np.where(mask, alpha_values, 0)

        # Create PIL image
        heatmap_image = Image.fromarray(heatmap_colored, mode='RGBA')
        return heatmap_image
