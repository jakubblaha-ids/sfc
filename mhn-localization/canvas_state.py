from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from PIL import Image
import numpy as np


@dataclass
class CanvasState:
    """
    Encapsulates all state needed to render the canvas.
    This allows complete separation of rendering logic from application state.
    """

    # Map images
    current_map_image: Image.Image
    map_with_noise: Optional[Image.Image] = None

    # Robot state
    robot_x: float = 0.0
    robot_y: float = 0.0
    robot_angle: float = 0.0
    robot_radius: float = 10.0

    # Estimated/localized position
    estimated_x: Optional[float] = None
    estimated_y: Optional[float] = None
    estimated_angle: Optional[float] = None

    # Noise visualization
    apply_noise: bool = False
    noise_circles: List[Tuple[float, float, float]] = field(default_factory=list)

    # Sample visualization
    sample_positions: List[Tuple[float, float, float]] = field(default_factory=list)
    sample_similarities: Optional[np.ndarray] = None

    # Localization results
    top_k_matches: List[dict] = field(default_factory=list)
    retrieved_sample_idx: Optional[int] = None

    # Test positions for confidence computation
    show_test_positions: bool = False
    test_positions: List[Tuple[float, float]] = field(default_factory=list)

    # Confidence heatmap
    show_confidence_heatmap: bool = False
    confidence_heatmap_image: Optional[Image.Image] = None

    # Energy heatmap
    show_energy_heatmap: bool = False
    energy_heatmap_image: Optional[Image.Image] = None

    average_heatmap: bool = False
    heatmap_data: Optional[dict] = None

    # Hovered sample visualization
    hovered_sample_idx: Optional[int] = None
    hovered_sample_info: Optional[dict] = None

    # Convergence visualization
    convergence_visualization_strips: List[Image.Image] = field(default_factory=list)
    converged_position: Optional[Tuple[float, float, float]] = None

    # Canvas scaling
    map_scale_factor: float = 1.0
    map_offset_x: float = 0.0
    map_offset_y: float = 0.0

    # Current camera view for cone visualization
    current_camera_angle: float = 0.0
    camera_fov: float = 90.0
    camera_cone_length: float = 40.0

    def image_to_canvas_coords(self, img_x: float, img_y: float) -> Tuple[float, float]:
        """Convert image coordinates to canvas coordinates"""
        canvas_x = img_x * self.map_scale_factor + self.map_offset_x
        canvas_y = img_y * self.map_scale_factor + self.map_offset_y
        return canvas_x, canvas_y

    def canvas_to_image_coords(self, canvas_x: float, canvas_y: float) -> Tuple[float, float]:
        """Convert canvas coordinates to image coordinates"""
        img_x = (canvas_x - self.map_offset_x) / self.map_scale_factor
        img_y = (canvas_y - self.map_offset_y) / self.map_scale_factor
        return img_x, img_y
