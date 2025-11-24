from .constants import SAMPLE_STRIDE, HEATMAP_RESOLUTION_SCALE
from .heatmap_builder import HeatmapBuilder


class ConfidenceAnalyzer:
    """
    Analyzes localization confidence across the map.
    Computes average confidence, heatmaps, and evaluation metrics.
    Pure logic class without any UI dependencies.
    """

    def __init__(self, map_width, map_height):
        """
        Initialize confidence analyzer.

        Args:
            map_width: Width of the map in pixels
            map_height: Height of the map in pixels
        """
        self.map_width = map_width
        self.map_height = map_height

        self.average_confidence = None
        self.confidence_num_tests = None
        self.test_positions = []

        self.heatmap_grid_positions_by_angle = {}
        self.heatmap_grid_confidences_by_angle = {}
        self.heatmap_computed = False
        self.heatmap_angles = []

    def compute_average_confidence(
            self, test_positions, localization_callback):
        """
        Compute average retrieval confidence over a set of test positions.

        Args:
            test_positions: List of (x, y, angle) tuples to evaluate
            localization_callback: Function(x, y, angle) that returns localization result

        Returns:
            Dictionary with 'average_confidence', 'num_tests', 'test_positions' (x, y only)
        """
        self.test_positions = []
        total_confidence = 0.0
        valid_tests = 0

        for x, y, angle in test_positions:
            result = localization_callback(x, y, angle)

            if result is not None and 'confidence' in result:
                confidence = float(result['confidence'])
                total_confidence += confidence
                valid_tests += 1

            self.test_positions.append((x, y))

        if valid_tests > 0:
            self.average_confidence = total_confidence / valid_tests
            self.confidence_num_tests = valid_tests

            return {
                'average_confidence': self.average_confidence,
                'num_tests': self.confidence_num_tests,
                'test_positions': self.test_positions
            }
        else:
            return None

    def compute_confidence_heatmap(
            self, grid_positions_by_angle, localization_callback):
        """
        Pre-compute confidence values at a dense grid of positions for each direction.
        Creates separate heatmaps for each angle.

        Args:
            grid_positions_by_angle: Dict mapping angle to list of (x, y) tuples
            localization_callback: Function(x, y, angle) that returns localization result

        Returns:
            Dictionary with heatmap data
        """
        self.heatmap_grid_positions_by_angle = {}
        self.heatmap_grid_confidences_by_angle = {}
        self.heatmap_angles = []

        total_evaluations = 0
        for angle, positions in grid_positions_by_angle.items():
            total_evaluations += len(positions)

        evaluated_count = 0

        for angle, positions in grid_positions_by_angle.items():
            positions_for_angle = []
            confidences_for_angle = []

            for x, y in positions:
                result = localization_callback(x, y, angle)

                confidence = result['confidence'] if result else 0.0
                positions_for_angle.append((x, y))
                confidences_for_angle.append(confidence)

                evaluated_count += 1

            self.heatmap_grid_positions_by_angle[angle] = positions_for_angle
            self.heatmap_grid_confidences_by_angle[angle] = confidences_for_angle
            self.heatmap_angles.append(angle)

        self.heatmap_computed = True

        return {
            'total_evaluations': total_evaluations,
            'angles': self.heatmap_angles
        }

    def build_heatmap_image(self, current_angle, average_all_angles=False,
                            colormap='jet', sigma=20.0):
        """
        Build a heatmap image for visualization.

        Args:
            current_angle: Current robot angle (used to find closest precomputed angle)
            average_all_angles: If True, average across all angles
            colormap: Matplotlib colormap name
            sigma: Gaussian blur sigma for smoothing

        Returns:
            PIL Image in RGBA mode, or None if no heatmap computed
        """
        if not self.heatmap_computed or not self.heatmap_angles:
            return None

        grid_stride = SAMPLE_STRIDE // 2
        builder = HeatmapBuilder(
            map_width=self.map_width,
            map_height=self.map_height,
            grid_stride=grid_stride,
            resolution_scale=HEATMAP_RESOLUTION_SCALE
        )

        if average_all_angles:
            heatmap_image = builder.build_averaged_heatmap(
                grid_positions_by_angle=self.heatmap_grid_positions_by_angle,
                grid_confidences_by_angle=self.heatmap_grid_confidences_by_angle,
                colormap_name=colormap,
                sigma=sigma,
                alpha_base=100,
                alpha_scale=155,
                threshold=0.001
            )
        else:
            closest_angle = self._find_closest_angle(current_angle)

            if closest_angle is None:
                return None

            grid_positions = self.heatmap_grid_positions_by_angle.get(
                closest_angle,
                [])
            grid_confidences = self.heatmap_grid_confidences_by_angle.get(
                closest_angle,
                [])

            if not grid_positions or not grid_confidences:
                return None

            heatmap_image = builder.build_heatmap(
                grid_positions=grid_positions,
                grid_confidences=grid_confidences,
                colormap_name=colormap,
                sigma=sigma,
                alpha_base=100,
                alpha_scale=155,
                threshold=0.001
            )

        return heatmap_image

    def _find_closest_angle(self, current_angle):
        """
        Find the closest precomputed heatmap angle to the current angle.

        Args:
            current_angle: Current angle in degrees (0-360)

        Returns:
            The closest angle from self.heatmap_angles, or None if no angles available
        """
        if not self.heatmap_angles:
            return None

        current_angle = current_angle % 360

        min_diff = float('inf')
        closest = None

        for angle in self.heatmap_angles:
            diff = abs(angle - current_angle)
            if diff > 180:
                diff = 360 - diff

            if diff < min_diff:
                min_diff = diff
                closest = angle

        return closest

    def reset(self):
        """Reset all computed data."""
        self.average_confidence = None
        self.confidence_num_tests = None
        self.test_positions = []
        self.heatmap_grid_positions_by_angle = {}
        self.heatmap_grid_confidences_by_angle = {}
        self.heatmap_computed = False
        self.heatmap_angles = []
