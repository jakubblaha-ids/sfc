from .constants import MAP_WIDTH, MAP_HEIGHT, SAMPLE_STRIDE, SAMPLE_ROTATIONS


class SamplingEngine:
    """
    Generates sample positions for training the localization system.
    Creates a grid of positions with multiple rotations at each position.
    Pure logic class without any UI dependencies.
    """

    def __init__(self, stride=SAMPLE_STRIDE, num_rotations=SAMPLE_ROTATIONS):
        """
        Initialize sampling engine.

        Args:
            stride: Distance between sample points
            num_rotations: Number of rotation angles to sample at each position
        """
        self.stride = stride
        self.num_rotations = num_rotations

    def _generate_grid(self, stride):
        """
        Generate a grid of (x, y) positions based on the stride.

        Args:
            stride: Grid spacing

        Returns:
            List of (x, y) tuples
        """
        half_stride = max(0, stride // 2)
        x_positions = list(range(half_stride, MAP_WIDTH, stride))
        y_positions = list(range(half_stride, MAP_HEIGHT, stride))
        
        grid = []
        for x in x_positions:
            for y in y_positions:
                grid.append((x, y))
        return grid

    def _generate_angles(self):
        """Generate list of rotation angles."""
        return [i * (360 / self.num_rotations) for i in range(self.num_rotations)]

    def generate_sample_positions(self):
        """
        Generate a grid of sample positions with rotations.

        Returns:
            List of (x, y, angle) tuples representing sample positions
        """
        grid = self._generate_grid(self.stride)
        angles = self._generate_angles()
        
        positions = []
        for x, y in grid:
            for angle in angles:
                positions.append((x, y, angle))

        return positions

    def generate_test_positions(self, num_positions=None):
        """
        Generate test positions for confidence evaluation.
        Uses a denser grid than training samples.

        Args:
            num_positions: Maximum number of positions (None for all)

        Returns:
            List of (x, y, angle) tuples representing test positions
        """
        grid_stride = max(1, self.stride // 2)
        grid = self._generate_grid(grid_stride)
        angles = self._generate_angles()

        all_positions = []
        for angle in angles:
            for x, y in grid:
                all_positions.append((x, y, angle))

        if num_positions is None:
            return all_positions
        else:
            return all_positions[:max(0, int(num_positions))]

    def generate_heatmap_grid_positions(self):
        """
        Generate positions for heatmap computation.
        Uses a denser grid for smoother heatmap visualization.

        Returns:
            Tuple of (positions_by_angle, angles) where:
                - positions_by_angle: Dict mapping angle to list of (x, y) tuples
                - angles: List of angles used
        """
        grid_stride = self.stride // 2
        grid = self._generate_grid(grid_stride)
        angles = self._generate_angles()

        positions_by_angle = {}
        for angle in angles:
            positions_by_angle[angle] = list(grid)

        return positions_by_angle, angles

    def get_total_samples(self):
        """
        Get the total number of samples that will be generated.

        Returns:
            Total number of (x, y, angle) combinations
        """
        grid = self._generate_grid(self.stride)
        return len(grid) * self.num_rotations
