from constants import MAP_WIDTH, MAP_HEIGHT, SAMPLE_STRIDE, SAMPLE_ROTATIONS


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

    def generate_sample_positions(self):
        """
        Generate a grid of sample positions with rotations.

        Returns:
            List of (x, y, angle) tuples representing sample positions
        """
        positions = []

        half_stride = self.stride // 2
        x_positions = list(range(half_stride, MAP_WIDTH, self.stride))
        y_positions = list(range(half_stride, MAP_HEIGHT, self.stride))

        angles = [i * (360 / self.num_rotations)
                  for i in range(self.num_rotations)]

        for x in x_positions:
            for y in y_positions:
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
        half_stride = max(0, grid_stride // 2)

        x_positions = list(range(half_stride, MAP_WIDTH, grid_stride))
        y_positions = list(range(half_stride, MAP_HEIGHT, grid_stride))
        test_angles = [i * (360 / self.num_rotations)
                       for i in range(self.num_rotations)]

        all_positions = []
        for angle in test_angles:
            for x in x_positions:
                for y in y_positions:
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
        half_stride = grid_stride // 2

        x_positions = list(range(half_stride, MAP_WIDTH, grid_stride))
        y_positions = list(range(half_stride, MAP_HEIGHT, grid_stride))

        test_angles = [i * (360 / self.num_rotations)
                       for i in range(self.num_rotations)]

        positions_by_angle = {}
        for angle in test_angles:
            positions_for_angle = []
            for x in x_positions:
                for y in y_positions:
                    positions_for_angle.append((x, y))
            positions_by_angle[angle] = positions_for_angle

        return positions_by_angle, test_angles

    def get_total_samples(self):
        """
        Get the total number of samples that will be generated.

        Returns:
            Total number of (x, y, angle) combinations
        """
        half_stride = self.stride // 2
        x_count = len(range(half_stride, MAP_WIDTH, self.stride))
        y_count = len(range(half_stride, MAP_HEIGHT, self.stride))

        return x_count * y_count * self.num_rotations
