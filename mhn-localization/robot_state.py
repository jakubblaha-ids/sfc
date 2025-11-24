from .constants import MAP_WIDTH, MAP_HEIGHT


class RobotState:
    """
    Manages the robot's position, orientation, and movement.
    Pure logic class without any UI dependencies.
    """

    def __init__(
            self, x=None, y=None, angle=0, radius=5, speed=5,
            rotation_speed=7.5):
        """
        Initialize robot state.

        Args:
            x: Initial x position (defaults to center)
            y: Initial y position (defaults to center)
            angle: Initial angle in degrees (0-360)
            radius: Robot collision radius
            speed: Movement speed in pixels per frame
            rotation_speed: Rotation speed in degrees per key press
        """
        self.x = x if x is not None else MAP_WIDTH // 2
        self.y = y if y is not None else MAP_HEIGHT // 2
        self.angle = angle
        self.radius = radius
        self.speed = speed
        self.rotation_speed = rotation_speed

    def move(self, dx, dy):
        """
        Update robot position with boundary checking.

        Args:
            dx: Change in x position
            dy: Change in y position
        """
        self.x += dx
        self.y += dy

        self.x = max(self.radius, min(MAP_WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(MAP_HEIGHT - self.radius, self.y))

    def rotate(self, angle_delta):
        """
        Update robot angle.

        Args:
            angle_delta: Change in angle (degrees)
        """
        self.angle += angle_delta
        self.angle = self.angle % 360

    def set_position(self, x, y, angle=None):
        """
        Set robot position and optionally angle.

        Args:
            x: New x position
            y: New y position
            angle: New angle (optional, keeps current if None)
        """
        self.x = x
        self.y = y
        if angle is not None:
            self.angle = angle

    def get_position(self):
        """
        Get current robot position and angle.

        Returns:
            Tuple of (x, y, angle)
        """
        return (self.x, self.y, self.angle)

    def copy_state(self):
        """
        Create a copy of the current state.

        Returns:
            Dictionary with current state values
        """
        return {
            'x': self.x,
            'y': self.y,
            'angle': self.angle,
            'radius': self.radius,
            'speed': self.speed,
            'rotation_speed': self.rotation_speed
        }

    def restore_state(self, state):
        """
        Restore state from a saved state dictionary.

        Args:
            state: Dictionary with state values
        """
        self.x = state['x']
        self.y = state['y']
        self.angle = state['angle']
        self.radius = state.get('radius', self.radius)
        self.speed = state.get('speed', self.speed)
        self.rotation_speed = state.get('rotation_speed', self.rotation_speed)
