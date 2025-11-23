import math
from PIL import Image, ImageFilter
from constants import MAP_WIDTH, MAP_HEIGHT


class CameraSimulator:
    """
    Simulates a robot's camera view using raycasting.
    Pure logic class without any UI dependencies.
    """

    def __init__(self, cone_angle=90, cone_length=40, camera_samples=100,
                 blur_radius=0.0, visibility_index=0.1):
        """
        Initialize camera simulator.

        Args:
            cone_angle: Field of view in degrees
            cone_length: Maximum viewing distance
            camera_samples: Number of pixel samples in the 1D strip
            blur_radius: Gaussian blur radius for the camera view
            visibility_index: Distance opacity factor (0.0 - 1.0)
        """
        self.cone_angle = cone_angle
        self.cone_length = cone_length
        self.camera_samples = camera_samples
        self.blur_radius = blur_radius
        self.visibility_index = visibility_index

    def capture_view(self, robot_x, robot_y, robot_angle, map_image):
        """
        Capture a 1D strip of pixels from what the robot sees.
        Samples pixels along rays cast from the robot's viewing direction.
        Walls appear less opaque (more white) the further they are from the robot.

        Args:
            robot_x: Robot x position
            robot_y: Robot y position
            robot_angle: Robot angle in degrees
            map_image: PIL Image of the map

        Returns:
            PIL Image (1D strip of camera samples)
        """
        half_cone = self.cone_angle / 2
        pixels = map_image.load()

        camera_strip = []
        max_distance = self.cone_length

        for i in range(self.camera_samples):
            angle_offset = -half_cone + (self.cone_angle * i /
                                         (self.camera_samples - 1))
            current_angle = math.radians(robot_angle + angle_offset)

            dx = math.cos(current_angle)
            dy = math.sin(current_angle)

            distance = self.cast_ray(robot_x, robot_y, dx, dy, map_image)

            end_x = robot_x + distance * dx
            end_y = robot_y + distance * dy

            pixel_x = int(round(max(0, min(MAP_WIDTH - 1, end_x))))
            pixel_y = int(round(max(0, min(MAP_HEIGHT - 1, end_y))))

            color = pixels[pixel_x, pixel_y]

            opacity = math.exp(-distance / max_distance * self.visibility_index)

            blended_color = (
                int(color[0] * opacity + 255 * (1 - opacity)),
                int(color[1] * opacity + 255 * (1 - opacity)),
                int(color[2] * opacity + 255 * (1 - opacity))
            )

            camera_strip.append(blended_color)

        camera_view = Image.new('RGB', (self.camera_samples, 1))
        camera_view.putdata(camera_strip)

        if self.blur_radius > 0:
            temp_height = 10
            temp_view = camera_view.resize(
                (self.camera_samples, temp_height),
                Image.Resampling.NEAREST
            )
            temp_view = temp_view.filter(ImageFilter.GaussianBlur(
                radius=self.blur_radius))
            camera_view = temp_view.resize(
                (self.camera_samples, 1),
                Image.Resampling.BILINEAR
            )

        return camera_view

    def cast_ray(self, x, y, dx, dy, map_image):
        """
        Cast a ray from (x, y) in direction (dx, dy) until hitting a wall or map edge.
        Returns the distance to the first obstacle.

        Args:
            x: Starting x position
            y: Starting y position
            dx: Direction x component (normalized)
            dy: Direction y component (normalized)
            map_image: PIL Image of the map

        Returns:
            Distance to first obstacle
        """
        max_distance = self._get_distance_to_edge(x, y, dx, dy)
        step_size = 5.0
        pixels = map_image.load()

        distance = 0
        while distance < max_distance:
            distance += step_size

            current_x = x + distance * dx
            current_y = y + distance * dy

            pixel_x = int(round(current_x))
            pixel_y = int(round(current_y))

            if pixel_x < 0 or pixel_x >= MAP_WIDTH or pixel_y < 0 or pixel_y >= MAP_HEIGHT:
                return distance

            pixel_color = pixels[pixel_x, pixel_y]

            if pixel_color[0] < 255 or pixel_color[1] < 255 or pixel_color[2] < 255:
                return distance

        return max_distance

    def _get_distance_to_edge(self, x, y, dx, dy):
        """
        Calculate the distance from (x, y) to the map edge in direction (dx, dy).

        Args:
            x: Starting x position
            y: Starting y position
            dx: Direction x component
            dy: Direction y component

        Returns:
            Distance to map edge
        """
        distances = []

        if dx > 0:
            t = (MAP_WIDTH - x) / dx
            distances.append(t)
        elif dx < 0:
            t = -x / dx
            distances.append(t)

        if dy > 0:
            t = (MAP_HEIGHT - y) / dy
            distances.append(t)
        elif dy < 0:
            t = -y / dy
            distances.append(t)

        return min(distances) if distances else 0

    def get_viewing_cone_points(
            self, robot_x, robot_y, robot_angle, map_image):
        """
        Get the polygon points for drawing the viewing cone.
        Used for visualization.

        Args:
            robot_x: Robot x position
            robot_y: Robot y position
            robot_angle: Robot angle in degrees
            map_image: PIL Image of the map

        Returns:
            List of (x, y) tuples defining the cone polygon
        """
        half_cone = self.cone_angle / 2
        points = [(robot_x, robot_y)]

        num_points = 50
        for i in range(num_points + 1):
            angle_offset = -half_cone + (self.cone_angle * i / num_points)
            current_angle = math.radians(robot_angle + angle_offset)

            dx = math.cos(current_angle)
            dy = math.sin(current_angle)

            distance = self.cast_ray(robot_x, robot_y, dx, dy, map_image)

            x = robot_x + distance * dx
            y = robot_y + distance * dy
            points.append((x, y))

        return points
