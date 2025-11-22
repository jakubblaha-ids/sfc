import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from constants import *
from PIL import Image, ImageTk
from editor import MapEditor
from config import ConfigManager
import math
import os


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Robot Localization via Modern Hopfield Networks")
        self.root.geometry(f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}")

        # Configuration manager
        self.config = ConfigManager()

        # Map Data
        self.current_map_image = Image.new(
            "RGB", (MAP_WIDTH, MAP_HEIGHT), "white")
        self.tk_map_image = None

        # Robot state
        self.robot_x = MAP_WIDTH // 2
        self.robot_y = MAP_HEIGHT // 2
        self.robot_radius = 5
        self.robot_speed = 5
        # Angle in degrees (0 = right, 90 = down, 180 = left, 270 = up)
        self.robot_angle = 0
        self.robot_rotation_speed = 7.5  # Degrees per key press
        self.cone_length = 40  # Length of the viewing cone
        self.cone_angle = 90  # Cone angle in degrees

        # Camera parameters
        self.camera_samples = 100  # Number of pixel samples in the 1D strip
        self.current_camera_view = None  # Store current camera strip

        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.create_layout()
        self.create_toolbar()
        self.create_left_panel()
        self.create_right_panel()

        # Load last used map if available
        self.load_last_map()

        # Bind keyboard controls
        self.root.bind(
            '<KeyPress-w>', lambda e: self.move_robot(0, -self.robot_speed))
        self.root.bind(
            '<KeyPress-a>', lambda e: self.move_robot(-self.robot_speed, 0))
        self.root.bind(
            '<KeyPress-s>', lambda e: self.move_robot(0, self.robot_speed))
        self.root.bind(
            '<KeyPress-d>', lambda e: self.move_robot(self.robot_speed, 0))
        self.root.bind('<KeyPress-j>',
                       lambda e: self.rotate_robot(-self.robot_rotation_speed))
        self.root.bind('<KeyPress-l>',
                       lambda e: self.rotate_robot(self.robot_rotation_speed))

    def create_layout(self):
        # Main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Toolbar area
        self.toolbar_frame = tk.Frame(
            self.main_container, height=TOOLBAR_HEIGHT)
        self.toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar_frame.pack_propagate(False)  # Prevent shrinking

        # Content area
        self.content_frame = ttk.Frame(self.main_container)
        self.content_frame.pack(
            side=tk.TOP, fill=tk.BOTH, expand=True, padx=PANEL_PADDING,
            pady=PANEL_PADDING)

    def create_toolbar(self):
        buttons = {
            "Edit Map": self.open_map_editor,
            "Import Map": self.import_map,
            "Export Map": self.export_map,
            "Auto Sample": lambda: print("Auto Sample")
        }

        for btn_text, command in buttons.items():
            btn = tk.Button(
                self.toolbar_frame,
                text=btn_text,
                command=command
            )
            btn.pack(side=tk.LEFT, padx=5, pady=5)

    def create_left_panel(self):
        # Left Panel Container
        self.left_panel = tk.Frame(
            self.content_frame, width=MAP_WIDTH,
            height=MAP_HEIGHT)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.left_panel.pack_propagate(False)

        # Label
        lbl = tk.Label(
            self.left_panel, text="World View (God Mode)")
        lbl.pack(side=tk.TOP, pady=5)

        # Canvas for Map
        self.map_canvas = tk.Canvas(
            self.left_panel, highlightthickness=1)
        self.map_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Placeholder text
        self.map_canvas.create_text(
            MAP_WIDTH // 2, MAP_HEIGHT // 2, text="Map Area (800x450)",
            fill="gray")

    def create_right_panel(self):
        # Right Panel Container
        self.right_panel = tk.Frame(self.content_frame)
        self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH,
                              expand=True, padx=(PANEL_PADDING, 0))

        # Label
        lbl = tk.Label(self.right_panel, text="Robot Perception",
                       font=("Arial", 12, "bold"))
        lbl.pack(side=tk.TOP, pady=10)

        # 1. Current Input
        input_frame = tk.Frame(self.right_panel)
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            input_frame, text="Current Input").pack(
            anchor="w")
        self.input_canvas = tk.Canvas(
            input_frame, height=100, highlightthickness=0)
        self.input_canvas.pack(fill=tk.X, pady=(5, 0))

        # 2. Retrieved Memory
        memory_frame = tk.Frame(self.right_panel)
        memory_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            memory_frame, text="Retrieved Memory").pack(
            anchor="w")
        self.memory_canvas = tk.Canvas(
            memory_frame, height=100, highlightthickness=0)
        self.memory_canvas.pack(fill=tk.X, pady=(5, 0))

        # 3. Similarity Metric
        sim_frame = tk.Frame(self.right_panel)
        sim_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            sim_frame, text="Similarity Metric").pack(
            anchor="w")
        self.sim_canvas = tk.Canvas(
            sim_frame, height=40, highlightthickness=0)
        self.sim_canvas.pack(fill=tk.X, pady=(5, 0))

    def open_map_editor(self):
        MapEditor(self.root, self.current_map_image, self.on_map_saved)

    def on_map_saved(self, new_map_image):
        self.current_map_image = new_map_image
        self.update_map_display()

    def load_last_map(self):
        """Load the last used map automatically on startup"""
        last_path = self.config.get_last_map_path()

        if last_path and os.path.exists(last_path):
            try:
                # Load the image
                imported_image = Image.open(last_path)

                # Resize to map dimensions if necessary
                if imported_image.size != (MAP_WIDTH, MAP_HEIGHT):
                    imported_image = imported_image.resize(
                        (MAP_WIDTH, MAP_HEIGHT),
                        Image.Resampling.LANCZOS
                    )

                # Convert to RGB if necessary
                if imported_image.mode != "RGB":
                    imported_image = imported_image.convert("RGB")

                self.current_map_image = imported_image
                print(f"Loaded last map: {last_path}")

            except Exception as e:
                print(f"Failed to load last map: {e}")
                # Keep the default white map

    def import_map(self):
        """Import a map from a PNG file"""
        # Get initial directory from last used path
        initial_dir = None
        last_path = self.config.get_last_map_path()
        if last_path and os.path.exists(os.path.dirname(last_path)):
            initial_dir = os.path.dirname(last_path)

        file_path = filedialog.askopenfilename(
            title="Import Map",
            initialdir=initial_dir,
            filetypes=[
                ("PNG files", "*.png")
            ]
        )

        if file_path:
            try:
                # Load the image
                imported_image = Image.open(file_path)

                # Resize to map dimensions if necessary
                if imported_image.size != (MAP_WIDTH, MAP_HEIGHT):
                    imported_image = imported_image.resize(
                        (MAP_WIDTH, MAP_HEIGHT),
                        Image.Resampling.LANCZOS
                    )

                # Convert to RGB if necessary
                if imported_image.mode != "RGB":
                    imported_image = imported_image.convert("RGB")

                self.current_map_image = imported_image
                self.update_map_display()

                # Save the path to config
                self.config.set_last_map_path(file_path)

                messagebox.showinfo("Success", "Map imported successfully!")

            except Exception as e:
                messagebox.showerror(
                    "Error", f"Failed to import map: {str(e)}")

    def export_map(self):
        """Export the current map to a PNG file"""
        # Get initial directory from last used path
        initial_dir = None
        initial_file = "map.png"
        last_path = self.config.get_last_map_path()
        if last_path:
            if os.path.exists(os.path.dirname(last_path)):
                initial_dir = os.path.dirname(last_path)
            initial_file = os.path.basename(last_path)

        file_path = filedialog.asksaveasfilename(
            title="Export Map",
            initialdir=initial_dir,
            initialfile=initial_file,
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png")
            ]
        )

        if file_path:
            try:
                self.current_map_image.save(file_path)

                # Save the path to config
                self.config.set_last_map_path(file_path)

                messagebox.showinfo("Success", f"Map exported to {file_path}")
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Failed to export map: {str(e)}")

    def update_map_display(self):
        self.tk_map_image = ImageTk.PhotoImage(self.current_map_image)
        self.map_canvas.delete("all")
        self.map_canvas.create_image(
            0, 0, image=self.tk_map_image, anchor="nw")
        self.draw_robot()

        # Update camera view
        self.capture_camera_view()
        self.display_camera_view()

    def draw_robot(self):
        # Draw the viewing cone first (behind the robot)
        self.draw_viewing_cone()

        # Draw robot as a blue circle on top
        self.map_canvas.create_oval(
            self.robot_x - self.robot_radius,
            self.robot_y - self.robot_radius,
            self.robot_x + self.robot_radius,
            self.robot_y + self.robot_radius,
            fill=COLOR_ROBOT_GT,
            outline=COLOR_ROBOT_GT,
            tags="robot"
        )

    def draw_viewing_cone(self):
        # Calculate the cone's arc points
        half_cone = self.cone_angle / 2

        # Starting point is the robot center
        points = [self.robot_x, self.robot_y]

        # Create the cone arc
        # Add points along the arc from -half_cone to +half_cone
        num_points = 20
        for i in range(num_points + 1):
            angle_offset = -half_cone + (self.cone_angle * i / num_points)
            current_angle = math.radians(self.robot_angle + angle_offset)

            # Calculate direction vector
            dx = math.cos(current_angle)
            dy = math.sin(current_angle)

            # Cast ray and find distance to wall or edge
            distance = self.cast_ray(
                self.robot_x, self.robot_y, dx, dy)

            x = self.robot_x + distance * dx
            y = self.robot_y + distance * dy
            points.extend([x, y])

        # Close the polygon back to the center
        points.extend([self.robot_x, self.robot_y])

        # Draw the cone with semi-transparent blue
        self.map_canvas.create_polygon(
            points,
            fill=COLOR_ROBOT_GT,
            stipple="gray50",  # Makes it semi-transparent
            outline=COLOR_ROBOT_GT,
            tags="cone"
        )

    def get_distance_to_edge(self, x, y, dx, dy):
        """Calculate the distance from (x, y) to the map edge in direction (dx, dy)"""
        # Calculate distances to each edge
        distances = []

        # Right edge (x = MAP_WIDTH)
        if dx > 0:
            t = (MAP_WIDTH - x) / dx
            distances.append(t)
        # Left edge (x = 0)
        elif dx < 0:
            t = -x / dx
            distances.append(t)

        # Bottom edge (y = MAP_HEIGHT)
        if dy > 0:
            t = (MAP_HEIGHT - y) / dy
            distances.append(t)
        # Top edge (y = 0)
        elif dy < 0:
            t = -y / dy
            distances.append(t)

        # Return the minimum positive distance
        return min(distances) if distances else 0

    def cast_ray(self, x, y, dx, dy):
        """
        Cast a ray from (x, y) in direction (dx, dy) until hitting a wall or map edge.
        Returns the distance to the first obstacle.
        """
        # Get maximum possible distance (to map edge)
        max_distance = self.get_distance_to_edge(x, y, dx, dy)

        # Step size for ray marching (smaller = more accurate but slower)
        step_size = 1.0

        # Get pixel data from the map image
        pixels = self.current_map_image.load()

        # March along the ray
        distance = 0
        while distance < max_distance:
            distance += step_size

            # Calculate current position
            current_x = x + distance * dx
            current_y = y + distance * dy

            # Round to integer pixel coordinates
            pixel_x = int(round(current_x))
            pixel_y = int(round(current_y))

            # Check if we're out of bounds (shouldn't happen, but safety check)
            if pixel_x < 0 or pixel_x >= MAP_WIDTH or pixel_y < 0 or pixel_y >= MAP_HEIGHT:
                return distance

            # Get pixel color
            pixel_color = pixels[pixel_x, pixel_y]

            # Check if pixel is black (wall) - threshold for near-black pixels
            # RGB values less than 50 are considered walls
            if pixel_color[0] < 50 and pixel_color[1] < 50 and pixel_color[2] < 50:
                return distance

        return max_distance

    def capture_camera_view(self):
        """
        Capture a 1D strip of pixels from what the robot sees.
        Samples pixels along rays cast from the robot's viewing direction.
        """
        half_cone = self.cone_angle / 2
        pixels = self.current_map_image.load()

        # Create a list to store RGB values for each sample
        camera_strip = []

        # Sample pixels across the cone angle
        for i in range(self.camera_samples):
            # Calculate angle for this sample
            angle_offset = -half_cone + (self.cone_angle * i /
                                         (self.camera_samples - 1))
            current_angle = math.radians(self.robot_angle + angle_offset)

            # Calculate direction vector
            dx = math.cos(current_angle)
            dy = math.sin(current_angle)

            # Cast ray to find the distance to wall/edge
            distance = self.cast_ray(self.robot_x, self.robot_y, dx, dy)

            # Get the pixel at that point
            end_x = self.robot_x + distance * dx
            end_y = self.robot_y + distance * dy

            # Clamp to map bounds
            pixel_x = int(round(max(0, min(MAP_WIDTH - 1, end_x))))
            pixel_y = int(round(max(0, min(MAP_HEIGHT - 1, end_y))))

            # Get the RGB color at this point
            color = pixels[pixel_x, pixel_y]
            camera_strip.append(color)

        # Store as PIL Image (1D image with height=1)
        self.current_camera_view = Image.new('RGB', (self.camera_samples, 1))
        self.current_camera_view.putdata(camera_strip)

    def display_camera_view(self):
        """Display the camera view in the input canvas"""
        if self.current_camera_view is None:
            return

        # Get canvas dimensions
        canvas_width = self.input_canvas.winfo_width()
        canvas_height = self.input_canvas.winfo_height()

        # If canvas not yet sized, use default
        if canvas_width <= 1:
            canvas_width = 300
            canvas_height = 100

        # Scale up the 1-pixel-high image to fill the canvas
        scaled_image = self.current_camera_view.resize(
            (canvas_width, canvas_height),
            Image.Resampling.NEAREST  # Use nearest neighbor to keep sharp pixels
        )

        # Convert to PhotoImage
        tk_image = ImageTk.PhotoImage(scaled_image)

        # Clear canvas and draw
        self.input_canvas.delete("all")
        self.input_canvas.create_image(0, 0, image=tk_image, anchor="nw")

        # Store reference to prevent garbage collection
        self.input_canvas.image = tk_image

    def move_robot(self, dx, dy):
        # Update robot position
        self.robot_x += dx
        self.robot_y += dy

        # Keep robot within bounds
        self.robot_x = max(self.robot_radius, min(
            MAP_WIDTH - self.robot_radius, self.robot_x))
        self.robot_y = max(self.robot_radius, min(
            MAP_HEIGHT - self.robot_radius, self.robot_y))

        # Redraw the map with the robot at new position
        self.update_map_display()

    def rotate_robot(self, angle_delta):
        # Update robot angle
        self.robot_angle += angle_delta
        # Keep angle in [0, 360) range
        self.robot_angle = self.robot_angle % 360

        # Redraw the map with the robot at new angle
        self.update_map_display()

    def run(self):
        self.update_map_display()
        self.root.mainloop()


if __name__ == "__main__":
    app = App()
    app.run()
