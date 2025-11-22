import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from constants import *
from PIL import Image, ImageTk, ImageFilter, ImageDraw
from editor import MapEditor
from config import ConfigManager
from hopfield import ModernHopfieldNetwork
import math
import os
import numpy as np


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
        self.cone_angle = 360  # Cone angle in degrees

        # Camera parameters
        self.camera_samples = 100  # Number of pixel samples in the 1D strip
        self.current_camera_view = None  # Store current camera strip

        # Sample memory storage
        self.sample_positions = []  # List of (x, y, angle) tuples
        self.sample_embeddings = []  # List of embeddings (numpy arrays)
        self.sample_views = []  # List of camera view images (PIL Images)
        self.sample_dots = []  # Canvas IDs of sample dots

        # Modern Hopfield Network
        self.hopfield_network = None
        self.is_trained = False

        # Estimated position from localization
        self.estimated_x = None
        self.estimated_y = None
        self.estimated_angle = None
        self.retrieved_sample_idx = None

        # Hover state
        self.hovered_sample_idx = None

        # Key press tracking for smooth movement
        self.keys_pressed = set()
        self.update_interval = 16  # ~60 FPS (16ms per frame)

        # Rotation tracking to prevent continuous rotation
        self.rotation_keys_pressed = set()

        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.create_layout()
        self.create_toolbar()
        self.create_left_panel()
        self.create_right_panel()

        # Load last used map if available
        self.load_last_map()

        # Bind keyboard controls for press and release
        self.root.bind('<KeyPress-w>', lambda e: self.on_key_press('w'))
        self.root.bind('<KeyRelease-w>', lambda e: self.on_key_release('w'))
        self.root.bind('<KeyPress-a>', lambda e: self.on_key_press('a'))
        self.root.bind('<KeyRelease-a>', lambda e: self.on_key_release('a'))
        self.root.bind('<KeyPress-s>', lambda e: self.on_key_press('s'))
        self.root.bind('<KeyRelease-s>', lambda e: self.on_key_release('s'))
        self.root.bind('<KeyPress-d>', lambda e: self.on_key_press('d'))
        self.root.bind('<KeyRelease-d>', lambda e: self.on_key_release('d'))
        self.root.bind(
            '<KeyPress-j>', lambda e: self.on_rotation_key_press('j'))
        self.root.bind('<KeyRelease-j>',
                       lambda e: self.on_rotation_key_release('j'))
        self.root.bind(
            '<KeyPress-l>', lambda e: self.on_rotation_key_press('l'))
        self.root.bind('<KeyRelease-l>',
                       lambda e: self.on_rotation_key_release('l'))

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

        # Progress bar area at the bottom
        self.progress_frame = tk.Frame(self.main_container)
        self.progress_frame.pack(
            side=tk.BOTTOM, fill=tk.X, padx=PANEL_PADDING,
            pady=(0, PANEL_PADDING))

        self.progress_label = tk.Label(
            self.progress_frame, text="", anchor="w")
        self.progress_label.pack(side=tk.LEFT, padx=(0, 10))

        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            orient=tk.HORIZONTAL,
            mode='determinate',
            length=400
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def create_toolbar(self):
        buttons = {
            "Edit Map": self.open_map_editor,
            "Import Map": self.import_map,
            "Export Map": self.export_map,
            "Auto Sample": self.auto_sample,
            "Train": self.train_network
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

        # Bind mouse motion for hover effects
        self.map_canvas.bind('<Motion>', self.on_canvas_hover)

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

    def auto_sample(self):
        """
        Automatically generate samples from the current map.
        Samples at discrete positions with SAMPLE_STRIDE spacing and
        SAMPLE_ROTATIONS rotations at each position.
        """
        # Clear existing samples
        self.sample_positions = []
        self.sample_embeddings = []
        self.sample_views = []
        self.clear_sample_dots()

        # Calculate grid positions
        x_positions = list(range(0, MAP_WIDTH, SAMPLE_STRIDE))
        y_positions = list(range(0, MAP_HEIGHT, SAMPLE_STRIDE))

        # Calculate angles (45-degree increments)
        angles = [i * (360 / SAMPLE_ROTATIONS)
                  for i in range(SAMPLE_ROTATIONS)]

        # Store original robot state
        original_x = self.robot_x
        original_y = self.robot_y
        original_angle = self.robot_angle

        # Generate samples
        total_samples = len(x_positions) * len(y_positions) * len(angles)
        sample_count = 0

        # Initialize progress bar
        self.progress_bar['maximum'] = total_samples
        self.progress_bar['value'] = 0
        self.progress_label['text'] = f"Sampling: 0/{total_samples}"

        for x in x_positions:
            for y in y_positions:
                for angle in angles:
                    # Position robot
                    self.robot_x = x
                    self.robot_y = y
                    self.robot_angle = angle

                    # Capture camera view
                    self.capture_camera_view()

                    # Create embedding (flatten the RGB values)
                    embedding = self.create_embedding(self.current_camera_view)

                    # Store sample
                    self.sample_positions.append((x, y, angle))
                    self.sample_embeddings.append(embedding)
                    self.sample_views.append(
                        self.current_camera_view.copy())

                    sample_count += 1

                    # Update progress bar
                    if sample_count % 10 == 0 or sample_count == total_samples:
                        self.progress_bar['value'] = sample_count
                        self.progress_label['text'] = f"Sampling: {
                            sample_count} /{total_samples} "
                        self.root.update()  # Force UI update

        # Restore original robot state
        self.robot_x = original_x
        self.robot_y = original_y
        self.robot_angle = original_angle

        # Update display
        self.update_map_display()
        self.draw_sample_dots()

        # Clear progress bar
        self.progress_bar['value'] = 0
        self.progress_label['text'] = ""

        messagebox.showinfo("Success",
                            f"Generated {len(self.sample_positions)} samples")

    def train_network(self):
        """
        Train the Modern Hopfield Network using the collected samples.
        """
        # Check if samples exist
        if len(self.sample_embeddings) == 0:
            messagebox.showerror(
                "Error",
                "No samples available. Please run 'Auto Sample' first.")
            return

        # Get embedding dimension from first sample
        embedding_dim = len(self.sample_embeddings[0])

        # Initialize the Hopfield Network
        self.hopfield_network = ModernHopfieldNetwork(embedding_dim)

        # Progress callback for training
        def update_training_progress(current, total):
            self.progress_bar['value'] = current
            self.progress_label['text'] = f"Training: {current}/{total}"
            self.root.update()  # Force UI update

        # Initialize progress bar
        self.progress_bar['maximum'] = len(self.sample_embeddings)
        self.progress_bar['value'] = 0
        self.progress_label['text'] = f"Training: 0/{
            len(self.sample_embeddings)} "

        try:
            # Train the network
            self.hopfield_network.train(
                self.sample_embeddings,
                progress_callback=update_training_progress
            )

            self.is_trained = True

            # Clear progress bar
            self.progress_bar['value'] = 0
            self.progress_label['text'] = ""

            messagebox.showinfo(
                "Success",
                f"Network trained with {len(self.sample_embeddings)} patterns")

        except Exception as e:
            self.progress_bar['value'] = 0
            self.progress_label['text'] = ""
            messagebox.showerror(
                "Training Error", f"Failed to train network: {str(e)}")

    def create_embedding(self, camera_view):
        """
        Create an embedding vector from a camera view image.
        For MVP, we simply flatten the RGB values.

        Args:
            camera_view: PIL Image (1D strip)

        Returns:
            numpy array representing the embedding
        """
        # Get pixel data
        pixels = list(camera_view.getdata())

        # Flatten RGB values into a single vector
        embedding = []
        for r, g, b in pixels:
            embedding.extend([r, g, b])

        return np.array(embedding, dtype=np.float32)

    def clear_sample_dots(self):
        """Remove all sample dots from the canvas"""
        for dot_id in self.sample_dots:
            self.map_canvas.delete(dot_id)
        self.sample_dots = []

    def draw_sample_dots(self):
        """Draw red dots at sample positions on the canvas"""
        self.clear_sample_dots()

        for i, (x, y, _angle) in enumerate(self.sample_positions):
            # Draw a red dot
            dot_id = self.map_canvas.create_oval(
                x - SAMPLE_DOT_RADIUS,
                y - SAMPLE_DOT_RADIUS,
                x + SAMPLE_DOT_RADIUS,
                y + SAMPLE_DOT_RADIUS,
                fill=COLOR_SAMPLE_DOT,
                outline=COLOR_SAMPLE_DOT,
                tags=("sample_dot", f"sample_{i}")
            )
            self.sample_dots.append(dot_id)

    def on_canvas_hover(self, event):
        """
        Handle mouse hover over the canvas.
        Display the saved sample when hovering over a red dot.
        """
        # Find if we're hovering over a sample dot
        canvas_x = event.x
        canvas_y = event.y

        # Find the closest sample within a reasonable distance
        closest_idx = None
        min_distance = SAMPLE_DOT_RADIUS * 3  # Search radius

        for i, (x, y, _angle) in enumerate(self.sample_positions):
            distance = math.sqrt((x - canvas_x)**2 + (y - canvas_y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i

        # If hovering over a new sample, update display
        if closest_idx != self.hovered_sample_idx:
            self.hovered_sample_idx = closest_idx

            if closest_idx is not None:
                # Display the saved sample view
                self.display_saved_sample(closest_idx)
            else:
                # Clear the retrieved memory canvas
                self.memory_canvas.delete("all")

    def update_map_display(self):
        # Create a copy of the map to draw on
        display_image = self.current_map_image.copy()

        # Draw the viewing cone with transparency on the image
        self.draw_viewing_cone_on_image(display_image)

        self.tk_map_image = ImageTk.PhotoImage(display_image)
        self.map_canvas.delete("all")
        self.map_canvas.create_image(
            0, 0, image=self.tk_map_image, anchor="nw")

        # Draw sample dots (behind robot)
        self.draw_sample_dots()

        # Update camera view
        self.capture_camera_view()
        self.display_camera_view()

        # Perform localization if network is trained
        if self.is_trained:
            self.localize()

        # Draw robot on top (after localization so estimated angle is current)
        self.draw_robot()

    def draw_robot(self):
        # Draw robot ground truth direction line (blue)
        line_length = 20
        angle_rad = math.radians(self.robot_angle)
        end_x = self.robot_x + line_length * math.cos(angle_rad)
        end_y = self.robot_y + line_length * math.sin(angle_rad)
        self.map_canvas.create_line(
            self.robot_x, self.robot_y,
            end_x, end_y,
            fill=COLOR_ROBOT_GT,  # Blue
            width=2,
            tags="robot_direction"
        )

        # Draw robot ground truth as a blue circle
        self.map_canvas.create_oval(
            self.robot_x - self.robot_radius,
            self.robot_y - self.robot_radius,
            self.robot_x + self.robot_radius,
            self.robot_y + self.robot_radius,
            fill=COLOR_ROBOT_GT,
            outline=COLOR_ROBOT_GT,
            tags="robot"
        )

        # Draw estimated position (green circle with purple direction line) last so it's always visible
        if self.estimated_x is not None and self.estimated_y is not None:
            # Draw purple direction line
            if self.estimated_angle is not None:
                line_length = 20
                angle_rad = math.radians(self.estimated_angle)
                end_x = self.estimated_x + line_length * math.cos(angle_rad)
                end_y = self.estimated_y + line_length * math.sin(angle_rad)
                self.map_canvas.create_line(
                    self.estimated_x, self.estimated_y,
                    end_x, end_y,
                    fill="#800080",  # Purple
                    width=2,
                    tags="estimated_direction"
                )

            # Draw green circle
            self.map_canvas.create_oval(
                self.estimated_x - self.robot_radius,
                self.estimated_y - self.robot_radius,
                self.estimated_x + self.robot_radius,
                self.estimated_y + self.robot_radius,
                fill="#00FF00",  # Green
                outline="#00FF00",
                tags="estimated"
            )

    def draw_viewing_cone_on_image(self, image):
        """Draw the viewing cone with 50% transparency directly on a PIL image"""
        # Calculate the cone's arc points
        half_cone = self.cone_angle / 2

        # Starting point is the robot center
        points = [(self.robot_x, self.robot_y)]

        # Create the cone arc
        # Add points along the arc from -half_cone to +half_cone
        num_points = 50  # More points for smoother curve
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
            points.append((x, y))

        # Create a transparent overlay
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Convert hex color to RGBA with 50% opacity (alpha=127)
        # COLOR_ROBOT_GT is "#0000FF" (blue)
        cone_color = (0, 0, 255, 127)  # RGBA: blue with 50% opacity

        # Draw the polygon
        draw.polygon(points, fill=cone_color, outline=None)

        # Composite the overlay onto the original image
        image.paste(overlay, (0, 0), overlay)

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

            # Check if pixel is not white (wall)
            # Any pixel that's not pure white (255, 255, 255) is considered a wall
            if pixel_color[0] < 255 or pixel_color[1] < 255 or pixel_color[2] < 255:
                return distance

        return max_distance

    def capture_camera_view(self):
        """
        Capture a 1D strip of pixels from what the robot sees.
        Samples pixels along rays cast from the robot's viewing direction.
        Walls appear less opaque (more white) the further they are from the robot.
        """
        half_cone = self.cone_angle / 2
        pixels = self.current_map_image.load()

        # Create a list to store RGB values for each sample
        camera_strip = []

        # Maximum distance for opacity calculation (cone length)
        max_distance = self.cone_length

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

            # Calculate opacity based on distance
            # Closer = more opaque (opacity = 1.0), further = less opaque (opacity approaches 0.0)
            # Use exponential decay for more natural distance perception
            opacity = math.exp(-distance / max_distance * 0.1)

            # Blend the color with white based on opacity
            # opacity = 1.0 -> full color, opacity = 0.0 -> white
            blended_color = (
                int(color[0] * opacity + 255 * (1 - opacity)),
                int(color[1] * opacity + 255 * (1 - opacity)),
                int(color[2] * opacity + 255 * (1 - opacity))
            )

            camera_strip.append(blended_color)

        # Store as PIL Image (1D image with height=1)
        self.current_camera_view = Image.new('RGB', (self.camera_samples, 1))
        self.current_camera_view.putdata(camera_strip)

        # Apply Gaussian blur to make observations more robust to small changes
        if CAMERA_BLUR_RADIUS > 0:
            # Resize to larger height temporarily for better blur effect
            temp_height = 10
            temp_view = self.current_camera_view.resize(
                (self.camera_samples, temp_height),
                Image.Resampling.NEAREST
            )
            # Apply blur
            temp_view = temp_view.filter(ImageFilter.GaussianBlur(
                radius=CAMERA_BLUR_RADIUS))
            # Resize back to 1D strip (take middle row)
            self.current_camera_view = temp_view.resize(
                (self.camera_samples, 1),
                Image.Resampling.BILINEAR
            )

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

    def localize(self):
        """
        Perform localization using the trained Hopfield Network.
        Retrieves the closest matching pattern and estimates the robot's position.
        """
        if not self.is_trained or self.hopfield_network is None:
            return

        if self.current_camera_view is None:
            return

        try:
            # Create embedding from current camera view
            query_embedding = self.create_embedding(self.current_camera_view)

            # Retrieve best matching pattern
            indices, weights = self.hopfield_network.retrieve(
                query_embedding, top_k=1)
            best_idx = indices[0]
            best_weight = weights[0]

            # Store retrieved sample index
            self.retrieved_sample_idx = best_idx

            # Get position from the sample
            x, y, angle = self.sample_positions[best_idx]
            self.estimated_x = x
            self.estimated_y = y
            self.estimated_angle = angle

            # Display the retrieved memory view
            self.display_retrieved_memory(best_idx, best_weight)

            # Display similarity metric
            self.display_similarity_metric(best_weight)

        except Exception as e:
            print(f"Localization error: {e}")

    def display_retrieved_memory(self, sample_idx, weight):
        """
        Display the retrieved memory view in the retrieved memory canvas.

        Args:
            sample_idx: Index of the retrieved sample
            weight: Attention weight/confidence score
        """
        if sample_idx < 0 or sample_idx >= len(self.sample_views):
            return

        saved_view = self.sample_views[sample_idx]

        # Get canvas dimensions
        canvas_width = self.memory_canvas.winfo_width()
        canvas_height = self.memory_canvas.winfo_height()

        # If canvas not yet sized, use default
        if canvas_width <= 1:
            canvas_width = 300
            canvas_height = 100

        # Scale up the 1-pixel-high image to fill the canvas
        scaled_image = saved_view.resize(
            (canvas_width, canvas_height),
            Image.Resampling.NEAREST
        )

        # Convert to PhotoImage
        tk_image = ImageTk.PhotoImage(scaled_image)

        # Clear canvas and draw
        self.memory_canvas.delete("all")
        self.memory_canvas.create_image(0, 0, image=tk_image, anchor="nw")

        # Add text overlay with position info
        x, y, angle = self.sample_positions[sample_idx]
        info_text = f"Retrieved: ({x}, {y}), {
            angle: .1f} ° | Weight: {
            weight: .4f} "
        self.memory_canvas.create_text(
            canvas_width // 2, 10,
            text=info_text,
            fill="white",
            font=("Arial", 9, "bold")
        )

        # Store reference to prevent garbage collection
        self.memory_canvas.image = tk_image

    def display_similarity_metric(self, weight):
        """
        Display the similarity metric as a progress bar.

        Args:
            weight: Attention weight/confidence (0 to 1)
        """
        canvas_width = self.sim_canvas.winfo_width()
        canvas_height = self.sim_canvas.winfo_height()

        if canvas_width <= 1:
            canvas_width = 300
            canvas_height = 40

        # Clear canvas
        self.sim_canvas.delete("all")

        # Draw background bar
        bar_height = 20
        bar_y = canvas_height // 2 - bar_height // 2
        self.sim_canvas.create_rectangle(
            10, bar_y, canvas_width - 10, bar_y + bar_height,
            fill="gray", outline="black"
        )

        # Draw filled bar based on weight
        bar_width = (canvas_width - 20) * weight
        self.sim_canvas.create_rectangle(
            10, bar_y, 10 + bar_width, bar_y + bar_height,
            fill="green", outline=""
        )

        # Add text
        percentage = weight * 100
        self.sim_canvas.create_text(
            canvas_width // 2, bar_y + bar_height // 2,
            text=f"Confidence: {percentage:.1f}%",
            fill="white",
            font=("Arial", 10, "bold")
        )

    def display_saved_sample(self, sample_idx):
        """
        Display a saved sample view in the retrieved memory canvas.

        Args:
            sample_idx: Index of the sample to display
        """
        if sample_idx < 0 or sample_idx >= len(self.sample_views):
            return

        saved_view = self.sample_views[sample_idx]

        # Get canvas dimensions
        canvas_width = self.memory_canvas.winfo_width()
        canvas_height = self.memory_canvas.winfo_height()

        # If canvas not yet sized, use default
        if canvas_width <= 1:
            canvas_width = 300
            canvas_height = 100

        # Scale up the 1-pixel-high image to fill the canvas
        scaled_image = saved_view.resize(
            (canvas_width, canvas_height),
            Image.Resampling.NEAREST  # Use nearest neighbor to keep sharp pixels
        )

        # Convert to PhotoImage
        tk_image = ImageTk.PhotoImage(scaled_image)

        # Clear canvas and draw
        self.memory_canvas.delete("all")
        self.memory_canvas.create_image(0, 0, image=tk_image, anchor="nw")

        # Add text overlay with position info
        x, y, angle = self.sample_positions[sample_idx]
        info_text = f"Pos: ({x}, {y}), Angle: {angle:.1f}°"
        self.memory_canvas.create_text(
            canvas_width // 2, 10,
            text=info_text,
            fill="white",
            font=("Arial", 10, "bold")
        )

        # Store reference to prevent garbage collection
        self.memory_canvas.image = tk_image

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
        # Update robot angle by the specified delta (already a discrete step)
        self.robot_angle += angle_delta
        # Keep angle in [0, 360) range
        self.robot_angle = self.robot_angle % 360

        # Redraw the map with the robot at new angle
        self.update_map_display()

    def on_key_press(self, key):
        """Track when a key is pressed"""
        self.keys_pressed.add(key)

    def on_key_release(self, key):
        """Track when a key is released"""
        self.keys_pressed.discard(key)

    def on_rotation_key_press(self, key):
        """Handle rotation key press - allows continuous rotation"""
        self.rotation_keys_pressed.add(key)

    def on_rotation_key_release(self, key):
        """Track when rotation key is released"""
        self.rotation_keys_pressed.discard(key)

    def update_loop(self):
        """Continuous update loop for smooth movement"""
        # Check which keys are currently pressed and move accordingly
        dx, dy = 0, 0
        d_angle = 0

        if 'w' in self.keys_pressed:
            dy -= self.robot_speed
        if 's' in self.keys_pressed:
            dy += self.robot_speed
        if 'a' in self.keys_pressed:
            dx -= self.robot_speed
        if 'd' in self.keys_pressed:
            dx += self.robot_speed

        # Handle rotation
        if 'j' in self.rotation_keys_pressed:
            d_angle -= self.robot_rotation_speed
        if 'l' in self.rotation_keys_pressed:
            d_angle += self.robot_rotation_speed

        # Update position if there's movement
        if dx != 0 or dy != 0:
            self.move_robot(dx, dy)

        # Update rotation if rotating
        if d_angle != 0:
            self.rotate_robot(d_angle)

        # Schedule next update
        self.root.after(self.update_interval, self.update_loop)

    def run(self):
        self.update_map_display()
        # Start the continuous update loop
        self.update_loop()
        self.root.mainloop()


if __name__ == "__main__":
    app = App()
    app.run()
