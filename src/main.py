import tkinter as tk
from tkinter import filedialog
from constants import *
from PIL import Image, ImageTk, ImageFilter, ImageDraw
from editor import MapEditor
from config import ConfigManager
from hopfield import ModernHopfieldNetwork
from heatmap_builder import HeatmapBuilder
import math
import os
import numpy as np


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Robot Localization via Modern Hopfield Networks")
        self.root.geometry(f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}")

        self.config = ConfigManager()

        self.robot_image = None
        self.tk_robot_image = None
        self.load_robot_image()

        self.current_map_image = Image.new(
            "RGB", (MAP_WIDTH, MAP_HEIGHT), "white")
        self.tk_map_image = None

        self.robot_x = MAP_WIDTH // 2
        self.robot_y = MAP_HEIGHT // 2
        self.robot_radius = 5
        self.robot_speed = 5
        self.robot_angle = 0
        self.robot_rotation_speed = 7.5  # Degrees per key press
        self.cone_length = 40  # Length of the viewing cone
        self.cone_angle = CAMERA_FOV  # Cone angle in degrees

        self.camera_blur_radius = CAMERA_BLUR_RADIUS
        self.visibility_index = 0.1  # Distance opacity factor (default 0.1)

        self.beta = DEFAULT_BETA  # Inverse temperature parameter

        self.interleaved_rgb = tk.BooleanVar(value=INTERLEAVED_RGB)

        self.camera_samples = 100  # Number of pixel samples in the 1D strip
        self.current_camera_view = None  # Store current camera strip

        self.sample_positions = []  # List of (x, y, angle) tuples
        self.sample_embeddings = []  # List of embeddings (numpy arrays)
        self.sample_views = []  # List of camera view images (PIL Images)
        self.sample_dots = []  # Canvas IDs of sample dots

        self.hopfield_network = None
        self.is_trained = False

        self.estimated_x = None
        self.estimated_y = None
        self.estimated_angle = None
        self.retrieved_sample_idx = None

        self.sample_similarities = None  # Array of similarity scores for each sample

        self.accuracy_avg_distance = None  # Average distance error
        self.accuracy_num_tests = None  # Number of test positions
        self.test_positions = []
        self.test_position_dots = []  # Canvas IDs of test position dots
        self.show_test_positions = tk.BooleanVar(value=False)  # Checkbox state

        self.show_confidence_heatmap = tk.BooleanVar(
            value=False)  # Checkbox state
        self.heatmap_overlay = None  # PIL Image for the heatmap overlay
        self.heatmap_grid_positions_by_angle = {}
        self.heatmap_grid_confidences_by_angle = {}
        self.heatmap_computed = False  # Flag to track if heatmap is computed
        self.heatmap_angles = []  # List of angles for which heatmaps were computed

        self.hovered_sample_idx = None

        self.keys_pressed = set()
        self.update_interval = 16  # ~60 FPS (16ms per frame)

        self.rotation_keys_pressed = set()

        self.map_scale_factor = 1.0
        self.map_offset_x = 0
        self.map_offset_y = 0

        self.create_layout()
        self.create_toolbar()
        self.create_left_panel()
        self.create_right_panel()

        self.load_last_map()

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
        self.main_container = tk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        self.toolbar_frame = tk.Frame(self.main_container)
        self.toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.content_frame = tk.Frame(self.main_container)
        self.content_frame.pack(side=tk.TOP, fill=tk.BOTH,
                                expand=True, padx=5, pady=5)

        self.status_frame = tk.Frame(self.main_container)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.status_label = tk.Label(self.status_frame, text="", anchor="w")
        self.status_label.pack(side=tk.LEFT, fill=tk.X,
                               expand=True, padx=5, pady=5)

    def create_toolbar(self):
        buttons = {
            "Edit Map": self.open_map_editor,
            "Import Map": self.import_map,
            "Export Map": self.export_map,
            "Sample & Train": self.sample_and_train
        }

        for btn_text, command in buttons.items():
            btn = tk.Button(self.toolbar_frame, text=btn_text, command=command)
            btn.pack(side=tk.LEFT, padx=5, pady=5)

    def create_left_panel(self):
        self.left_panel = tk.Frame(self.content_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH,
                             expand=True, padx=5, pady=5)

        self.map_canvas = tk.Canvas(
            self.left_panel, highlightthickness=1, bg="#000000")
        self.map_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.map_canvas.bind('<Motion>', self.on_canvas_hover)
        self.map_canvas.bind('<Configure>', self.on_map_canvas_resize)

        self.map_canvas.create_text(
            MAP_WIDTH // 2, MAP_HEIGHT // 2,
            text=f"Map Area ({MAP_WIDTH}x{MAP_HEIGHT})",
            fill="gray"
        )

    def create_right_panel(self):
        # Create a frame for the right panel with a scrollbar
        right_panel_container = tk.Frame(self.content_frame, width=400)
        right_panel_container.pack(
            side=tk.LEFT, fill=tk.Y, expand=False, padx=(5, 10), pady=5)
        right_panel_container.pack_propagate(False)

        # Add a canvas and scrollbar for scrolling
        canvas = tk.Canvas(right_panel_container)
        scrollbar = tk.Scrollbar(
            right_panel_container, orient=tk.VERTICAL, command=canvas.yview)
        self.right_panel = tk.Frame(canvas)

        self.right_panel.bind(
            "<Configure>", lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=self.right_panel, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Adjust the layout to prevent the scrollbar from overlapping the content
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        # Add content to the right panel
        input_frame = tk.LabelFrame(self.right_panel, text="Current Input")
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        self.input_canvas = tk.Canvas(
            input_frame, height=25, highlightthickness=0, bg="#f0f0f0"
        )
        self.input_canvas.pack(fill=tk.X, padx=5, pady=5)

        self.memory_frame = tk.LabelFrame(
            self.right_panel, text="Retrieved Memory"
        )
        self.memory_frame.pack(fill=tk.X, padx=5, pady=5)

        self.memory_canvas = tk.Canvas(
            self.memory_frame, height=25, highlightthickness=0, bg="#f0f0f0"
        )
        self.memory_canvas.pack(fill=tk.X, padx=5, pady=5)

        sim_frame = tk.LabelFrame(self.right_panel, text="Similarity Metric")
        sim_frame.pack(fill=tk.X, padx=5, pady=5)

        self.sim_label = tk.Label(
            sim_frame, text="Confidence: 0.0%", anchor=tk.CENTER
        )
        self.sim_label.pack(fill=tk.X, padx=5, pady=5)

        self.sim_progressbar = tk.Canvas(sim_frame, height=10, bg="#d9d9d9")
        self.sim_progressbar.pack(fill=tk.X, padx=5, pady=5)

        settings_frame = tk.LabelFrame(
            self.right_panel, text="Camera Settings"
        )
        settings_frame.pack(fill=tk.X, padx=5, pady=5)

        blur_container = tk.LabelFrame(settings_frame, text="Blur Radius")
        blur_container.pack(fill=tk.X, padx=5, pady=5)

        self.blur_value_label = tk.Label(
            blur_container, text=f"{self.camera_blur_radius:.1f}"
        )
        self.blur_value_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.blur_slider = tk.Scale(
            blur_container, from_=0.0, to=5.0, orient=tk.HORIZONTAL,
            command=self.on_blur_change
        )
        self.blur_slider.set(self.camera_blur_radius)
        self.blur_slider.pack(fill=tk.X, padx=5, pady=5)

        fov_container = tk.LabelFrame(settings_frame, text="Field of View")
        fov_container.pack(fill=tk.X, padx=5, pady=5)

        self.fov_value_label = tk.Label(
            fov_container, text=f"{self.cone_angle:.0f}°"
        )
        self.fov_value_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.fov_slider = tk.Scale(
            fov_container, from_=30, to=360, orient=tk.HORIZONTAL,
            command=self.on_fov_change
        )
        self.fov_slider.set(self.cone_angle)
        self.fov_slider.pack(fill=tk.X, padx=5, pady=5)

        visibility_container = tk.LabelFrame(
            settings_frame, text="Visibility Index"
        )
        visibility_container.pack(fill=tk.X, padx=5, pady=5)

        self.visibility_value_label = tk.Label(
            visibility_container, text=f"{self.visibility_index:.2f}"
        )
        self.visibility_value_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.visibility_slider = tk.Scale(
            visibility_container, from_=0.01, to=1.0, resolution=0.01,
            orient=tk.HORIZONTAL, command=self.on_visibility_change
        )
        self.visibility_slider.set(self.visibility_index)
        self.visibility_slider.pack(fill=tk.X, padx=5, pady=5)

        beta_container = tk.LabelFrame(
            settings_frame, text="Beta (Inverse Temp)"
        )
        beta_container.pack(fill=tk.X, padx=5, pady=5)

        self.beta_value_label = tk.Label(
            beta_container, text=f"{self.beta:.1f}"
        )
        self.beta_value_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.beta_slider = tk.Scale(
            beta_container, from_=1.0, to=200.0, orient=tk.HORIZONTAL,
            command=self.on_beta_change
        )
        self.beta_slider.set(self.beta)
        self.beta_slider.pack(fill=tk.X, padx=5, pady=5)

        self.interleaved_rgb_checkbox = tk.Checkbutton(
            settings_frame, text="Interleaved RGB encoding",
            variable=self.interleaved_rgb,
            command=self.on_interleaved_rgb_toggle
        )
        self.interleaved_rgb_checkbox.pack(fill=tk.X, padx=5, pady=5)

        stats_frame = tk.LabelFrame(
            self.right_panel, text="Accuracy Statistics"
        )
        stats_frame.pack(fill=tk.X, padx=5, pady=5)

        self.stats_label = tk.Label(
            stats_frame, text="Train the network to see accuracy statistics",
            font=("Arial", 9),
            justify=tk.LEFT
        )
        self.stats_label.pack(fill=tk.X, padx=5, pady=5)

        self.show_test_positions_checkbox = tk.Checkbutton(
            stats_frame, text="Show test positions",
            variable=self.show_test_positions,
            command=self.on_show_test_positions_toggle
        )
        self.show_test_positions_checkbox.pack(fill=tk.X, padx=5, pady=5)

        self.show_confidence_heatmap_checkbox = tk.Checkbutton(
            stats_frame, text="Show confidence heatmap",
            variable=self.show_confidence_heatmap,
            command=self.on_show_confidence_heatmap_toggle
        )
        self.show_confidence_heatmap_checkbox.pack(fill=tk.X, padx=5, pady=5)

    def on_show_test_positions_toggle(self):
        """Handle checkbox toggle for showing test positions"""
        self.update_map_display()

    def on_show_confidence_heatmap_toggle(self):
        """Handle checkbox toggle for showing confidence heatmap"""
        if self.show_confidence_heatmap.get():
            if not self.heatmap_computed:
                self.compute_confidence_heatmap()
        self.update_map_display()

    def on_blur_change(self, value):
        """Handle blur slider change"""
        self.camera_blur_radius = float(value)
        self.blur_value_label.config(text=f"{self.camera_blur_radius:.1f}")
        self.update_map_display()

    def on_fov_change(self, value):
        """Handle FOV slider change"""
        self.cone_angle = float(value)
        self.fov_value_label.config(text=f"{self.cone_angle:.0f}°")
        self.update_map_display()

    def on_visibility_change(self, value):
        """Handle visibility index slider change"""
        self.visibility_index = float(value)
        self.visibility_value_label.config(text=f"{self.visibility_index:.2f}")
        self.update_map_display()

    def on_beta_change(self, value):
        """Handle beta slider change"""
        self.beta = float(value)
        self.beta_value_label.config(text=f"{self.beta:.1f}")

        if self.hopfield_network is not None:
            self.hopfield_network.beta = self.beta
            if self.is_trained:
                self.update_map_display()

    def on_interleaved_rgb_toggle(self):
        """Handle interleaved RGB checkbox toggle"""
        if self.is_trained:
            self.status_label['text'] = "⚠️ Retraining required: Changing RGB encoding requires retraining. Click 'Sample & Train' again."

    def compute_confidence_heatmap(self):
        """
        Pre-compute confidence values at a dense grid of positions for each direction.
        Creates separate heatmaps for each of the SAMPLE_ROTATIONS angles.
        """
        if not self.is_trained or self.hopfield_network is None:
            self.status_label['text'] = "⚠️ Network not trained. Please use 'Sample & Train' first before computing confidence heatmap."
            self.show_confidence_heatmap.set(False)
            return

        self.heatmap_grid_positions_by_angle = {}
        self.heatmap_grid_confidences_by_angle = {}
        self.heatmap_angles = []

        grid_stride = SAMPLE_STRIDE // 2
        half_stride = grid_stride // 2

        x_positions = list(range(half_stride, MAP_WIDTH, grid_stride))
        y_positions = list(range(half_stride, MAP_HEIGHT, grid_stride))

        test_angles = [i * (360 / SAMPLE_ROTATIONS)
                       for i in range(SAMPLE_ROTATIONS)]
        self.heatmap_angles = test_angles

        original_x = self.robot_x
        original_y = self.robot_y
        original_angle = self.robot_angle

        total_evaluations = len(x_positions) * \
            len(y_positions) * len(test_angles)
        evaluated_count = 0

        self.status_label['text'] = f"Computing heatmap: 0/{total_evaluations}"

        for angle_idx, test_angle in enumerate(test_angles):
            positions_for_angle = []
            confidences_for_angle = []

            for x in x_positions:
                for y in y_positions:
                    self.robot_x = x
                    self.robot_y = y
                    self.robot_angle = test_angle

                    self.capture_camera_view()

                    query_embedding = self.create_embedding(
                        self.current_camera_view)

                    _best_idx, best_weight = self.hopfield_network.retrieve(
                        query_embedding, top_k=1)

                    confidence = best_weight[0]
                    positions_for_angle.append((x, y))
                    confidences_for_angle.append(confidence)

                    evaluated_count += 1

                    if evaluated_count % 1000 == 0 or evaluated_count == total_evaluations:
                        self.status_label['text'] = f"Computing heatmap: {
                            evaluated_count}/{total_evaluations} (angle {angle_idx + 1}/{len(test_angles)})"
                        self.root.update()

            self.heatmap_grid_positions_by_angle[test_angle] = positions_for_angle
            self.heatmap_grid_confidences_by_angle[test_angle] = confidences_for_angle

        self.robot_x = original_x
        self.robot_y = original_y
        self.robot_angle = original_angle

        self.heatmap_computed = True

        self.status_label[
            'text'] = f"✓ Heatmap computed: {total_evaluations} positions across {len(test_angles)} angles."

    def open_map_editor(self):
        MapEditor(self.root, self.current_map_image, self.on_map_saved)

    def on_map_saved(self, new_map_image):
        self.current_map_image = new_map_image
        self.update_map_display()

    def load_robot_image(self):
        """Load the robot image from resources"""
        try:
            robot_image_path = os.path.join(
                os.path.dirname(__file__), "resources", "uii.png")
            self.robot_image = Image.open(robot_image_path)
            if self.robot_image.mode != "RGBA":
                self.robot_image = self.robot_image.convert("RGBA")
            print(f"Loaded robot image: {robot_image_path}")
        except Exception as e:
            print(f"Failed to load robot image: {e}")
            self.robot_image = None

    def load_last_map(self):
        """Load the last used map automatically on startup"""
        last_path = self.config.get_last_map_path()

        if last_path and os.path.exists(last_path):
            try:
                imported_image = Image.open(last_path)

                if imported_image.size != (MAP_WIDTH, MAP_HEIGHT):
                    imported_image = imported_image.resize(
                        (MAP_WIDTH, MAP_HEIGHT),
                        Image.Resampling.LANCZOS
                    )

                if imported_image.mode != "RGB":
                    imported_image = imported_image.convert("RGB")

                self.current_map_image = imported_image
                print(f"Loaded last map: {last_path}")

            except Exception as e:
                print(f"Failed to load last map: {e}")

    def import_map(self):
        """Import a map from a PNG file"""
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
                imported_image = Image.open(file_path)

                if imported_image.size != (MAP_WIDTH, MAP_HEIGHT):
                    imported_image = imported_image.resize(
                        (MAP_WIDTH, MAP_HEIGHT),
                        Image.Resampling.LANCZOS
                    )

                if imported_image.mode != "RGB":
                    imported_image = imported_image.convert("RGB")

                self.current_map_image = imported_image
                self.update_map_display()

                self.config.set_last_map_path(file_path)

                self.status_label[
                    'text'] = f"✓ Map imported successfully from {os.path.basename(file_path)}"

            except Exception as e:
                self.status_label['text'] = f"❌ Failed to import map: {str(e)}"

    def export_map(self):
        """Export the current map to a PNG file"""
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

                self.config.set_last_map_path(file_path)

                self.status_label['text'] = f"✓ Map exported to {
                    os.path.basename(file_path)} "
            except Exception as e:
                self.status_label['text'] = f"❌ Failed to export map: {str(e)}"

    def sample_and_train(self):
        """
        Automatically generate samples and train the network.
        """
        self.auto_sample(show_message=False)

        if len(self.sample_embeddings) > 0:
            self.train_network(show_message=False)

            self.status_label['text'] = f"✓ Generated {
                len(self.sample_positions)}  samples and trained network with {
                len(self.sample_embeddings)}  patterns"

    def auto_sample(self, show_message=True):
        """
        Automatically generate samples from the current map.
        Samples at discrete positions with SAMPLE_STRIDE spacing and
        SAMPLE_ROTATIONS rotations at each position.

        Args:
            show_message: Whether to show success message (default True)
        """
        self.sample_positions = []
        self.sample_embeddings = []
        self.sample_views = []
        self.clear_sample_dots()

        half_stride = SAMPLE_STRIDE // 2
        x_positions = list(range(half_stride, MAP_WIDTH, SAMPLE_STRIDE))
        y_positions = list(range(half_stride, MAP_HEIGHT, SAMPLE_STRIDE))

        angles = [i * (360 / SAMPLE_ROTATIONS)
                  for i in range(SAMPLE_ROTATIONS)]

        original_x = self.robot_x
        original_y = self.robot_y
        original_angle = self.robot_angle

        total_samples = len(x_positions) * len(y_positions) * len(angles)
        sample_count = 0

        self.status_label['text'] = f"Sampling: 0/{total_samples}"

        for x in x_positions:
            for y in y_positions:
                for angle in angles:
                    self.robot_x = x
                    self.robot_y = y
                    self.robot_angle = angle

                    self.capture_camera_view()

                    embedding = self.create_embedding(self.current_camera_view)

                    self.sample_positions.append((x, y, angle))
                    self.sample_embeddings.append(embedding)
                    self.sample_views.append(
                        self.current_camera_view.copy())

                    sample_count += 1

                    if SAMPLE_UPDATE_FREQUENCY is not None and (
                            sample_count % SAMPLE_UPDATE_FREQUENCY == 0
                            or sample_count == total_samples):
                        self.status_label['text'] = f"Sampling: {
                            sample_count}/{total_samples}"

                        self.update_map_display_during_sampling()

                        self.root.update()

        self.robot_x = original_x
        self.robot_y = original_y
        self.robot_angle = original_angle

        self.update_map_display()
        self.draw_sample_dots()

        self.status_label['text'] = ""

        if show_message:
            self.status_label['text'] = f"✓ Generated {
                len(self.sample_positions)}  samples"

    def train_network(self, show_message=True):
        """
        Train the Modern Hopfield Network using the collected samples.

        Args:
            show_message: Whether to show success message (default True)
        """
        if len(self.sample_embeddings) == 0:
            self.status_label['text'] = "❌ No samples available. Please run 'Sample & Train' first."
            return

        embedding_dim = len(self.sample_embeddings[0])

        self.hopfield_network = ModernHopfieldNetwork(
            embedding_dim, beta=self.beta)

        def update_training_progress(current, total):
            self.status_label['text'] = f"Training: {current}/{total}"
            self.root.update()

        self.status_label['text'] = f"Training: 0/{
            len(self.sample_embeddings)} "

        try:
            self.hopfield_network.train(
                self.sample_embeddings,
                progress_callback=update_training_progress
            )

            self.is_trained = True

            self.heatmap_computed = False
            self.heatmap_grid_positions_by_angle = {}
            self.heatmap_grid_confidences_by_angle = {}
            self.heatmap_angles = []

            self.evaluate_accuracy()

            if show_message:
                self.status_label['text'] = f"✓ Network trained with {len(self.sample_embeddings)} patterns. Avg distance error: {self.accuracy_avg_distance:.2f} px"
            else:
                self.status_label['text'] = ""

        except Exception as e:
            self.status_label['text'] = f"❌ Training error: {str(e)}"

    def evaluate_accuracy(self, num_tests=100):
        """
        Evaluate the accuracy of the trained network by testing on random positions.
        Computes the average distance between predicted and actual closest sample positions.

        Args:
            num_tests: Number of random test positions (default: 100)
        """
        if not self.is_trained or self.hopfield_network is None:
            return

        original_x = self.robot_x
        original_y = self.robot_y
        original_angle = self.robot_angle

        self.status_label['text'] = f"Evaluating accuracy: 0/{num_tests}"

        total_distance_error = 0.0
        valid_tests = 0

        self.test_positions = []

        for i in range(num_tests):
            test_x = np.random.randint(0, MAP_WIDTH)
            test_y = np.random.randint(0, MAP_HEIGHT)
            test_angle = np.random.uniform(0, 360)

            self.robot_x = test_x
            self.robot_y = test_y
            self.robot_angle = test_angle

            self.capture_camera_view()

            query_embedding = self.create_embedding(self.current_camera_view)

            best_idx, _best_weight = self.hopfield_network.retrieve(
                query_embedding, top_k=1)
            best_idx = best_idx[0]

            pred_x, pred_y, _pred_angle = self.sample_positions[best_idx]

            min_distance = float('inf')
            closest_x = test_x
            closest_y = test_y
            for sample_x, sample_y, sample_angle in self.sample_positions:
                pos_distance = math.sqrt(
                    (sample_x - test_x) ** 2 + (sample_y - test_y) ** 2)
                angle_diff = abs(sample_angle - test_angle)
                angle_diff = min(angle_diff, 360 - angle_diff)

                combined_distance = pos_distance + (angle_diff / 180.0) * 10

                if combined_distance < min_distance:
                    min_distance = combined_distance
                    closest_x = sample_x
                    closest_y = sample_y

            distance_error = math.sqrt(
                (pred_x - closest_x) ** 2 + (pred_y - closest_y) ** 2)
            total_distance_error += distance_error
            valid_tests += 1

            self.test_positions.append((test_x, test_y))

            self.status_label['text'] = f"Evaluating accuracy: {
                i + 1} /{num_tests} "
            self.root.update()

        self.robot_x = original_x
        self.robot_y = original_y
        self.robot_angle = original_angle

        if valid_tests > 0:
            self.accuracy_avg_distance = total_distance_error / valid_tests
            self.accuracy_num_tests = valid_tests

            self.update_statistics_display()

        self.status_label['text'] = ""

    def update_statistics_display(self):
        """Update the statistics display with accuracy information"""
        if self.accuracy_avg_distance is not None:
            stats_text = f"Test Positions: {self.accuracy_num_tests}\n"
            stats_text += f"Avg Distance Error: {
                self.accuracy_avg_distance: .2f}  pixels"
            self.stats_label.config(text=stats_text)
        else:
            self.stats_label.config(
                text="Train the network to see accuracy statistics")

    def create_embedding(self, camera_view):
        """
        Create an embedding vector from a camera view image.
        For MVP, we simply flatten the RGB values.

        Args:
            camera_view: PIL Image (1D strip)

        Returns:
            numpy array representing the embedding
        """
        pixels = list(camera_view.getdata())

        if self.interleaved_rgb.get():
            embedding = []
            for r, g, b in pixels:
                embedding.extend([r, g, b])
        else:
            r_channel = [pixel[0] for pixel in pixels]
            g_channel = [pixel[1] for pixel in pixels]
            b_channel = [pixel[2] for pixel in pixels]
            embedding = r_channel + g_channel + b_channel

        return np.array(embedding, dtype=np.float32)

    def clear_sample_dots(self):
        """Remove all sample dots from the canvas"""
        for dot_id in self.sample_dots:
            self.map_canvas.delete(dot_id)
        self.sample_dots = []

    def draw_sample_dots(self):
        """Draw red dots at sample positions on the canvas, sized by similarity"""
        self.clear_sample_dots()

        max_similarity = None
        if self.sample_similarities is not None and len(
                self.sample_similarities) > 0:
            max_similarity = max(self.sample_similarities.max(), 1e-6)

        for i, (x, y, _angle) in enumerate(self.sample_positions):
            if self.sample_similarities is not None and i < len(
                    self.sample_similarities):
                similarity = self.sample_similarities[i]
                min_radius = 1
                max_radius = 5

                normalized_similarity = similarity / max_similarity
                dot_radius = min_radius + (
                    max_radius - min_radius) * normalized_similarity
            else:
                dot_radius = SAMPLE_DOT_RADIUS

            canvas_x, canvas_y = self.image_to_canvas_coords(x, y)
            scaled_radius = dot_radius * self.map_scale_factor

            dot_id = self.map_canvas.create_rectangle(
                canvas_x - scaled_radius,
                canvas_y - scaled_radius,
                canvas_x + scaled_radius,
                canvas_y + scaled_radius,
                fill=COLOR_SAMPLE_DOT,
                outline=COLOR_SAMPLE_DOT,
                tags=("sample_dot", f"sample_{i}")
            )
            self.sample_dots.append(dot_id)

    def clear_test_position_dots(self):
        """Remove all test position dots from the canvas"""
        for dot_id in self.test_position_dots:
            self.map_canvas.delete(dot_id)
        self.test_position_dots = []

    def draw_test_position_dots(self):
        """Draw blue squares at test positions on the canvas"""
        self.clear_test_position_dots()

        if not self.show_test_positions.get():
            return

        dot_radius = 2

        for x, y in self.test_positions:
            canvas_x, canvas_y = self.image_to_canvas_coords(x, y)
            scaled_radius = dot_radius * self.map_scale_factor

            dot_id = self.map_canvas.create_rectangle(
                canvas_x - scaled_radius,
                canvas_y - scaled_radius,
                canvas_x + scaled_radius,
                canvas_y + scaled_radius,
                fill="#0000FF",
                outline="#0000FF",
                tags="test_position_dot"
            )
            self.test_position_dots.append(dot_id)

    def on_map_canvas_resize(self, event):
        """Handle map canvas resize and update display"""
        self.update_map_display()

    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to image coordinates"""
        img_x = (canvas_x - self.map_offset_x) / self.map_scale_factor
        img_y = (canvas_y - self.map_offset_y) / self.map_scale_factor
        return img_x, img_y

    def image_to_canvas_coords(self, img_x, img_y):
        """Convert image coordinates to canvas coordinates"""
        canvas_x = img_x * self.map_scale_factor + self.map_offset_x
        canvas_y = img_y * self.map_scale_factor + self.map_offset_y
        return canvas_x, canvas_y

    def on_canvas_hover(self, event):
        """
        Handle mouse hover over the canvas.
        Display the saved sample when hovering over a red dot.
        """
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)

        closest_idx = None
        min_distance = 15

        max_similarity = None
        if self.sample_similarities is not None and len(
                self.sample_similarities) > 0:
            max_similarity = max(self.sample_similarities.max(), 1e-6)

        for i, (x, y, _angle) in enumerate(self.sample_positions):
            distance = math.sqrt((x - img_x)**2 + (y - img_y)**2)
            if self.sample_similarities is not None and i < len(
                    self.sample_similarities):
                similarity = self.sample_similarities[i]
                min_radius = 1
                max_radius = 5

                normalized_similarity = similarity / max_similarity
                dot_radius = min_radius + (
                    max_radius - min_radius) * normalized_similarity

                hover_threshold = dot_radius + 5
            else:
                hover_threshold = SAMPLE_DOT_RADIUS + 5

            if distance < hover_threshold and distance < min_distance:
                min_distance = distance
                closest_idx = i

        if closest_idx != self.hovered_sample_idx:
            self.hovered_sample_idx = closest_idx

            if closest_idx is not None:
                self.display_saved_sample(closest_idx)
            else:
                self.memory_canvas.delete("all")
                self.memory_frame.config(text="Retrieved Memory")

    def update_map_display_during_sampling(self):
        """Update map display during sampling to show robot position and camera view"""
        display_image = self.current_map_image.copy()

        self.draw_viewing_cone_on_image(display_image)

        self._display_scaled_map_image(display_image)

        self.draw_sample_dots()

        self.draw_robot()

        self.display_camera_view()

    def _display_scaled_map_image(self, display_image):
        """Scale and display the map image to fit canvas width while maintaining aspect ratio"""
        canvas_width = self.map_canvas.winfo_width()
        canvas_height = self.map_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            self.tk_map_image = ImageTk.PhotoImage(display_image)
            self.map_canvas.delete("all")
            self.map_canvas.create_image(
                0, 0, image=self.tk_map_image, anchor="nw")
            return

        img_width, img_height = display_image.size
        width_scale = canvas_width / img_width
        height_scale = canvas_height / img_height

        self.map_scale_factor = min(width_scale, height_scale)

        new_width = int(img_width * self.map_scale_factor)
        new_height = int(img_height * self.map_scale_factor)

        self.map_offset_x = (canvas_width - new_width) / 2
        self.map_offset_y = (canvas_height - new_height) / 2

        scaled_image = display_image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS)
        self.tk_map_image = ImageTk.PhotoImage(scaled_image)

        self.map_canvas.delete("all")
        self.map_canvas.create_image(
            self.map_offset_x, self.map_offset_y, image=self.tk_map_image,
            anchor="nw")

    def update_map_display(self):
        display_image = self.current_map_image.copy()

        self.draw_viewing_cone_on_image(display_image)

        if self.show_confidence_heatmap.get() and self.sample_similarities is not None:
            self.draw_confidence_heatmap_on_image(display_image)

        self._display_scaled_map_image(display_image)

        self.capture_camera_view()
        self.display_camera_view()

        if self.is_trained:
            self.localize()

        self.draw_sample_dots()

        self.draw_test_position_dots()

        self.draw_robot()

    def draw_robot(self):
        canvas_robot_x, canvas_robot_y = self.image_to_canvas_coords(
            self.robot_x, self.robot_y)
        scaled_radius = self.robot_radius * self.map_scale_factor

        if self.robot_image is not None:
            rotated_image = self.robot_image.rotate(
                -self.robot_angle, expand=True)

            robot_size = int(50 * self.map_scale_factor)
            rotated_image = rotated_image.resize(
                (robot_size, robot_size),
                Image.Resampling.LANCZOS)

            self.tk_robot_image = ImageTk.PhotoImage(rotated_image)

            self.map_canvas.create_image(
                canvas_robot_x, canvas_robot_y,
                image=self.tk_robot_image,
                anchor="center",
                tags="robot"
            )
        else:
            self.map_canvas.create_oval(
                canvas_robot_x - scaled_radius,
                canvas_robot_y - scaled_radius,
                canvas_robot_x + scaled_radius,
                canvas_robot_y + scaled_radius,
                fill=COLOR_ROBOT_GT,
                outline=COLOR_ROBOT_GT,
                tags="robot"
            )

        if self.estimated_x is not None and self.estimated_y is not None:
            canvas_est_x, canvas_est_y = self.image_to_canvas_coords(
                self.estimated_x, self.estimated_y)

            if self.estimated_angle is not None:
                line_length = 20 * self.map_scale_factor
                angle_rad = math.radians(self.estimated_angle)
                end_x = canvas_est_x + line_length * math.cos(angle_rad)
                end_y = canvas_est_y + line_length * math.sin(angle_rad)
                self.map_canvas.create_line(
                    canvas_est_x, canvas_est_y,
                    end_x, end_y,
                    fill="#800080",
                    width=max(2, int(2 * self.map_scale_factor)),
                    tags="estimated_direction"
                )

            self.map_canvas.create_oval(
                canvas_est_x - scaled_radius,
                canvas_est_y - scaled_radius,
                canvas_est_x + scaled_radius,
                canvas_est_y + scaled_radius,
                fill="#00FF00",
                outline="#00FF00",
                tags="estimated"
            )

    def draw_confidence_heatmap_on_image(self, image):
        """Draw a smooth 2D color gradient heatmap using HeatmapBuilder"""
        if not self.heatmap_computed or not self.heatmap_angles:
            return

        closest_angle = self._find_closest_heatmap_angle(self.robot_angle)

        if closest_angle is None:
            return

        grid_positions = self.heatmap_grid_positions_by_angle.get(
            closest_angle, [])
        grid_confidences = self.heatmap_grid_confidences_by_angle.get(
            closest_angle, [])

        if not grid_positions or not grid_confidences:
            return

        # Use lower resolution for faster computation
        # The result is upscaled smoothly to full size
        builder = HeatmapBuilder(
            map_width=MAP_WIDTH,
            map_height=MAP_HEIGHT,
            grid_stride=SAMPLE_STRIDE // 2,
            resolution_scale=HEATMAP_RESOLUTION_SCALE
        )

        heatmap_image = builder.build_heatmap(
            grid_positions=grid_positions,
            grid_confidences=grid_confidences,
            colormap_name='jet',
            sigma=20.0,
            alpha_base=100,
            alpha_scale=155,
            threshold=0.001
        )

        if heatmap_image is not None:
            image.paste(heatmap_image, (0, 0), heatmap_image)

    def _find_closest_heatmap_angle(self, current_angle):
        """
        Find the closest precomputed heatmap angle to the current robot angle.

        Args:
            current_angle: Current robot angle in degrees (0-360)

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

    def draw_viewing_cone_on_image(self, image):
        """Draw the viewing cone with 50% transparency directly on a PIL image"""
        half_cone = self.cone_angle / 2

        points = [(self.robot_x, self.robot_y)]

        num_points = 50
        for i in range(num_points + 1):
            angle_offset = -half_cone + (self.cone_angle * i / num_points)
            current_angle = math.radians(self.robot_angle + angle_offset)

            dx = math.cos(current_angle)
            dy = math.sin(current_angle)

            distance = self.cast_ray(
                self.robot_x, self.robot_y, dx, dy)

            x = self.robot_x + distance * dx
            y = self.robot_y + distance * dy
            points.append((x, y))

        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        cone_color = (0, 0, 255, 127)

        draw.polygon(points, fill=cone_color, outline=None)

        image.paste(overlay, (0, 0), overlay)

    def get_distance_to_edge(self, x, y, dx, dy):
        """Calculate the distance from (x, y) to the map edge in direction (dx, dy)"""
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

    def cast_ray(self, x, y, dx, dy):
        """
        Cast a ray from (x, y) in direction (dx, dy) until hitting a wall or map edge.
        Returns the distance to the first obstacle.
        """
        max_distance = self.get_distance_to_edge(x, y, dx, dy)

        step_size = 5.0

        pixels = self.current_map_image.load()

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

    def capture_camera_view(self):
        """
        Capture a 1D strip of pixels from what the robot sees.
        Samples pixels along rays cast from the robot's viewing direction.
        Walls appear less opaque (more white) the further they are from the robot.
        """
        half_cone = self.cone_angle / 2
        pixels = self.current_map_image.load()

        camera_strip = []

        max_distance = self.cone_length

        for i in range(self.camera_samples):
            angle_offset = -half_cone + (self.cone_angle * i /
                                         (self.camera_samples - 1))
            current_angle = math.radians(self.robot_angle + angle_offset)

            dx = math.cos(current_angle)
            dy = math.sin(current_angle)

            distance = self.cast_ray(self.robot_x, self.robot_y, dx, dy)

            end_x = self.robot_x + distance * dx
            end_y = self.robot_y + distance * dy

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

        self.current_camera_view = Image.new('RGB', (self.camera_samples, 1))
        self.current_camera_view.putdata(camera_strip)

        if self.camera_blur_radius > 0:
            temp_height = 10
            temp_view = self.current_camera_view.resize(
                (self.camera_samples, temp_height),
                Image.Resampling.NEAREST
            )
            temp_view = temp_view.filter(ImageFilter.GaussianBlur(
                radius=self.camera_blur_radius))
            self.current_camera_view = temp_view.resize(
                (self.camera_samples, 1),
                Image.Resampling.BILINEAR
            )

    def display_camera_view(self):
        """Display the camera view in the input canvas"""
        if self.current_camera_view is None:
            return

        canvas_width = self.input_canvas.winfo_width()
        canvas_height = self.input_canvas.winfo_height()

        if canvas_width <= 1:
            canvas_width = 300
            canvas_height = 50

        scaled_image = self.current_camera_view.resize(
            (canvas_width, canvas_height),
            Image.Resampling.NEAREST
        )

        tk_image = ImageTk.PhotoImage(scaled_image)

        self.input_canvas.delete("all")
        self.input_canvas.create_image(0, 0, image=tk_image, anchor="nw")

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
            query_embedding = self.create_embedding(self.current_camera_view)

            all_indices, all_weights = self.hopfield_network.retrieve(
                query_embedding, top_k=len(self.sample_embeddings))

            self.sample_similarities = np.zeros(len(self.sample_embeddings))
            for idx, weight in zip(all_indices, all_weights):
                self.sample_similarities[idx] = weight

            best_idx = all_indices[0]
            best_weight = all_weights[0]

            self.retrieved_sample_idx = best_idx

            x, y, angle = self.sample_positions[best_idx]
            self.estimated_x = x
            self.estimated_y = y
            self.estimated_angle = angle

            self.display_retrieved_memory(best_idx, best_weight)

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

        canvas_width = self.memory_canvas.winfo_width()
        canvas_height = self.memory_canvas.winfo_height()

        if canvas_width <= 1:
            canvas_width = 300
            canvas_height = 50

        scaled_image = saved_view.resize(
            (canvas_width, canvas_height),
            Image.Resampling.NEAREST
        )

        tk_image = ImageTk.PhotoImage(scaled_image)

        self.memory_canvas.delete("all")
        self.memory_canvas.create_image(0, 0, image=tk_image, anchor="nw")

        x, y, angle = self.sample_positions[sample_idx]
        info_text = f"({x}, {y}), {angle: .1f} ° | Weight: {weight: .4f} "
        self.memory_frame.config(text=f"Retrieved Memory: {info_text}")

        self.memory_canvas.image = tk_image

    def display_similarity_metric(self, weight):
        """
        Display the similarity metric as a progress bar.

        Args:
            weight: Attention weight/confidence (0 to 1)
        """
        percentage = weight * 100
        self.sim_progressbar['value'] = percentage
        self.sim_label.config(text=f"Confidence: {percentage:.1f}%")

    def display_saved_sample(self, sample_idx):
        """
        Display a saved sample view in the retrieved memory canvas.

        Args:
            sample_idx: Index of the sample to display
        """
        if sample_idx < 0 or sample_idx >= len(self.sample_views):
            return

        saved_view = self.sample_views[sample_idx]

        canvas_width = self.memory_canvas.winfo_width()
        canvas_height = self.memory_canvas.winfo_height()

        if canvas_width <= 1:
            canvas_width = 300
            canvas_height = 50

        scaled_image = saved_view.resize(
            (canvas_width, canvas_height),
            Image.Resampling.NEAREST  # Use nearest neighbor to keep sharp pixels
        )

        tk_image = ImageTk.PhotoImage(scaled_image)

        self.memory_canvas.delete("all")
        self.memory_canvas.create_image(0, 0, image=tk_image, anchor="nw")

        x, y, angle = self.sample_positions[sample_idx]
        info_text = f"Pos: ({x}, {y}), Angle: {angle:.1f}°"
        self.memory_frame.config(text=f"Retrieved Memory: {info_text}")

        self.memory_canvas.image = tk_image

    def move_robot(self, dx, dy):
        """Update robot position"""
        self.robot_x += dx
        self.robot_y += dy

        self.robot_x = max(self.robot_radius, min(
            MAP_WIDTH - self.robot_radius, self.robot_x))
        self.robot_y = max(self.robot_radius, min(
            MAP_HEIGHT - self.robot_radius, self.robot_y))

        self.update_map_display()

    def rotate_robot(self, angle_delta):
        """Update robot angle by the specified delta (already a discrete step)"""
        self.robot_angle += angle_delta
        self.robot_angle = self.robot_angle % 360

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

        if 'j' in self.rotation_keys_pressed:
            d_angle -= self.robot_rotation_speed
        if 'l' in self.rotation_keys_pressed:
            d_angle += self.robot_rotation_speed

        if dx != 0 or dy != 0:
            self.move_robot(dx, dy)

        if d_angle != 0:
            self.rotate_robot(d_angle)

        self.root.after(self.update_interval, self.update_loop)

    def run(self):
        self.update_map_display()
        self.update_loop()
        self.root.mainloop()


if __name__ == "__main__":
    app = App()
    app.run()
