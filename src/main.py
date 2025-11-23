import tkinter as tk
from tkinter import filedialog, ttk
from constants import *
from PIL import Image, ImageTk, ImageDraw
from editor import MapEditor
from config import ConfigManager
from robot_state import RobotState
from camera_simulator import CameraSimulator
from localization_engine import LocalizationEngine
from sampling_engine import SamplingEngine
from confidence_analyzer import ConfidenceAnalyzer
import math
import os


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Robot Localization via Modern Hopfield Networks")
        self.root.geometry(f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}")

        self.config = ConfigManager()

        # Logic components (separated from UI)
        self.robot = RobotState()
        self.camera = CameraSimulator(
            cone_angle=CAMERA_FOV,
            cone_length=40,
            camera_samples=100,
            blur_radius=CAMERA_BLUR_RADIUS,
            visibility_index=0.1
        )
        self.localization = LocalizationEngine(
            beta=DEFAULT_BETA,
            interleaved_rgb=INTERLEAVED_RGB
        )
        self.sampling = SamplingEngine()
        self.confidence = ConfidenceAnalyzer(MAP_WIDTH, MAP_HEIGHT)

        # UI-related state
        self.robot_image = None
        self.tk_robot_image = None
        self.load_robot_image()

        self.current_map_image = Image.new(
            "RGB", (MAP_WIDTH, MAP_HEIGHT), "white")
        self.map_with_noise = None
        self.tk_map_image = None

        self.interleaved_rgb = tk.BooleanVar(value=INTERLEAVED_RGB)

        self.noise_amount = DEFAULT_NOISE_AMOUNT
        self.apply_noise = tk.BooleanVar(value=False)
        self.noise_circles = []

        self.current_camera_view = None

        self.sample_dots = []

        self.estimated_x = None
        self.estimated_y = None
        self.estimated_angle = None
        self.retrieved_sample_idx = None
        self.top_k_matches = []

        self.test_position_dots = []
        self.show_test_positions = tk.BooleanVar(value=False)

        self.show_confidence_heatmap = tk.BooleanVar(value=False)
        self.average_heatmap = tk.BooleanVar(value=False)

        self.top_k = DEFAULT_TOP_K

        self.hovered_sample_idx = None

        self.keys_pressed = set()
        self.update_interval = 16

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
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        self.toolbar_frame = ttk.Frame(self.main_container)
        self.toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        self.content_frame = ttk.Frame(self.main_container)
        self.content_frame.pack(side=tk.TOP, fill=tk.BOTH,
                                expand=True, padx=5)

        self.status_frame = ttk.Frame(self.main_container)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.status_label = ttk.Label(
            self.status_frame, text="Press Sample & Train!", anchor="w")
        self.status_label.pack(side=tk.LEFT, fill=tk.X,
                               expand=True, padx=5, pady=2)

    def create_toolbar(self):
        buttons = {
            "Edit Map": self.open_map_editor,
            "Import Map": self.import_map,
            "Export Map": self.export_map,
            "Sample & Train": self.sample_and_train
        }

        for btn_text, command in buttons.items():
            btn = ttk.Button(self.toolbar_frame,
                             text=btn_text, command=command)
            btn.pack(side=tk.LEFT, padx=5)

    def create_left_panel(self):
        self.left_panel = ttk.Frame(self.content_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH,
                             expand=True, padx=5)

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
        right_panel_container = ttk.Frame(self.content_frame, width=400)
        right_panel_container.pack(
            side=tk.LEFT, fill=tk.Y, expand=False, padx=(5, 10))
        right_panel_container.pack_propagate(False)

        # Add a canvas and scrollbar for scrolling
        canvas = tk.Canvas(right_panel_container)
        scrollbar = ttk.Scrollbar(
            right_panel_container, orient=tk.VERTICAL, command=canvas.yview)
        self.right_panel = ttk.Frame(canvas)

        self.right_panel.bind(
            "<Configure>", lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=self.right_panel, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        spacer_width = 100
        spacer_frame = ttk.Frame(self.right_panel, width=spacer_width)
        spacer_frame.pack(side=tk.RIGHT, fill=tk.Y)

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

        # Align all label texts to the left side
        self.sim_label = ttk.Label(
            sim_frame, text="Confidence: 0.0%", anchor="w"
        )
        self.sim_label.pack(fill=tk.X, padx=5, pady=5)

        self.sim_progressbar = ttk.Progressbar(sim_frame, orient='horizontal', mode='determinate', maximum=100)
        self.sim_progressbar.pack(fill=tk.X, padx=5, pady=5)

        settings_frame = tk.LabelFrame(
            self.right_panel, text="Settings"
        )
        settings_frame.pack(fill=tk.X, padx=5, pady=5)

        blur_container = tk.LabelFrame(settings_frame, text="Blur Radius")
        blur_container.pack(fill=tk.X, padx=5, pady=5)

        self.blur_value_label = ttk.Label(
            blur_container, text=f"{self.camera.blur_radius:.1f}", anchor="w"
        )
        self.blur_value_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.blur_slider = ttk.Scale(
            blur_container, from_=0.0, to=5.0, orient=tk.HORIZONTAL,
            command=self.on_blur_change
        )
        self.blur_slider.set(self.camera.blur_radius)
        self.blur_slider.pack(fill=tk.X, padx=5, pady=5)

        fov_container = tk.LabelFrame(settings_frame, text="Field of View")
        fov_container.pack(fill=tk.X, padx=5, pady=5)

        self.fov_value_label = ttk.Label(
            fov_container, text=f"{self.camera.cone_angle:.0f}°", anchor="w"
        )
        self.fov_value_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.fov_slider = ttk.Scale(
            fov_container, from_=30, to=360, orient=tk.HORIZONTAL,
            command=self.on_fov_change
        )
        self.fov_slider.set(self.camera.cone_angle)
        self.fov_slider.pack(fill=tk.X, padx=5, pady=5)

        visibility_container = tk.LabelFrame(
            settings_frame, text="Visibility Index"
        )
        visibility_container.pack(fill=tk.X, padx=5, pady=5)

        self.visibility_value_label = ttk.Label(
            visibility_container, text=f"{self.camera.visibility_index:.2f}", anchor="w"
        )
        self.visibility_value_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.visibility_slider = ttk.Scale(
            visibility_container, from_=0.01, to=1.0,
            orient=tk.HORIZONTAL, command=self.on_visibility_change
        )
        self.visibility_slider.set(self.camera.visibility_index)
        self.visibility_slider.pack(fill=tk.X, padx=5, pady=5)

        beta_container = tk.LabelFrame(
            settings_frame, text="Beta (Inverse Temp)"
        )
        beta_container.pack(fill=tk.X, padx=5, pady=5)

        self.beta_value_label = ttk.Label(
            beta_container, text=f"{DEFAULT_BETA:.1f}", anchor="w"
        )
        self.beta_value_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.beta_slider = ttk.Scale(
            beta_container, from_=1.0, to=200.0, orient=tk.HORIZONTAL,
            command=self.on_beta_change
        )
        self.beta_slider.set(DEFAULT_BETA)
        self.beta_slider.pack(fill=tk.X, padx=5, pady=5)

        top_k_container = tk.LabelFrame(
            settings_frame, text="Combine top k matches"
        )
        top_k_container.pack(fill=tk.X, padx=5, pady=5)

        self.top_k_value_label = ttk.Label(
            top_k_container, text=f"{DEFAULT_TOP_K}", anchor="w"
        )
        self.top_k_value_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.top_k_slider = ttk.Scale(
            top_k_container, from_=1, to=20, orient=tk.HORIZONTAL,
            command=self.on_top_k_change
        )
        self.top_k_slider.set(DEFAULT_TOP_K)
        self.top_k_slider.pack(fill=tk.X, padx=5, pady=5)

        noise_container = tk.LabelFrame(
            settings_frame, text="Noise Settings"
        )
        noise_container.pack(fill=tk.X, padx=5, pady=5)

        self.noise_value_label = ttk.Label(
            noise_container, text=f"{int(self.noise_amount)} objects", anchor="w"
        )
        self.noise_value_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.noise_slider = ttk.Scale(
            noise_container, from_=0, to=100, orient=tk.HORIZONTAL,
            command=self.on_noise_change
        )
        self.noise_slider.set(self.noise_amount)
        self.noise_slider.pack(fill=tk.X, padx=5, pady=5)

        self.apply_noise_checkbox = ttk.Checkbutton(
            noise_container, text="Apply noise to map",
            variable=self.apply_noise,
            command=self.on_apply_noise_toggle
        )
        self.apply_noise_checkbox.pack(fill=tk.X, padx=5, pady=5)

        # Interleaved RGB checkbox (left-aligned)
        self.interleaved_rgb_checkbox = ttk.Checkbutton(
            settings_frame, text="Interleaved RGB encoding",
            variable=self.interleaved_rgb,
            command=self.on_interleaved_rgb_toggle
        )
        self.interleaved_rgb_checkbox.pack(fill=tk.X, padx=5, pady=5)

        # Confidence statistics frame (contains computation-position and heatmap checkboxes)
        stats_frame = tk.LabelFrame(
            self.right_panel, text="Confidence Statistics"
        )
        stats_frame.pack(fill=tk.X, padx=5, pady=5)

        self.stats_label = ttk.Label(
            stats_frame, text="Train the network to see confidence statistics",
            font=("Arial", 9),
            anchor="w"
        )
        self.stats_label.pack(fill=tk.X, padx=5, pady=5)

        # Align checkbox labels to the left and pack them into stats_frame
        # Checkbox: show the positions used during the confidence computation
        self.show_test_positions_checkbox = ttk.Checkbutton(
            stats_frame, text="Show confidence computation positions",
            variable=self.show_test_positions,
            command=self.on_show_test_positions_toggle
        )
        self.show_test_positions_checkbox.pack(fill=tk.X, padx=5, pady=5)

        self.show_confidence_heatmap_checkbox = ttk.Checkbutton(
            stats_frame, text="Show confidence heatmap",
            variable=self.show_confidence_heatmap,
            command=self.on_show_confidence_heatmap_toggle
        )
        self.show_confidence_heatmap_checkbox.pack(fill=tk.X, padx=5, pady=5)

        self.average_heatmap_checkbox = ttk.Checkbutton(
            stats_frame, text="Average heatmap across all angles",
            variable=self.average_heatmap,
            command=self.on_average_heatmap_toggle
        )
        self.average_heatmap_checkbox.pack(fill=tk.X, padx=5, pady=5)

    def on_show_test_positions_toggle(self):
        """Handle checkbox toggle for showing test positions"""
        self.update_map_display()

    def on_show_confidence_heatmap_toggle(self):
        """Handle checkbox toggle for showing confidence heatmap"""
        if self.show_confidence_heatmap.get():
            if not self.confidence.heatmap_computed:
                self.compute_confidence_heatmap()
        self.update_map_display()

    def on_average_heatmap_toggle(self):
        """Handle checkbox toggle for averaging heatmap across all angles"""
        self.update_map_display()

    def on_blur_change(self, value):
        """Handle blur slider change"""
        self.camera.blur_radius = float(value)
        self.blur_value_label.config(text=f"{self.camera.blur_radius:.1f}")
        self.update_map_display()

    def on_fov_change(self, value):
        """Handle FOV slider change"""
        self.camera.cone_angle = float(value)
        self.fov_value_label.config(text=f"{self.camera.cone_angle:.0f}°")
        self.update_map_display()

    def on_visibility_change(self, value):
        """Handle visibility index slider change"""
        self.camera.visibility_index = float(value)
        self.visibility_value_label.config(
            text=f"{self.camera.visibility_index:.2f}")
        self.update_map_display()

    def on_beta_change(self, value):
        """Handle beta slider change"""
        beta = float(value)
        self.beta_value_label.config(text=f"{beta:.1f}")
        self.localization.update_beta(beta)
        if self.localization.is_trained:
            self.update_map_display()

    def on_top_k_change(self, value):
        """Handle top-k slider change"""
        self.top_k = int(float(value))
        self.top_k_value_label.config(text=f"{self.top_k}")
        if self.localization.is_trained:
            self.localize()
            self.update_map_display()

    def on_interleaved_rgb_toggle(self):
        """Handle interleaved RGB checkbox toggle"""
        if self.localization.is_trained:
            self.status_label['text'] = "⚠️ Retraining required: Changing RGB encoding requires retraining. Click 'Sample & Train' again."

    def on_noise_change(self, value):
        """Handle noise amount slider change"""
        self.noise_amount = int(float(value))
        self.noise_value_label.config(text=f"{self.noise_amount} objects")
        if self.apply_noise.get():
            self.generate_noise_circles()
            self.update_map_display()

    def on_apply_noise_toggle(self):
        """Handle apply noise checkbox toggle"""
        if self.apply_noise.get():
            self.generate_noise_circles()
        else:
            self.map_with_noise = None
        self.update_map_display()

    def generate_noise_circles(self):
        """Generate random circles for noise based on noise_amount and create cached map"""
        import random

        self.noise_circles = []

        for _ in range(self.noise_amount):
            x = random.randint(0, MAP_WIDTH)
            y = random.randint(0, MAP_HEIGHT)
            radius = random.randint(3, 15)
            self.noise_circles.append((x, y, radius))

        self.update_map_with_noise()

    def update_map_with_noise(self):
        """Create a cached map with noise applied for raytracing"""
        if self.apply_noise.get() and len(self.noise_circles) > 0:
            self.map_with_noise = self.current_map_image.copy()
            self.draw_noise_circles_on_image(self.map_with_noise)
        else:
            self.map_with_noise = None

    def get_map_for_raytracing(self):
        """Get the appropriate map for raytracing (with or without noise)"""
        if self.map_with_noise is not None:
            return self.map_with_noise
        return self.current_map_image

    def compute_confidence_heatmap(self):
        """
        Pre-compute confidence values at a dense grid of positions for each direction.
        Creates separate heatmaps for each of the SAMPLE_ROTATIONS angles.
        """
        if not self.localization.is_trained:
            self.status_label['text'] = "⚠️ Network not trained. Please use 'Sample & Train' first before computing confidence heatmap."
            self.show_confidence_heatmap.set(False)
            return

        grid_positions_by_angle, _ = self.sampling.generate_heatmap_grid_positions()

        def localization_callback(x, y, angle):
            map_image = self.get_map_for_raytracing()
            camera_view = self.camera.capture_view(x, y, angle, map_image)
            return self.localization.localize(camera_view, top_k=1)

        result = self.confidence.compute_confidence_heatmap(
            grid_positions_by_angle, localization_callback)

        if result:
            self.status_label['text'] = f"✓ Heatmap computed: {
                result['total_evaluations']}  positions across {
                len(result['angles'])}  angles."

    def open_map_editor(self):
        MapEditor(self.root, self.current_map_image, self.on_map_saved)

    def on_map_saved(self, new_map_image):
        self.current_map_image = new_map_image
        if self.apply_noise.get():
            self.update_map_with_noise()
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
                if self.apply_noise.get():
                    self.update_map_with_noise()
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

        if self.localization.get_num_samples() > 0:
            self.train_network(show_message=False)

            self.status_label['text'] = f"✓ Generated {
                self.localization.get_num_samples()}  samples and trained network with {
                self.localization.get_num_samples()}  patterns"

    def auto_sample(self, show_message=True):
        """
        Automatically generate samples from the current map.
        Samples at discrete positions with SAMPLE_STRIDE spacing and
        SAMPLE_ROTATIONS rotations at each position.

        Args:
            show_message: Whether to show success message (default True)
        """
        self.localization.clear_samples()
        self.clear_sample_dots()

        sample_positions = self.sampling.generate_sample_positions()
        total_samples = len(sample_positions)

        original_state = self.robot.copy_state()

        self.status_label['text'] = f"Sampling: 0/{total_samples}"

        map_image = self.get_map_for_raytracing()

        for x, y, angle in sample_positions:
            camera_view = self.camera.capture_view(x, y, angle, map_image)
            self.localization.add_sample(x, y, angle, camera_view)

        self.robot.restore_state(original_state)

        self.update_map_display()
        self.draw_sample_dots()

        self.status_label['text'] = ""

        if show_message:
            self.status_label['text'] = f"✓ Generated {
                self.localization.get_num_samples()}  samples"

    def train_network(self, show_message=True):
        """
        Train the Modern Hopfield Network using the collected samples.

        Args:
            show_message: Whether to show success message (default True)
        """
        if self.localization.get_num_samples() == 0:
            self.status_label['text'] = "❌ No samples available. Please run 'Sample & Train' first."
            return

        self.status_label['text'] = f"Training: 0/{
            self.localization.get_num_samples()} "

        success = self.localization.train()

        if success:
            self.confidence.reset()
            self.compute_average_confidence()

            if show_message:
                if self.confidence.average_confidence is not None:
                    self.status_label['text'] = f"✓ Network trained with {
                        self.localization.get_num_samples()}  patterns. Avg confidence: {
                        self.confidence.average_confidence * 100: .1f} %"
                else:
                    self.status_label['text'] = f"✓ Network trained with {
                        self.localization.get_num_samples()}  patterns."
            else:
                self.status_label['text'] = ""
        else:
            self.status_label['text'] = "❌ Training error"

    def compute_average_confidence(self, num_tests=None):
        """
        Deterministically compute average retrieval confidence over a set of
        positions. Positions are taken from a grid (SAMPLE_STRIDE) and a set
        of discrete angles so the result is repeatable.

        Args:
            num_tests: maximum number of evaluation positions to use (default: 100)
        """
        if not self.localization.is_trained:
            return

        test_positions = self.sampling.generate_test_positions(num_tests)
        map_image = self.get_map_for_raytracing()

        def localization_callback(x, y, angle):
            camera_view = self.camera.capture_view(x, y, angle, map_image)
            return self.localization.localize(camera_view, top_k=1)

        result = self.confidence.compute_average_confidence(
            test_positions, localization_callback)

        if result:
            self.update_statistics_display()

        self.status_label['text'] = ""

    def update_statistics_display(self):
        """Update the statistics display with confidence information"""
        if self.confidence.average_confidence is not None:
            stats_text = f"Test Positions: {
                self.confidence.confidence_num_tests} \n"
            stats_text += f"Avg Confidence: {
                self.confidence.average_confidence * 100: .1f} %"
            self.stats_label.config(text=stats_text)
        else:
            self.stats_label.config(
                text="Train the network to see confidence statistics")

    def clear_sample_dots(self):
        """Remove all sample dots from the canvas"""
        for dot_id in self.sample_dots:
            self.map_canvas.delete(dot_id)
        self.sample_dots = []

    def draw_sample_dots(self):
        """Draw red dots at sample positions on the canvas, sized by similarity"""
        self.clear_sample_dots()

        sample_similarities = self.localization.sample_similarities
        max_similarity = None
        if sample_similarities is not None and len(sample_similarities) > 0:
            max_similarity = max(sample_similarities.max(), 1e-6)

        for i, (x, y, _angle) in enumerate(self.localization.sample_positions):
            if sample_similarities is not None and i < len(sample_similarities):
                similarity = sample_similarities[i]
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

        for x, y in self.confidence.test_positions:
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

    def draw_top_k_interpolation_lines(self):
        """Draw black lines from top k matches to the final predicted position"""
        if not self.top_k_matches or self.estimated_x is None or self.estimated_y is None:
            return

        canvas_est_x, canvas_est_y = self.image_to_canvas_coords(
            self.estimated_x, self.estimated_y)

        for match in self.top_k_matches:
            match_x = match['x']
            match_y = match['y']
            canvas_match_x, canvas_match_y = self.image_to_canvas_coords(
                match_x, match_y)

            self.map_canvas.create_line(
                canvas_match_x, canvas_match_y,
                canvas_est_x, canvas_est_y,
                fill="#000000",
                width=max(1, int(1 * self.map_scale_factor)),
                tags="top_k_line"
            )

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

        sample_similarities = self.localization.sample_similarities
        max_similarity = None
        if sample_similarities is not None and len(sample_similarities) > 0:
            max_similarity = max(sample_similarities.max(), 1e-6)

        for i, (x, y, _angle) in enumerate(self.localization.sample_positions):
            distance = math.sqrt((x - img_x)**2 + (y - img_y)**2)
            if sample_similarities is not None and i < len(sample_similarities):
                similarity = sample_similarities[i]
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

        if self.apply_noise.get():
            self.draw_noise_circles_on_image(display_image)

        self.draw_viewing_cone_on_image(display_image)

        if self.show_confidence_heatmap.get():
            self.draw_confidence_heatmap_on_image(display_image)

        self._display_scaled_map_image(display_image)

        self.capture_camera_view()
        self.display_camera_view()

        if self.localization.is_trained:
            self.localize()

        self.draw_sample_dots()

        self.draw_test_position_dots()

        self.draw_top_k_interpolation_lines()

        self.draw_robot()

    def draw_robot(self):
        canvas_robot_x, canvas_robot_y = self.image_to_canvas_coords(
            self.robot.x, self.robot.y)
        scaled_radius = self.robot.radius * self.map_scale_factor

        if self.robot_image is not None:
            rotated_image = self.robot_image.rotate(
                -self.robot.angle, expand=True)

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
        """Draw a smooth 2D color gradient heatmap using ConfidenceAnalyzer"""
        heatmap_image = self.confidence.build_heatmap_image(
            current_angle=self.robot.angle,
            average_all_angles=self.average_heatmap.get(),
            colormap='jet',
            sigma=20.0
        )

        if heatmap_image is not None:
            image.paste(heatmap_image, (0, 0), heatmap_image)

    def draw_noise_circles_on_image(self, image):
        """Draw random noise circles on the image"""
        draw = ImageDraw.Draw(image)

        for x, y, radius in self.noise_circles:
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill='black',
                outline='black'
            )

    def draw_viewing_cone_on_image(self, image):
        """Draw the viewing cone with 50% transparency directly on a PIL image"""
        map_image = self.get_map_for_raytracing()
        points = self.camera.get_viewing_cone_points(
            self.robot.x, self.robot.y, self.robot.angle, map_image)

        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        cone_color = (0, 0, 255, 127)
        draw.polygon(points, fill=cone_color, outline=None)

        image.paste(overlay, (0, 0), overlay)

    def capture_camera_view(self):
        """Capture camera view using the camera simulator"""
        map_image = self.get_map_for_raytracing()
        self.current_camera_view = self.camera.capture_view(
            self.robot.x, self.robot.y, self.robot.angle, map_image)

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
        if not self.localization.is_trained:
            return

        if self.current_camera_view is None:
            return

        result = self.localization.localize(self.current_camera_view, top_k=self.top_k)

        if result:
            self.estimated_x = result['x']
            self.estimated_y = result['y']
            self.estimated_angle = result['angle']
            self.retrieved_sample_idx = result['sample_idx']
            self.top_k_matches = result.get('top_k_matches', [])

            self.display_retrieved_memory(
                result['sample_idx'],
                result['confidence'])
            self.display_similarity_metric(result['confidence'])

    def display_retrieved_memory(self, sample_idx, weight):
        """
        Display the retrieved memory view in the retrieved memory canvas.

        Args:
            sample_idx: Index of the retrieved sample
            weight: Attention weight/confidence score
        """
        sample_info = self.localization.get_sample_info(sample_idx)
        if not sample_info:
            return

        saved_view = sample_info['view']

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

        x, y, angle = sample_info['x'], sample_info['y'], sample_info['angle']
        info_text = f"({x}, {y}), {angle:.1f}° | Weight: {weight:.4f}"
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
        sample_info = self.localization.get_sample_info(sample_idx)
        if not sample_info:
            return

        saved_view = sample_info['view']

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

        x, y, angle = sample_info['x'], sample_info['y'], sample_info['angle']
        info_text = f"Pos: ({x}, {y}), Angle: {angle:.1f}°"
        self.memory_frame.config(text=f"Retrieved Memory: {info_text}")

        self.memory_canvas.image = tk_image

    def move_robot(self, dx, dy):
        """Update robot position"""
        self.robot.move(dx, dy)
        self.update_map_display()

    def rotate_robot(self, angle_delta):
        """Update robot angle by the specified delta (already a discrete step)"""
        self.robot.rotate(angle_delta)
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
            dy -= self.robot.speed
        if 's' in self.keys_pressed:
            dy += self.robot.speed
        if 'a' in self.keys_pressed:
            dx -= self.robot.speed
        if 'd' in self.keys_pressed:
            dx += self.robot.speed

        if 'j' in self.rotation_keys_pressed:
            d_angle -= self.robot.rotation_speed
        if 'l' in self.rotation_keys_pressed:
            d_angle += self.robot.rotation_speed

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
