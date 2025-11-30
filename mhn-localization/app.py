import tkinter as tk
from tkinter import filedialog, ttk
from .constants import *
from .constants import DEFAULT_NUM_ANGLES, CAMERA_NUM_RAYS
from PIL import Image, ImageTk
from .editor import MapEditor
from .config import ConfigManager
from .robot_state import RobotState
from .camera import CameraSimulator
from .localization_engine import LocalizationEngine
from .sampling_engine import SamplingEngine
from .confidence_analyzer import ConfidenceAnalyzer
from .canvas_state import CanvasState
from .canvas_renderer import CanvasRenderer
import math
import os
from .convergence_controller import ConvergenceController
import random
from tkinter import simpledialog

import matplotlib.pyplot as plt


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Robot Localization via Modern Hopfield Networks")
        self.root.geometry(f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}")

        self.config = ConfigManager()

        self._suppress_updates = True

        # Load saved parameters or use defaults
        self._blur_radius = self.config.get("blur_radius", CAMERA_BLUR_RADIUS)
        self._fov = self.config.get("fov", CAMERA_FOV)
        self._visibility_index = self.config.get("visibility_index", 0.1)
        self._beta = self.config.get("beta", DEFAULT_BETA)
        self._top_k = self.config.get("top_k", DEFAULT_TOP_K)
        self._noise_amount = self.config.get("noise_amount", DEFAULT_NOISE_AMOUNT)
        self._apply_noise = self.config.get("apply_noise", False)
        self._interleaved_rgb = self.config.get("interleaved_rgb", INTERLEAVED_RGB)
        self._num_angles = self.config.get("num_angles", DEFAULT_NUM_ANGLES)
        self._num_rays = self.config.get("num_rays", CAMERA_NUM_RAYS)

        self.robot = RobotState()
        self.camera = CameraSimulator(
            cone_angle=self._fov,
            cone_length=40,
            camera_samples=self._num_rays,
            blur_radius=self._blur_radius,
            visibility_index=self._visibility_index
        )
        self.localization = LocalizationEngine(
            beta=self._beta,
            interleaved_rgb=self._interleaved_rgb
        )
        self.sampling = SamplingEngine(num_rotations=self._num_angles)
        self.confidence = ConfidenceAnalyzer(MAP_WIDTH, MAP_HEIGHT)

        self.renderer = CanvasRenderer()

        self.canvas_state = CanvasState(
            current_map_image=Image.new("RGB", (MAP_WIDTH, MAP_HEIGHT), "white"),
            robot_x=self.robot.x,
            robot_y=self.robot.y,
            robot_angle=self.robot.angle,
            robot_radius=self.robot.radius,
            apply_noise=self._apply_noise,
            camera_fov=self._fov,
            camera_cone_length=40,
        )

        # UI-related state
        self.robot_image = None
        self.load_robot_image()

        self.interleaved_rgb = tk.BooleanVar(value=self._interleaved_rgb)

        self.noise_amount = self._noise_amount
        self.apply_noise = tk.BooleanVar(value=self._apply_noise)

        self.current_camera_view = None

        self.show_test_positions = tk.BooleanVar(value=False)
        self.show_confidence_heatmap = tk.BooleanVar(value=False)
        self.show_energy_heatmap = tk.BooleanVar(value=False)
        self.average_heatmap = tk.BooleanVar(value=False)

        self.top_k = self._top_k

        self._hovered_strip_images = []

        self.convergence_controller = ConvergenceController(
            self.localization,
            self.canvas_state,
            self.set_status,
            self.update_map_display
        )
        self.convergence_controller.on_convergence_finished = lambda: self.converge_btn.config(
            text="Converge to Pattern")

        self.keys_pressed = set()
        self.update_interval = 16

        self.rotation_keys_pressed = set()

        self.create_layout()
        self.create_toolbar()
        self.create_left_panel()
        self.create_right_panel()

        self.load_last_map()

        if self._apply_noise:
            self.generate_noise_circles()

        self.initialize_ui_values()
        self._suppress_updates = False
        self.update_map_display()

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

    def set_status(self, message):
        """Set the status label text"""
        self.status_label.config(text=message)

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
            self.status_frame, text="Train using sampling or SGD!", anchor="w")
        self.status_label.pack(side=tk.LEFT, fill=tk.X,
                               expand=True, padx=5, pady=2)

    def create_toolbar(self):
        buttons = {
            "Edit Map": self.open_map_editor,
            "Import Map": self.import_map,
            "Export Map": self.export_map,
            "Train using sampling": self.sample_and_train
        }

        for btn_text, command in buttons.items():
            btn = ttk.Button(self.toolbar_frame,
                             text=btn_text, command=command)
            btn.pack(side=tk.LEFT, padx=5)

        self.auto_explore_btn = ttk.Button(
            self.toolbar_frame,
            text="Train using SGD",
            command=self.start_auto_exploration
        )
        self.auto_explore_btn.pack(side=tk.LEFT, padx=5)

        self.converge_btn = ttk.Button(
            self.toolbar_frame,
            text="Converge to Pattern",
            command=self.start_convergence
        )
        self.converge_btn.pack(side=tk.LEFT, padx=5)

        self.clear_convergence_btn = ttk.Button(
            self.toolbar_frame,
            text="Clear Convergence",
            command=self.clear_convergence
        )
        self.clear_convergence_btn.pack(side=tk.LEFT, padx=5)

        help_btn = ttk.Button(self.toolbar_frame,
                              text="Help", command=self.show_help)
        help_btn.pack(side=tk.RIGHT, padx=5)

    def create_left_panel(self):
        self.left_panel = ttk.Frame(self.content_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH,
                             expand=True, padx=5)

        self.map_canvas = tk.Canvas(
            self.left_panel, highlightthickness=1, bg="#000000")
        self.map_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.map_canvas.bind('<Motion>', self.on_canvas_hover)
        self.map_canvas.bind('<Configure>', self.on_map_canvas_resize)

    def create_right_panel(self):
        right_panel_container = ttk.Frame(self.content_frame, width=400)
        right_panel_container.pack(
            side=tk.LEFT, fill=tk.Y, expand=False, padx=(5, 10))
        right_panel_container.pack_propagate(False)

        canvas = tk.Canvas(right_panel_container)
        scrollbar = ttk.Scrollbar(
            right_panel_container, orient=tk.VERTICAL, command=canvas.yview)
        self.right_panel = ttk.Frame(canvas)

        self.right_panel.bind(
            "<Configure>", lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=self.right_panel, anchor="nw", width=380)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

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

        sim_frame = tk.LabelFrame(self.right_panel, text="Confidence")
        sim_frame.pack(fill=tk.X, padx=5, pady=5)

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
            blur_container, anchor="w"
        )
        self.blur_value_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.blur_slider = ttk.Scale(
            blur_container, from_=0.0, to=5.0, orient=tk.HORIZONTAL,
            command=self.on_blur_change
        )
        self.blur_slider.set(self._blur_radius)
        self.blur_slider.pack(fill=tk.X, padx=5, pady=5)

        fov_container = tk.LabelFrame(settings_frame, text="Field of View")
        fov_container.pack(fill=tk.X, padx=5, pady=5)

        self.fov_value_label = ttk.Label(
            fov_container, anchor="w"
        )
        self.fov_value_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.fov_slider = ttk.Scale(
            fov_container, from_=30, to=360, orient=tk.HORIZONTAL,
            command=self.on_fov_change
        )
        self.fov_slider.set(self._fov)
        self.fov_slider.pack(fill=tk.X, padx=5, pady=5)

        num_rays_container = tk.LabelFrame(settings_frame, text="Number of Camera Rays")
        num_rays_container.pack(fill=tk.X, padx=5, pady=5)

        self.num_rays_value_label = ttk.Label(
            num_rays_container, anchor="w"
        )
        self.num_rays_value_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.num_rays_slider = ttk.Scale(
            num_rays_container, from_=10, to=300, orient=tk.HORIZONTAL,
            command=self.on_num_rays_change
        )
        self.num_rays_slider.set(self._num_rays)
        self.num_rays_slider.pack(fill=tk.X, padx=5, pady=5)

        visibility_container = tk.LabelFrame(
            settings_frame, text="Visibility Index"
        )
        visibility_container.pack(fill=tk.X, padx=5, pady=5)

        self.visibility_value_label = ttk.Label(
            visibility_container, anchor="w"
        )
        self.visibility_value_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.visibility_slider = ttk.Scale(
            visibility_container, from_=0.01, to=1.0,
            orient=tk.HORIZONTAL, command=self.on_visibility_change
        )
        self.visibility_slider.set(self._visibility_index)
        self.visibility_slider.pack(fill=tk.X, padx=5, pady=5)

        beta_container = tk.LabelFrame(
            settings_frame, text="Beta (Inverse Temp)"
        )
        beta_container.pack(fill=tk.X, padx=5, pady=5)

        self.beta_value_label = ttk.Label(
            beta_container, anchor="w"
        )
        self.beta_value_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.beta_slider = ttk.Scale(
            beta_container, from_=1.0, to=200.0, orient=tk.HORIZONTAL,
            command=self.on_beta_change
        )
        self.beta_slider.set(self._beta)
        self.beta_slider.pack(fill=tk.X, padx=5, pady=5)

        top_k_container = tk.LabelFrame(
            settings_frame, text="Combine top k matches"
        )
        top_k_container.pack(fill=tk.X, padx=5, pady=5)

        self.top_k_value_label = ttk.Label(
            top_k_container, anchor="w"
        )
        self.top_k_value_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.top_k_slider = ttk.Scale(
            top_k_container, from_=1, to=20, orient=tk.HORIZONTAL,
            command=self.on_top_k_change
        )
        self.top_k_slider.set(self._top_k)
        self.top_k_slider.pack(fill=tk.X, padx=5, pady=5)

        num_angles_container = tk.LabelFrame(
            settings_frame, text="Number of Angles per Location"
        )
        num_angles_container.pack(fill=tk.X, padx=5, pady=5)

        self.num_angles_value_label = ttk.Label(
            num_angles_container, anchor="w"
        )
        self.num_angles_value_label.pack(side=tk.LEFT, padx=5, pady=5)

        slider_position = VALID_NUM_ANGLES.index(self._num_angles) if self._num_angles in VALID_NUM_ANGLES else 2
        self.num_angles_slider = ttk.Scale(
            num_angles_container, from_=0, to=3, orient=tk.HORIZONTAL,
            command=self.on_num_angles_change
        )
        self.num_angles_slider.set(slider_position)
        self.num_angles_slider.pack(fill=tk.X, padx=5, pady=5)

        noise_container = tk.LabelFrame(
            settings_frame, text="Noise Settings"
        )
        noise_container.pack(fill=tk.X, padx=5, pady=5)

        self.noise_value_label = ttk.Label(
            noise_container, anchor="w"
        )
        self.noise_value_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.noise_slider = ttk.Scale(
            noise_container, from_=0, to=100, orient=tk.HORIZONTAL,
            command=self.on_noise_change
        )
        self.noise_slider.set(self._noise_amount)
        self.noise_slider.pack(fill=tk.X, padx=5, pady=5)

        self.apply_noise_checkbox = ttk.Checkbutton(
            noise_container, text="Apply noise to map",
            variable=self.apply_noise,
            command=self.on_apply_noise_toggle
        )
        self.apply_noise_checkbox.pack(fill=tk.X, padx=5, pady=5)

        self.interleaved_rgb_checkbox = ttk.Checkbutton(
            settings_frame, text="Interleaved RGB encoding",
            variable=self.interleaved_rgb,
            command=self.on_interleaved_rgb_toggle
        )
        self.interleaved_rgb_checkbox.pack(fill=tk.X, padx=5, pady=5)

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

        self.show_test_positions_checkbox = ttk.Checkbutton(
            stats_frame, text="Show confidence computation positions",
            variable=self.show_test_positions,
            command=self.on_show_test_positions_toggle
        )
        self.show_test_positions_checkbox.pack(fill=tk.X, padx=5, pady=5)

        self.show_confidence_heatmap_checkbox = ttk.Checkbutton(
            stats_frame, text="Show confidence heatmap (warmer = better)",
            variable=self.show_confidence_heatmap,
            command=self.on_show_confidence_heatmap_toggle
        )
        self.show_confidence_heatmap_checkbox.pack(fill=tk.X, padx=5, pady=5)

        self.show_energy_heatmap_checkbox = ttk.Checkbutton(
            stats_frame, text="Show energy heatmap (warmer = better)",
            variable=self.show_energy_heatmap,
            command=self.on_show_energy_heatmap_toggle
        )
        self.show_energy_heatmap_checkbox.pack(fill=tk.X, padx=5, pady=5)

        self.average_heatmap_checkbox = ttk.Checkbutton(
            stats_frame, text="Average heatmap across all angles",
            variable=self.average_heatmap,
            command=self.on_average_heatmap_toggle
        )
        self.average_heatmap_checkbox.pack(fill=tk.X, padx=5, pady=5)

    def initialize_ui_values(self):
        """Initialize UI values by calling update functions"""
        self.on_blur_change(self._blur_radius)
        self.on_fov_change(self._fov)
        self.on_num_rays_change(self._num_rays)
        self.on_visibility_change(self._visibility_index)
        self.on_beta_change(self._beta)
        self.on_top_k_change(self._top_k)

        slider_position = VALID_NUM_ANGLES.index(self._num_angles) if self._num_angles in VALID_NUM_ANGLES else 2
        self.on_num_angles_change(slider_position)

        self.on_noise_change(self._noise_amount)

    def show_help(self):
        """Display a help dialog with information about controls and features"""
        help_window = tk.Toplevel(self.root)
        help_window.title("Help - Robot Localization")
        help_window.geometry("700x600")

        frame = ttk.Frame(help_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        text_widget = tk.Text(frame, wrap=tk.WORD, font=("Arial", 11),
                              padx=10, pady=10)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL,
                                  command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        text_widget.insert("1.0", HELP_TEXT)
        text_widget.configure(state="disabled")

        close_btn = ttk.Button(help_window, text="Close",
                               command=help_window.destroy)
        close_btn.pack(pady=10)

    def on_show_test_positions_toggle(self):
        """Handle checkbox toggle for showing test positions"""
        self.update_map_display()

    def on_show_confidence_heatmap_toggle(self):
        """Handle checkbox toggle for showing confidence heatmap"""
        if self.show_confidence_heatmap.get():
            if not self.confidence.heatmap_computed:
                self.compute_confidence_heatmap()

        self.update_map_display()

    def on_show_energy_heatmap_toggle(self):
        """Handle checkbox toggle for showing energy heatmap"""
        if self.show_energy_heatmap.get():
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
        self.config.set("blur_radius", self.camera.blur_radius)
        self.update_map_display()

    def on_fov_change(self, value):
        """Handle FOV slider change"""
        self.camera.cone_angle = float(value)
        self.fov_value_label.config(text=f"{self.camera.cone_angle:.0f}°")
        self.config.set("fov", self.camera.cone_angle)
        self.update_map_display()

    def on_num_rays_change(self, value):
        """Handle number of rays slider change"""
        num_rays = int(float(value))
        embedding_dim = num_rays * 3

        self._num_rays = num_rays
        self.num_rays_value_label.config(text=f"{num_rays} rays (embedding dim: {embedding_dim})")
        self.config.set("num_rays", num_rays)

        self.camera.camera_samples = num_rays

        self._ensure_retrain()
        self.update_map_display()

    def on_visibility_change(self, value):
        """Handle visibility index slider change"""
        self.camera.visibility_index = float(value)
        self.visibility_value_label.config(
            text=f"{self.camera.visibility_index:.2f}")
        self.config.set("visibility_index", self.camera.visibility_index)
        self.update_map_display()

    def on_beta_change(self, value):
        """Handle beta slider change"""
        beta = float(value)
        self.beta_value_label.config(text=f"{beta:.1f}")
        self.localization.update_beta(beta)
        self.config.set("beta", beta)

        if self.localization.is_trained:
            self.update_map_display()

    def on_top_k_change(self, value):
        """Handle top-k slider change"""
        self.top_k = int(float(value))
        self.top_k_value_label.config(text=f"{self.top_k}")
        self.config.set("top_k", self.top_k)

        if self.localization.is_trained:
            self.localize()
            self.update_map_display()

    def on_num_angles_change(self, value):
        """Handle number of angles slider change"""
        slider_position = int(round(float(value)))
        slider_position = max(0, min(3, slider_position))

        num_angles = VALID_NUM_ANGLES[slider_position]

        self._num_angles = num_angles
        self.num_angles_value_label.config(text=f"{self._num_angles}")
        self.config.set("num_angles", self._num_angles)

        self.sampling.num_rotations = self._num_angles

        self._ensure_retrain()

    def on_interleaved_rgb_toggle(self):
        """Handle interleaved RGB checkbox toggle"""
        self.config.set("interleaved_rgb", self.interleaved_rgb.get())
        self._ensure_retrain()

    def _ensure_retrain(self):
        """Update status if retraining is required"""
        if self.localization.is_trained:
            self.set_status(RETRAINING_REQUIRED_MSG)

    def on_noise_change(self, value):
        """Handle noise amount slider change"""
        self.noise_amount = int(float(value))
        self.noise_value_label.config(text=f"{self.noise_amount} objects")
        self.config.set("noise_amount", self.noise_amount)
        if self.apply_noise.get():
            self.generate_noise_circles()
            self.update_map_display()

    def on_apply_noise_toggle(self):
        """Handle apply noise checkbox toggle"""
        self.config.set("apply_noise", self.apply_noise.get())
        self.canvas_state.apply_noise = self.apply_noise.get()
        if self.apply_noise.get():
            self.generate_noise_circles()
        else:
            self.canvas_state.map_with_noise = None
        self.update_map_display()

    def generate_noise_circles(self):
        """Generate random circles for noise based on noise_amount"""

        self.canvas_state.noise_circles = []

        for _ in range(self.noise_amount):
            x = random.randint(0, MAP_WIDTH)
            y = random.randint(0, MAP_HEIGHT)
            radius = random.randint(3, 15)
            self.canvas_state.noise_circles.append((x, y, radius))

        self.renderer.update_map_with_noise(self.canvas_state)

    def get_map_for_raytracing(self):
        """Get the appropriate map for raytracing (with or without noise)"""
        return self.renderer.get_map_for_raytracing(self.canvas_state)

    def compute_confidence_heatmap(self):
        """
        Pre-compute confidence values at a dense grid of positions for each direction.
        Creates separate heatmaps for each of the SAMPLE_ROTATIONS angles.
        """
        if not self.localization.is_trained:
            self.set_status(NETWORK_NOT_TRAINED_MSG)
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
            self.set_status(f"✓ Heatmap computed: {
                result['total_evaluations']}  positions across {
                len(result['angles'])}  angles.")

    def open_map_editor(self):
        MapEditor(self.root, self.canvas_state.current_map_image, self.on_map_saved)

    def on_map_saved(self, new_map_image):
        self.canvas_state.current_map_image = new_map_image
        if self.apply_noise.get():
            self.renderer.update_map_with_noise(self.canvas_state)
        self.update_map_display()

    def load_robot_image(self):
        """Load the robot image from resources"""
        try:
            robot_image_path = os.path.join(
                os.path.dirname(__file__), "resources", "uii.png")
            self.robot_image = Image.open(robot_image_path)
            if self.robot_image.mode != "RGBA":
                self.robot_image = self.robot_image.convert("RGBA")
            self.renderer.set_robot_image(self.robot_image)
            print(f"Loaded robot image: {robot_image_path}")
        except Exception as e:
            print(f"Failed to load robot image: {e}")
            self.robot_image = None

    def _load_and_process_map(self, file_path):
        """Load and process a map image from a file path."""
        imported_image = Image.open(file_path)

        if imported_image.size != (MAP_WIDTH, MAP_HEIGHT):
            imported_image = imported_image.resize(
                (MAP_WIDTH, MAP_HEIGHT),
                Image.Resampling.LANCZOS
            )

        if imported_image.mode != "RGB":
            imported_image = imported_image.convert("RGB")

        return imported_image

    def load_last_map(self):
        last_path = self.config.get_last_map_path()

        if last_path and os.path.exists(last_path):
            try:
                self.canvas_state.current_map_image = self._load_and_process_map(last_path)
                print(f"Loaded last map: {last_path}")

            except (OSError, ValueError) as e:
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
                self.canvas_state.current_map_image = self._load_and_process_map(file_path)
                if self.apply_noise.get():
                    self.renderer.update_map_with_noise(self.canvas_state)
                self.update_map_display()

                self.config.set_last_map_path(file_path)

                self.set_status(f"✓ Map imported successfully from {os.path.basename(file_path)}")

            except (OSError, ValueError) as e:
                self.set_status(f"❌ Failed to import map: {str(e)}")

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
                self.canvas_state.current_map_image.save(file_path)

                self.config.set_last_map_path(file_path)

                self.set_status(f"✓ Map exported to {
                    os.path.basename(file_path)} ")
            except (OSError, ValueError) as e:
                self.set_status(f"❌ Failed to export map: {str(e)}")

    def sample_and_train(self):
        """
        Automatically generate samples and train the network.
        """
        self.auto_sample()

        if self.localization.get_num_samples() == 0:
            return

        self.train_network()

        self.set_status(f"✓ Generated {
            self.localization.get_num_samples()}  samples and trained network with {
            self.localization.get_num_samples()}  patterns")

    def auto_sample(self):
        """
        Automatically generate samples from the current map.
        Samples at discrete positions with SAMPLE_STRIDE spacing and
        SAMPLE_ROTATIONS rotations at each position.
        """
        self.localization.clear_samples()

        sample_positions = self.sampling.generate_sample_positions()
        total_samples = len(sample_positions)

        original_state = self.robot.copy_state()

        map_image = self.get_map_for_raytracing()

        for x, y, angle in sample_positions:
            camera_view = self.camera.capture_view(x, y, angle, map_image)
            self.localization.add_sample(x, y, angle, camera_view)

        self.robot.restore_state(original_state)
        self.update_map_display()

        self.set_status(f"✓ Generated {total_samples} samples.")

    def start_auto_exploration(self):
        """
        Start the auto-exploration and training process.
        Collects samples from a grid and trains the network to find optimal prototypes.
        """

        # Get grid positions to calculate total samples
        grid_positions = self.sampling.generate_sample_positions()
        total_samples = len(grid_positions)

        # Ask for number of patterns to learn
        num_patterns = simpledialog.askinteger(
            "Auto-Explore & Train",
            f"Grid sampling will collect {total_samples} observations.\n\n"
            "How many patterns (prototypes) should the network learn?",
            parent=self.root, minvalue=1, maxvalue=total_samples, initialvalue=min(100, total_samples // 10)
        )

        if num_patterns is None:
            return

        self.set_status(EXPLORING_MSG)
        self.root.update()

        original_state = self.robot.copy_state()

        # Clear existing samples
        self.localization.clear_samples()

        map_image = self.get_map_for_raytracing()

        for i, (x, y, angle) in enumerate(grid_positions):
            self.robot.x = x
            self.robot.y = y
            self.robot.angle = angle
            self.canvas_state.robot_x = x
            self.canvas_state.robot_y = y
            self.canvas_state.robot_angle = angle

            # Capture view
            camera_view = self.camera.capture_view(x, y, angle, map_image)
            self.localization.add_sample(x, y, angle, camera_view)

            if i % 50 == 0:
                self.set_status(f"⟳ Exploring... Sample {i+1}/{total_samples}")
                self.root.update()

        self.update_map_display()

        # Train
        self.set_status(f"⟳ Training network with {num_patterns} patterns...")
        self.root.update()

        def train_callback(epoch, total, loss):
            if epoch % 5 == 0 or epoch == total:
                self.set_status(f"⟳ Training... Epoch {epoch}/{total}, Loss: {loss:.4f}")
                self.root.update()

        success, loss_history = self.localization.train_sgd(
            num_patterns=num_patterns,
            learning_rate=0.1,
            epochs=100,
            progress_callback=train_callback
        )

        if success:
            self.set_status(f"✓ Exploration complete! Learned {num_patterns} patterns from {total_samples} samples.")
            self.show_training_stats(loss_history)
        else:
            self.set_status(TRAINING_FAILED_MSG)

        self.robot.restore_state(original_state)
        self.update_map_display()

    def show_training_stats(self, loss_history):
        """
        Show a popup window with training statistics (loss curve) using Matplotlib.
        """
        if not loss_history:
            return

        plt.figure(figsize=(8, 5))
        plt.plot(loss_history, 'b-', linewidth=2)
        plt.title("Training Loss (Energy)")
        plt.xlabel("Epochs")
        plt.ylabel("Energy")
        plt.grid(True)
        plt.show(block=False)

    def train_network(self):
        """
        Train the Modern Hopfield Network using the collected samples.
        """
        if self.localization.get_num_samples() == 0:
            self.set_status(NO_SAMPLES_MSG)
            return

        self.set_status(f"Training: 0/{
            self.localization.get_num_samples()} ")

        success = self.localization.train()

        if success:
            self.confidence.reset()
            self.compute_average_confidence()

            if self.confidence.average_confidence is not None:
                self.set_status(f"✓ Network trained with {
                    self.localization.get_num_samples()}  patterns. Avg confidence: {
                    self.confidence.average_confidence * 100: .1f} %")
            else:
                self.set_status(f"✓ Network trained with {
                    self.localization.get_num_samples()}  patterns.")
        else:
            self.set_status(TRAINING_ERROR_MSG)

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

        self.set_status("")

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

    def on_map_canvas_resize(self, event):
        """Handle map canvas resize and update display"""
        self.update_map_display()

    def on_canvas_hover(self, event):
        """
        Handle mouse hover over the canvas.
        Display the saved sample when hovering over a red dot.
        """
        img_x, img_y = self.canvas_state.canvas_to_image_coords(event.x, event.y)

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

        if closest_idx != self.canvas_state.hovered_sample_idx:
            self.canvas_state.hovered_sample_idx = closest_idx

            # Clear stored hover images
            self._hovered_strip_images = []

            if closest_idx is not None:
                self.display_saved_sample(closest_idx)
                self.update_map_display()
            else:
                self.memory_canvas.delete("all")
                self.memory_frame.config(text="Retrieved Memory")
                self.update_map_display()

    def update_map_display(self):
        """Update the map canvas display using the renderer."""
        if self._suppress_updates:
            return

        self.capture_camera_view()
        self.display_camera_view()

        if self.localization.is_trained:
            self.localize()

        # Sync robot state to canvas_state
        self.canvas_state.robot_x = self.robot.x
        self.canvas_state.robot_y = self.robot.y
        self.canvas_state.robot_angle = self.robot.angle

        # Sync other dynamic state
        self.canvas_state.apply_noise = self.apply_noise.get()
        self.canvas_state.show_test_positions = self.show_test_positions.get()
        self.canvas_state.show_confidence_heatmap = self.show_confidence_heatmap.get()
        self.canvas_state.show_energy_heatmap = self.show_energy_heatmap.get()
        self.canvas_state.average_heatmap = self.average_heatmap.get()
        self.canvas_state.sample_positions = self.localization.sample_positions.copy() if self.localization.sample_positions else []
        self.canvas_state.sample_similarities = self.localization.sample_similarities
        self.canvas_state.test_positions = self.confidence.test_positions.copy() if self.confidence.test_positions else []

        # Generate heatmap images if needed
        if self.canvas_state.show_confidence_heatmap and self.confidence.heatmap_computed:
            self.canvas_state.confidence_heatmap_image = self.confidence.build_heatmap_image(
                current_angle=self.robot.angle,
                average_all_angles=self.canvas_state.average_heatmap,
                colormap='jet',
                sigma=20.0
            )

        if self.canvas_state.show_energy_heatmap and self.confidence.heatmap_computed:
            self.canvas_state.energy_heatmap_image = self.confidence.build_energy_heatmap(
                current_angle=self.robot.angle,
                average_all_angles=self.canvas_state.average_heatmap,
                colormap='jet_r',
                sigma=20.0
            )

        # Render
        self.renderer.render(
            self.canvas_state,
            self.map_canvas,
            camera_simulator=self.camera,
            localization_engine=self.localization
        )

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
            self.canvas_state.estimated_x = result['x']
            self.canvas_state.estimated_y = result['y']
            self.canvas_state.estimated_angle = result['angle']
            self.canvas_state.retrieved_sample_idx = result['sample_idx']
            self.canvas_state.top_k_matches = result.get('top_k_matches', [])

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
        # Handle convergence mode
        if self.convergence_controller.is_converging:
            self.convergence_controller.step()
        else:
            # Normal movement controls
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

    def start_convergence(self):
        """Start the convergence process using MHN update rule on embedding"""
        if self.convergence_controller.is_converging:
            self.convergence_controller.stop_convergence()
        else:
            started = self.convergence_controller.start_convergence(self.current_camera_view)
            if started:
                self.converge_btn.config(text="Stop Convergence")

    def clear_convergence(self):
        """Clear convergence visualization from the canvas but keep the final red circle"""
        self.convergence_controller.clear_convergence()

        # Update display to remove strips but keep red circle
        self.update_map_display()
        self.set_status("✓ Convergence trace cleared (final pattern position kept)")

    def run(self):
        self.update_map_display()
        self.update_loop()
        self.root.mainloop()


if __name__ == "__main__":
    app = App()
    app.run()
