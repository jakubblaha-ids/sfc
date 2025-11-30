import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import math
from .canvas_state import CanvasState
from .constants import (
    COLOR_SAMPLE_DOT,
    COLOR_ROBOT_GT,
)


class CanvasRenderer:
    """
    Handles all canvas rendering logic based on a CanvasState.
    Completely decoupled from application state management.
    """

    def __init__(self):
        self._tk_map_image = None
        self._tk_robot_image = None
        self._hovered_strip_images = []
        self._convergence_strip_images = []
        self.robot_image = None

    def set_robot_image(self, robot_image: Image.Image):
        """Set the robot image for rendering"""
        self.robot_image = robot_image

    def update_map_with_noise(self, state: CanvasState):
        """Create a cached map with noise applied for raytracing"""
        if not state.apply_noise or len(state.noise_circles) == 0:
            state.map_with_noise = None
            return

        state.map_with_noise = state.current_map_image.copy()
        draw = ImageDraw.Draw(state.map_with_noise)
        for x, y, radius in state.noise_circles:
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill='black',
                outline='black'
            )

    def get_map_for_raytracing(self, state: CanvasState) -> Image.Image:
        """Get the appropriate map for raytracing (with or without noise)"""
        if state.map_with_noise is not None:
            return state.map_with_noise
        return state.current_map_image

    def render(self, state: CanvasState, canvas: tk.Canvas,
               camera_simulator=None, localization_engine=None):
        """
        Render the complete canvas based on the provided state.

        Args:
            state: CanvasState containing all rendering data
            canvas: tkinter Canvas to render to
            camera_simulator: Optional CameraSimulator for drawing viewing cone
            localization_engine: Optional LocalizationEngine for sample info
        """
        # Build the display image with all overlays
        display_image = state.current_map_image.copy()

        if state.apply_noise:
            self._draw_noise_circles_on_image(display_image, state.noise_circles)

        if camera_simulator:
            self._draw_viewing_cone_on_image(
                display_image, state, camera_simulator)

        if state.show_confidence_heatmap and state.confidence_heatmap_image:
            display_image.paste(state.confidence_heatmap_image, (0, 0), state.confidence_heatmap_image)

        if state.show_energy_heatmap and state.energy_heatmap_image:
            display_image.paste(state.energy_heatmap_image, (0, 0), state.energy_heatmap_image)

        self._draw_sample_dots_on_image(display_image, state)
        self._display_scaled_map_image(display_image, canvas, state)

        self._draw_test_position_dots(canvas, state)
        self._draw_top_k_interpolation_lines(canvas, state)
        self._draw_converged_pattern_highlight(canvas, state)
        self._draw_convergence_path(canvas, state)
        self._draw_robot(canvas, state)

        if localization_engine:
            self._draw_hovered_sample_visualization(
                canvas, state, localization_engine)

    def _display_scaled_map_image(self, display_image: Image.Image,
                                  canvas: tk.Canvas, state: CanvasState):
        """Scale and display the map image to fit canvas while maintaining aspect ratio"""
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            self._tk_map_image = ImageTk.PhotoImage(display_image)
            canvas.delete("all")
            canvas.create_image(0, 0, image=self._tk_map_image, anchor="nw")
            return

        img_width, img_height = display_image.size
        width_scale = canvas_width / img_width
        height_scale = canvas_height / img_height

        scale_factor = min(width_scale, height_scale)
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)

        offset_x = (canvas_width - new_width) / 2
        offset_y = (canvas_height - new_height) / 2

        state.map_scale_factor = scale_factor
        state.map_offset_x = offset_x
        state.map_offset_y = offset_y

        scaled_image = display_image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS)
        self._tk_map_image = ImageTk.PhotoImage(scaled_image)

        canvas.delete("all")
        canvas.create_image(offset_x, offset_y, image=self._tk_map_image, anchor="nw")

    def _draw_noise_circles_on_image(self, image: Image.Image,
                                     noise_circles: list):
        """Draw random noise circles on the image"""
        draw = ImageDraw.Draw(image)
        for x, y, radius in noise_circles:
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill='black',
                outline='black'
            )

    def _draw_viewing_cone_on_image(self, image: Image.Image,
                                    state: CanvasState, camera_simulator):
        """Draw the viewing cone with transparency directly on a PIL image"""
        map_image = state.map_with_noise if state.map_with_noise else state.current_map_image
        points = camera_simulator.get_viewing_cone_points(
            state.robot_x, state.robot_y, state.robot_angle, map_image)

        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        cone_color = (0, 0, 255, 127)
        draw.polygon(points, fill=cone_color, outline=None)

        image.paste(overlay, (0, 0), overlay)

    def _draw_sample_dots_on_image(self, image: Image.Image, state: CanvasState):
        """Draw sample dots directly on the PIL image (batch rendering)"""
        if not state.sample_positions:
            return

        draw = ImageDraw.Draw(image)

        sample_similarities = state.sample_similarities
        max_similarity = None
        if sample_similarities is not None and len(sample_similarities) > 0:
            max_similarity = max(sample_similarities.max(), 1e-6)

        dot_color = COLOR_SAMPLE_DOT

        for i, (x, y, _angle) in enumerate(state.sample_positions):
            if sample_similarities is not None and i < len(sample_similarities):
                similarity = sample_similarities[i]
                min_radius = 1
                max_radius = 5
                normalized_similarity = similarity / max_similarity
                radius = min_radius + (max_radius - min_radius) * normalized_similarity
            else:
                radius = 3

            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=dot_color,
                outline=dot_color
            )

    def _draw_test_position_dots(self, canvas: tk.Canvas, state: CanvasState):
        """Draw blue squares at test positions on the canvas"""
        if not state.show_test_positions:
            return

        dot_radius = 2

        for x, y in state.test_positions:
            canvas_x, canvas_y = state.image_to_canvas_coords(x, y)
            scaled_radius = dot_radius * state.map_scale_factor

            canvas.create_rectangle(
                canvas_x - scaled_radius,
                canvas_y - scaled_radius,
                canvas_x + scaled_radius,
                canvas_y + scaled_radius,
                fill="#0000FF",
                outline="#0000FF",
                tags="test_position_dot"
            )

    def _draw_top_k_interpolation_lines(self, canvas: tk.Canvas, state: CanvasState):
        """Draw black lines from top k matches to the final predicted position"""
        if not state.top_k_matches or state.estimated_x is None or state.estimated_y is None:
            return

        canvas_est_x, canvas_est_y = state.image_to_canvas_coords(
            state.estimated_x, state.estimated_y)

        for match in state.top_k_matches:
            match_x = match['x']
            match_y = match['y']
            canvas_match_x, canvas_match_y = state.image_to_canvas_coords(
                match_x, match_y)

            canvas.create_line(
                canvas_match_x, canvas_match_y,
                canvas_est_x, canvas_est_y,
                fill="#000000",
                width=max(1, int(1 * state.map_scale_factor)),
                tags="top_k_line"
            )

    def _draw_robot(self, canvas: tk.Canvas, state: CanvasState):
        """Draw the robot and estimated position"""
        canvas_robot_x, canvas_robot_y = state.image_to_canvas_coords(
            state.robot_x, state.robot_y)
        scaled_radius = state.robot_radius * state.map_scale_factor

        # Draw actual robot
        if self.robot_image is not None:
            rotated_image = self.robot_image.rotate(
                -state.robot_angle, expand=True)

            robot_size = int(50 * state.map_scale_factor)
            rotated_image = rotated_image.resize(
                (robot_size, robot_size),
                Image.Resampling.LANCZOS)

            self._tk_robot_image = ImageTk.PhotoImage(rotated_image)

            canvas.create_image(
                canvas_robot_x, canvas_robot_y,
                image=self._tk_robot_image,
                anchor="center",
                tags="robot"
            )
        else:
            canvas.create_oval(
                canvas_robot_x - scaled_radius,
                canvas_robot_y - scaled_radius,
                canvas_robot_x + scaled_radius,
                canvas_robot_y + scaled_radius,
                fill=COLOR_ROBOT_GT,
                outline=COLOR_ROBOT_GT,
                tags="robot"
            )

        # Draw estimated position
        if state.estimated_x is not None and state.estimated_y is not None:
            canvas_est_x, canvas_est_y = state.image_to_canvas_coords(
                state.estimated_x, state.estimated_y)

            if state.estimated_angle is not None:
                line_length = 20 * state.map_scale_factor
                angle_rad = math.radians(state.estimated_angle)
                end_x = canvas_est_x + line_length * math.cos(angle_rad)
                end_y = canvas_est_y + line_length * math.sin(angle_rad)
                canvas.create_line(
                    canvas_est_x, canvas_est_y,
                    end_x, end_y,
                    fill="#800080",
                    width=max(2, int(2 * state.map_scale_factor)),
                    tags="estimated_direction"
                )

            canvas.create_oval(
                canvas_est_x - scaled_radius,
                canvas_est_y - scaled_radius,
                canvas_est_x + scaled_radius,
                canvas_est_y + scaled_radius,
                fill="#00FF00",
                outline="#00FF00",
                tags="estimated"
            )

    def _draw_hovered_sample_visualization(self, canvas: tk.Canvas,
                                           state: CanvasState, localization_engine):
        """Draw visualization of all directions for the hovered sample position"""
        if state.hovered_sample_idx is None:
            return

        sample_info = localization_engine.get_sample_info(state.hovered_sample_idx)
        if not sample_info:
            return

        x, y = sample_info['x'], sample_info['y']

        # Highlight the hovered position on the map
        canvas_x, canvas_y = state.image_to_canvas_coords(x, y)
        highlight_radius = 10 * state.map_scale_factor

        canvas.create_oval(
            canvas_x - highlight_radius,
            canvas_y - highlight_radius,
            canvas_x + highlight_radius,
            canvas_y + highlight_radius,
            outline="#FFFF00",
            width=max(2, int(2 * state.map_scale_factor)),
            tags="hovered_highlight"
        )

        # Get all samples at this position
        samples_at_pos = localization_engine.get_samples_at_position(x, y)

        if not samples_at_pos:
            return

        # Draw all direction samples in top-left corner
        strip_height = 15
        margin = 10
        start_x = margin
        start_y = margin

        self._hovered_strip_images = []

        for i, sample in enumerate(samples_at_pos):
            view = sample['view']
            angle = sample['angle']

            view_width = view.size[0]

            resized_view = view.resize(
                (view_width, strip_height),
                Image.Resampling.NEAREST
            )

            tk_view = ImageTk.PhotoImage(resized_view)

            y_pos = start_y + i * (strip_height + 2)
            canvas.create_image(
                start_x, y_pos,
                image=tk_view,
                anchor="nw",
                tags="hovered_direction_strip"
            )

            self._hovered_strip_images.append(tk_view)

            label_x = start_x + view_width + 5
            canvas.create_text(
                label_x, y_pos + strip_height // 2,
                text=f"{angle:.0f}°",
                fill="#FF0000",
                anchor="w",
                font=("Arial", 10),
                tags="hovered_direction_label"
            )

    def _draw_convergence_path(self, canvas: tk.Canvas, state: CanvasState):
        """Draw convergence steps as camera view strips"""
        if not state.convergence_visualization_strips:
            return

        strip_height = 15
        margin = 10
        start_x = margin
        start_y = margin

        canvas_height = canvas.winfo_height()
        max_strips_per_column = max(1, (canvas_height - margin) // (strip_height + 2))

        self._convergence_strip_images = []

        for i, view in enumerate(state.convergence_visualization_strips):
            column = i // max_strips_per_column
            row = i % max_strips_per_column

            view_width = view.size[0]

            column_width = view_width + 60
            x_pos = start_x + column * column_width
            y_pos = start_y + row * (strip_height + 2)

            resized_view = view.resize(
                (view_width, strip_height),
                Image.Resampling.NEAREST
            )

            tk_view = ImageTk.PhotoImage(resized_view)

            canvas.create_image(
                x_pos, y_pos,
                image=tk_view,
                anchor="nw",
                tags="convergence_step_strip"
            )

            self._convergence_strip_images.append(tk_view)

            label_x = x_pos + view_width + 5
            color = "#00FF00" if i == len(state.convergence_visualization_strips) - 1 else "#FFA500"
            canvas.create_text(
                label_x, y_pos + strip_height // 2,
                text=f"Step {i}",
                fill=color,
                anchor="w",
                font=("Arial", 9),
                tags="convergence_step_label"
            )

    def _draw_converged_pattern_highlight(self, canvas: tk.Canvas, state: CanvasState):
        """Draw a red circle around the converged pattern position with direction line"""
        if state.converged_position is None:
            return

        x, y, angle = state.converged_position
        canvas_x, canvas_y = state.image_to_canvas_coords(x, y)

        radius = 20 * state.map_scale_factor

        canvas.create_oval(
            canvas_x - radius,
            canvas_y - radius,
            canvas_x + radius,
            canvas_y + radius,
            outline="#FF0000",
            width=max(2, int(3 * state.map_scale_factor)),
            tags="converged_pattern_highlight"
        )

        angle_rad = math.radians(angle)
        end_x = canvas_x + radius * math.cos(angle_rad)
        end_y = canvas_y + radius * math.sin(angle_rad)

        canvas.create_line(
            canvas_x, canvas_y,
            end_x, end_y,
            fill="#FF0000",
            width=max(2, int(3 * state.map_scale_factor)),
            tags="converged_pattern_direction"
        )
