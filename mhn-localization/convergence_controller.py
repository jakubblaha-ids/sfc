import numpy as np
from .utils import embedding_to_image

class ConvergenceController:
    def __init__(self, localization_engine, canvas_state, status_callback, update_display_callback, on_convergence_finished=None):
        self.localization = localization_engine
        self.canvas_state = canvas_state
        self.status_callback = status_callback
        self.update_display_callback = update_display_callback
        self.on_convergence_finished = on_convergence_finished
        
        self.is_converging = False
        self.convergence_step = 0
        self.convergence_embedding = None
        self.convergence_history = []
        self._convergence_strip_images = []

    def start_convergence(self, current_camera_view):
        """Start the convergence process using MHN update rule on embedding"""
        if not self.localization.is_trained:
            self.status_callback("⚠️ Network not trained.")
            return False

        if self.is_converging:
            # Stop convergence
            self.stop_convergence()
            return False
        else:
            # Start convergence
            self.is_converging = True
            self.convergence_step = 0
            self.canvas_state.convergence_visualization_strips = []
            self.canvas_state.converged_position = None
            
            # Initialize with current camera view
            self.convergence_embedding = self.localization.create_embedding(current_camera_view)
            self.convergence_history = [self.convergence_embedding.copy()]
            
            # Add initial embedding as first strip
            initial_view = embedding_to_image(self.convergence_embedding, self.localization.interleaved_rgb)
            self.canvas_state.convergence_visualization_strips.append(initial_view)
            
            self.status_callback("⟳ Converging to pattern...")
            return True

    def stop_convergence(self):
        """Stop the convergence process"""
        self.is_converging = False
        self.status_callback(f"✓ Convergence stopped after {self.convergence_step} steps")
        self.convergence_embedding = None
        # Clear visualization
        self.canvas_state.convergence_visualization_strips = []
        self.canvas_state.converged_position = None
        self._convergence_strip_images = []
        self.update_display_callback()
        if self.on_convergence_finished:
            self.on_convergence_finished()

    def step(self):
        """Perform one step of the MHN update rule on the embedding vector"""
        if not self.is_converging:
            return

        if not self.localization.is_trained or not self.localization.hopfield_network:
            self.is_converging = False
            return

        # Perform one update step
        updated_embedding, converged = self.localization.hopfield_network.update_step(
            self.convergence_embedding
        )

        self.convergence_step += 1
        self.convergence_embedding = updated_embedding
        self.convergence_history.append(updated_embedding.copy())

        # Add updated embedding as a new strip
        updated_view = embedding_to_image(updated_embedding, self.localization.interleaved_rgb)
        self.canvas_state.convergence_visualization_strips.append(updated_view)

        # Find which pattern this embedding is closest to
        indices, weights = self.localization.hopfield_network.retrieve(updated_embedding, top_k=1)
        best_match_idx = indices[0]
        confidence = weights[0]
        
        target_x = 0
        target_y = 0
        target_angle = 0

        # Get the position of the best match
        if best_match_idx < len(self.localization.sample_positions):
            target_x, target_y, target_angle = self.localization.sample_positions[best_match_idx]
            self.canvas_state.estimated_x = target_x
            self.canvas_state.estimated_y = target_y
            self.canvas_state.estimated_angle = target_angle
            self.canvas_state.retrieved_sample_idx = best_match_idx

        # Update display
        self.update_display_callback()

        # Update status
        change = np.linalg.norm(
            updated_embedding - self.convergence_history[-2]) if len(self.convergence_history) > 1 else 0
        self.status_callback(f"⟳ Converging... Step {self.convergence_step}, Change: {change:.6f}, Confidence: {confidence*100:.1f}%")

        # Check convergence
        if converged or self.convergence_step > 100:
            self.is_converging = False
            if converged:
                self.status_callback(f"✓ Converged in {self.convergence_step} steps! Position: ({target_x:.1f}, {target_y:.1f}), Confidence: {confidence*100:.1f}%")
            else:
                self.status_callback(f"⚠️ Stopped after {self.convergence_step} iterations")

            if self.on_convergence_finished:
                self.on_convergence_finished()

            # Store the converged position and angle for visualization
            self.canvas_state.converged_position = (target_x, target_y, target_angle)
            self.update_display_callback()

            # Clean up
            self.convergence_embedding = None

    def clear_convergence(self):
        """Clear convergence visualization from the canvas but keep the final red circle"""
        # Stop convergence if running
        if self.is_converging:
            self.is_converging = False
            if self.on_convergence_finished:
                self.on_convergence_finished()

        # Clear convergence strips but keep the converged position (red circle)
        self.canvas_state.convergence_visualization_strips = []
        self.convergence_step = 0
        self._convergence_strip_images = []
        self.convergence_embedding = None

        # Update display to remove strips but keep red circle
        self.update_display_callback()
        self.status_callback("✓ Convergence trace cleared (final pattern position kept)")
