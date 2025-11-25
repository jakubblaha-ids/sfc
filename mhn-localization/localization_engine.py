import numpy as np
from .mhn import ModernHopfieldNetwork
from .constants import INTERLEAVED_RGB, MAP_WIDTH, MAP_HEIGHT
import traceback


class LocalizationEngine:
    """
    Manages localization using Modern Hopfield Networks.
    Handles sample storage, training, and retrieval.
    Pure logic class without any UI dependencies.
    """

    def __init__(self, beta=1.0, interleaved_rgb=INTERLEAVED_RGB):
        """
        Initialize localization engine.

        Args:
            beta: Inverse temperature parameter for Hopfield network
            interleaved_rgb: Whether to use interleaved RGB encoding
        """
        self.beta = beta
        self.interleaved_rgb = interleaved_rgb

        self.sample_positions = []
        self.sample_embeddings = []
        self.sample_views = []

        self.hopfield_network = None
        self.is_trained = False

        self.last_retrieved_idx = None
        self.sample_similarities = None

    def add_sample(self, x, y, angle, camera_view):
        """
        Add a sample position and its camera view to the database.

        Args:
            x: X position
            y: Y position
            angle: Angle in degrees
            camera_view: PIL Image of camera view
        """
        embedding = self._create_embedding(camera_view)

        self.sample_positions.append((x, y, angle))
        self.sample_embeddings.append(embedding)
        self.sample_views.append(camera_view.copy())

    def clear_samples(self):
        """Clear all samples."""
        self.sample_positions = []
        self.sample_embeddings = []
        self.sample_views = []
        self.is_trained = False
        self.hopfield_network = None
        self.last_retrieved_idx = None
        self.sample_similarities = None

    def train(self):
        """
        Train the Hopfield Network using collected samples.

        Returns:
            bool: True if training successful, False otherwise
        """
        if len(self.sample_embeddings) == 0:
            return False

        embedding_dim = len(self.sample_embeddings[0])
        self.hopfield_network = ModernHopfieldNetwork(
            embedding_dim, beta=self.beta)

        try:
            self.hopfield_network.train(self.sample_embeddings)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Training error: {e}")
            return False

    def train_sgd(self, num_patterns=None, learning_rate=0.01, epochs=100, progress_callback=None):
        """
        Train the Hopfield Network using SGD on collected samples.
        
        Args:
            num_patterns: Number of patterns to learn
            learning_rate: Learning rate
            epochs: Number of epochs
            progress_callback: Optional callback(epoch, total, loss)
            
        Returns:
            tuple: (success, loss_history)
        """
        if len(self.sample_embeddings) == 0:
            return False, []
            
        embedding_dim = len(self.sample_embeddings[0])
        self.hopfield_network = ModernHopfieldNetwork(
            embedding_dim, beta=self.beta)
            
        # Encode positions for training
        encoded_positions = self._encode_positions(self.sample_positions)
            
        try:
            loss_history, indices = self.hopfield_network.train_sgd(
                self.sample_embeddings, 
                associated_data=encoded_positions,
                num_patterns=num_patterns,
                learning_rate=learning_rate,
                epochs=epochs,
                progress_callback=progress_callback
            )
            
            # Update internal state with trained prototypes
            # 1. Update embeddings
            self.sample_embeddings = list(self.hopfield_network.memory)
            
            # 2. Update positions (decode from associated data)
            if self.hopfield_network.memory_associated_data is not None:
                self.sample_positions = [self._decode_position(p) for p in self.hopfield_network.memory_associated_data]
            
            # 3. Update views (approximate by using views from initialization indices)
            # This is an approximation since the prototype has drifted, but it gives a visual reference
            new_views = []
            for idx in indices:
                new_views.append(self.sample_views[idx])
            self.sample_views = new_views
            
            self.is_trained = True
            return True, loss_history
        except Exception as e:
            print(f"SGD Training error: {e}")
            traceback.print_exc()
            return False, []

    def localize(self, camera_view, top_k=1):
        """
        Perform localization given a camera view.
        Retrieves the closest matching pattern(s) and estimates position.

        Args:
            camera_view: PIL Image of current camera view
            top_k: Number of best matches to combine for position prediction

        Returns:
            Dictionary with keys: 'x', 'y', 'angle', 'confidence', 'sample_idx'
            Returns None if not trained or error occurs
        """
        if not self.is_trained or self.hopfield_network is None:
            return None

        if camera_view is None:
            return None

        try:
            query_embedding = self._create_embedding(camera_view)

            # Use current sample positions (which are now updated after training)
            reference_positions = self.sample_positions
            num_patterns = len(self.sample_embeddings)

            all_indices, all_weights = self.hopfield_network.retrieve(
                query_embedding, top_k=num_patterns)

            self.sample_similarities = np.zeros(num_patterns)
            for idx, weight in zip(all_indices, all_weights):
                self.sample_similarities[idx] = weight

            # Compute energy for this query
            energy = self.hopfield_network.get_energy(query_embedding)

            actual_top_k = min(top_k, len(all_indices))
            top_indices = all_indices[:actual_top_k]
            top_weights = all_weights[:actual_top_k]

            if actual_top_k == 1:
                best_idx = top_indices[0]
                best_weight = top_weights[0]
                self.last_retrieved_idx = best_idx
                x, y, angle = reference_positions[best_idx]
                top_k_matches = []
            else:
                weight_sum = np.sum(top_weights)
                if weight_sum == 0:
                    weight_sum = 1.0

                normalized_weights = top_weights / weight_sum

                x_weighted = 0.0
                y_weighted = 0.0
                
                # For angle averaging, we need vector components
                sin_sum = 0.0
                cos_sum = 0.0

                top_k_matches = []
                for idx, weight in zip(top_indices, normalized_weights):
                    pos_x, pos_y, pos_angle = reference_positions[idx]
                    x_weighted += pos_x * weight
                    y_weighted += pos_y * weight
                    
                    rad = np.radians(pos_angle)
                    sin_sum += np.sin(rad) * weight
                    cos_sum += np.cos(rad) * weight
                    
                    top_k_matches.append({
                        'x': pos_x,
                        'y': pos_y,
                        'angle': pos_angle,
                        'weight': float(weight),
                        'sample_idx': int(idx)
                    })

                best_idx = top_indices[0]
                best_weight = top_weights[0]
                self.last_retrieved_idx = best_idx

                x = x_weighted
                y = y_weighted
                angle = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360

            return {
                'x': x,
                'y': y,
                'angle': angle,
                'confidence': float(best_weight),
                'energy': float(energy),
                'sample_idx': int(best_idx),
                'top_k_matches': top_k_matches
            }

        except Exception as e:
            print(f"Localization error: {e}")
            return None

    def _encode_positions(self, positions):
        """
        Encode positions (x, y, angle) into a continuous vector format for averaging.
        Format: [x/W, y/H, sin(angle), cos(angle)]
        """
        encoded = []
        for x, y, angle in positions:
            rad = np.radians(angle)
            encoded.append([
                x / MAP_WIDTH,
                y / MAP_HEIGHT,
                np.sin(rad),
                np.cos(rad)
            ])
        return np.array(encoded, dtype=np.float32)

    def _decode_position(self, encoded_pos):
        """
        Decode encoded position vector back to (x, y, angle).
        """
        nx, ny, ns, nc = encoded_pos
        x = nx * MAP_WIDTH
        y = ny * MAP_HEIGHT
        angle = np.degrees(np.arctan2(ns, nc)) % 360
        return (x, y, angle)

    def get_sample_info(self, sample_idx):
        """
        Get information about a specific sample.

        Args:
            sample_idx: Index of the sample

        Returns:
            Dictionary with 'x', 'y', 'angle', 'view' or None if invalid index
        """
        if sample_idx < 0 or sample_idx >= len(self.sample_positions):
            return None

        x, y, angle = self.sample_positions[sample_idx]
        return {
            'x': x,
            'y': y,
            'angle': angle,
            'view': self.sample_views[sample_idx]
        }

    def get_samples_at_position(self, x, y, tolerance=1.0):
        """
        Get all samples at a given position (with different angles).

        Args:
            x: X position
            y: Y position
            tolerance: Distance tolerance for position matching

        Returns:
            List of dictionaries with 'idx', 'angle', 'view' sorted by angle
        """
        samples_at_pos = []

        for idx, (sample_x, sample_y, angle) in enumerate(self.sample_positions):
            distance = ((sample_x - x) ** 2 + (sample_y - y) ** 2) ** 0.5
            if distance <= tolerance:
                samples_at_pos.append({
                    'idx': idx,
                    'angle': angle,
                    'view': self.sample_views[idx]
                })

        samples_at_pos.sort(key=lambda s: s['angle'])
        return samples_at_pos

    def get_num_samples(self):
        """Get the number of stored samples."""
        return len(self.sample_positions)

    def update_beta(self, beta):
        """
        Update beta parameter for the Hopfield network.

        Args:
            beta: New beta value
        """
        self.beta = beta
        if self.hopfield_network is not None:
            self.hopfield_network.beta = beta

    def _create_embedding(self, camera_view):
        """
        Create an embedding vector from a camera view image.

        Args:
            camera_view: PIL Image (1D strip)

        Returns:
            numpy array representing the embedding
        """
        pixels = list(camera_view.getdata())

        if self.interleaved_rgb:
            embedding = []
            for r, g, b in pixels:
                embedding.extend([r, g, b])
        else:
            r_channel = [pixel[0] for pixel in pixels]
            g_channel = [pixel[1] for pixel in pixels]
            b_channel = [pixel[2] for pixel in pixels]
            embedding = r_channel + g_channel + b_channel

        return np.array(embedding, dtype=np.float32)
