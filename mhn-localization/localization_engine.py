import numpy as np
from .hopfield import ModernHopfieldNetwork
from .constants import INTERLEAVED_RGB


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

            all_indices, all_weights = self.hopfield_network.retrieve(
                query_embedding, top_k=len(self.sample_embeddings))

            self.sample_similarities = np.zeros(len(self.sample_embeddings))
            for idx, weight in zip(all_indices, all_weights):
                self.sample_similarities[idx] = weight

            actual_top_k = min(top_k, len(all_indices))
            top_indices = all_indices[:actual_top_k]
            top_weights = all_weights[:actual_top_k]

            if actual_top_k == 1:
                best_idx = top_indices[0]
                best_weight = top_weights[0]
                self.last_retrieved_idx = best_idx
                x, y, angle = self.sample_positions[best_idx]
                top_k_matches = []
            else:
                weight_sum = np.sum(top_weights)
                if weight_sum == 0:
                    weight_sum = 1.0

                normalized_weights = top_weights / weight_sum

                x_weighted = 0.0
                y_weighted = 0.0

                top_k_matches = []
                for idx, weight in zip(top_indices, normalized_weights):
                    pos_x, pos_y, pos_angle = self.sample_positions[idx]
                    x_weighted += pos_x * weight
                    y_weighted += pos_y * weight
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
                _, _, angle = self.sample_positions[best_idx]

            return {
                'x': x,
                'y': y,
                'angle': angle,
                'confidence': float(best_weight),
                'sample_idx': int(best_idx),
                'top_k_matches': top_k_matches
            }

        except Exception as e:
            print(f"Localization error: {e}")
            return None

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
