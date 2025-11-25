import numpy as np


class ModernHopfieldNetwork:
    """
    Modern Hopfield Network for pattern retrieval.
    Implements continuous Modern Hopfield Network energy function/update rule.
    """

    def __init__(self, embedding_dim, beta=1.0):
        """
        Initialize the Modern Hopfield Network.

        Args:
            embedding_dim: Dimension of the embedding vectors
            beta: Inverse temperature parameter (controls sharpness of retrieval)
        """
        self.embedding_dim = embedding_dim
        self.beta = beta  # Inverse temperature for softmax
        # Matrix of stored patterns (N_patterns x embedding_dim)
        self.memory = None
        self.memory_normalized = None  # Normalized patterns for cosine similarity
        self.num_patterns = 0

    def train(self, patterns):
        """
        Train the network by storing patterns in memory.
        For Modern Hopfield Networks, training is simply storing the patterns.
        We also compute normalized versions for cosine similarity.

        Args:
            patterns: list or numpy array of shape (N_patterns, embedding_dim)

        Returns:
            None
        """
        if patterns is None or len(patterns) == 0:
            raise ValueError("No patterns provided for training")

        self.num_patterns = len(patterns)

        # Convert to numpy array if needed
        self.memory = np.array(patterns, dtype=np.float32)

        # Normalize patterns for cosine similarity
        # Compute L2 norm for each pattern
        norms = np.linalg.norm(self.memory, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        self.memory_normalized = self.memory / norms

        print(
            f"Training complete: {self.num_patterns} patterns stored and normalized")

    def retrieve(self, query, top_k=1):
        """
        Retrieve the closest matching pattern(s) from memory using Modern Hopfield Network.
        Uses softmax attention mechanism with cosine similarity.

        Args:
            query: numpy array of shape (embedding_dim,) - the query pattern
            top_k: Number of top matches to retrieve (default: 1)

        Returns:
            tuple: (indices, similarities)
                - indices: array of shape (top_k,) with indices of best matches
                - similarities: array of shape (top_k,) with attention weights/scores
        """
        if self.memory is None:
            raise ValueError("Network not trained. Call train() first.")

        if len(query) != self.embedding_dim:
            raise ValueError(
                f"Query dimension {len(query)} does not match embedding dimension {self.embedding_dim}")

        # Normalize query for cosine similarity
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            query_norm = 1
        query_normalized = query / query_norm

        # Compute cosine similarities between query and all stored patterns
        # similarities shape: (N_patterns,)
        similarities = np.dot(self.memory_normalized, query_normalized)

        # Apply scaled softmax (Modern Hopfield update rule)
        # Scale by beta (inverse temperature) and dimension for numerical stability
        scaled_similarities = self.beta * \
            similarities * np.sqrt(self.embedding_dim)

        # Compute softmax to get attention weights
        # Subtract max for numerical stability
        exp_similarities = np.exp(
            scaled_similarities - np.max(scaled_similarities))
        attention_weights = exp_similarities / np.sum(exp_similarities)

        # Get top-k indices based on attention weights
        top_indices = np.argsort(attention_weights)[-top_k:][::-1]
        top_weights = attention_weights[top_indices]

        return top_indices, top_weights

    def get_energy(self, query):
        """
        Calculate the energy of a query pattern using Modern Hopfield Network energy function.
        Energy function: E(x) = -log(sum_i exp(beta * x^T * memory_i))
        Lower energy indicates better match.

        Args:
            query: numpy array of shape (embedding_dim,)

        Returns:
            float: Energy value (negative log-sum-exp of similarities)
        """
        if self.memory is None:
            raise ValueError("Network not trained. Call train() first.")

        # Normalize query
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            query_norm = 1
        query_normalized = query / query_norm

        # Compute cosine similarities
        similarities = np.dot(self.memory_normalized, query_normalized)

        # Scale by beta and dimension
        scaled_similarities = self.beta * \
            similarities * np.sqrt(self.embedding_dim)

        # Compute energy: E = -log(sum(exp(scaled_similarities)))
        # Use log-sum-exp trick for numerical stability
        max_sim = np.max(scaled_similarities)
        log_sum_exp = max_sim + np.log(
            np.sum(np.exp(scaled_similarities - max_sim)))
        energy = -log_sum_exp

        return energy

    def update_step(self, query):
        """
        Perform one step of the Modern Hopfield Network update rule.
        Update rule: x^(t+1) = softmax(beta * M^T * x^(t)) * M
        where M is the memory matrix (stored patterns).

        Args:
            query: numpy array of shape (embedding_dim,) - current state vector

        Returns:
            tuple: (updated_query, converged)
                - updated_query: numpy array of shape (embedding_dim,) - new state
                - converged: bool - True if converged to a fixed point
        """
        if self.memory is None:
            raise ValueError("Network not trained. Call train() first.")

        # Normalize query for cosine similarity
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            query_norm = 1
        query_normalized = query / query_norm

        # Compute cosine similarities between query and all stored patterns
        similarities = np.dot(self.memory_normalized, query_normalized)

        # Apply scaled softmax (Modern Hopfield update rule)
        scaled_similarities = self.beta * similarities * np.sqrt(self.embedding_dim)

        # Compute softmax to get attention weights
        exp_similarities = np.exp(scaled_similarities - np.max(scaled_similarities))
        attention_weights = exp_similarities / np.sum(exp_similarities)

        # Update rule: weighted sum of stored patterns
        updated_query = np.dot(attention_weights, self.memory)

        # Check convergence: if the update doesn't change the state significantly
        convergence_threshold = 1e-6
        change = np.linalg.norm(updated_query - query)
        converged = change < convergence_threshold

        return updated_query, converged

    def converge(self, query, max_iterations=100):
        """
        Iteratively apply the update rule until convergence.
        This finds the fixed point (attractor) in the energy landscape.

        Args:
            query: numpy array of shape (embedding_dim,) - initial state vector
            max_iterations: Maximum number of update steps (default: 100)

        Returns:
            dict with keys:
                - 'final_state': final converged embedding vector
                - 'history': list of states at each iteration
                - 'converged': whether it actually converged
                - 'iterations': number of iterations performed
                - 'best_match_idx': index of the pattern closest to final state
        """
        if self.memory is None:
            raise ValueError("Network not trained. Call train() first.")

        current_state = query.copy()
        history = [current_state.copy()]
        converged = False

        for _ in range(max_iterations):
            current_state, converged = self.update_step(current_state)
            history.append(current_state.copy())

            if converged:
                break

        # Find which stored pattern the final state is closest to
        final_norm = np.linalg.norm(current_state)
        if final_norm == 0:
            final_norm = 1
        final_normalized = current_state / final_norm

        similarities = np.dot(self.memory_normalized, final_normalized)
        best_match_idx = int(np.argmax(similarities))

        return {
            'final_state': current_state,
            'history': history,
            'converged': converged,
            'iterations': len(history) - 1,
            'best_match_idx': best_match_idx
        }

    def clear(self):
        """Clear the network memory."""
        self.memory = None
        self.num_patterns = 0
