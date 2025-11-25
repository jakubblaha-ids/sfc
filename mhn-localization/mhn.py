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

    def train_sgd(self, observations, associated_data=None, num_patterns=None, learning_rate=0.01, epochs=100, batch_size=32, progress_callback=None):
        """
        Train the network using Gradient Descent to minimize energy on observations.
        This learns optimal patterns (prototypes) that represent the data.
        Also updates associated data (e.g., positions) if provided.

        Args:
            observations: numpy array of shape (N_samples, embedding_dim)
            associated_data: numpy array of shape (N_samples, data_dim) or None
            num_patterns: Number of patterns to learn. If None, uses len(observations)
            learning_rate: Learning rate for SGD
            epochs: Number of training epochs
            batch_size: Batch size for training
            progress_callback: Optional function(epoch, total_epochs, loss) called after each epoch

        Returns:
            list: History of average energy (loss) per epoch
        """
        observations = np.array(observations, dtype=np.float32)
        n_samples = len(observations)
        
        if num_patterns is None:
            num_patterns = n_samples
            
        # Initialize memory (patterns)
        # We can initialize with a random subset of observations
        indices = np.random.choice(n_samples, num_patterns, replace=(num_patterns > n_samples))
        self.memory = observations[indices].copy()
        self.num_patterns = num_patterns
        
        # Initialize associated data memory if provided
        self.memory_associated_data = None
        if associated_data is not None:
            associated_data = np.array(associated_data, dtype=np.float32)
            self.memory_associated_data = associated_data[indices].copy()
        
        # Normalize initial memory
        norms = np.linalg.norm(self.memory, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.memory_normalized = self.memory / norms
        
        loss_history = []
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            perm = np.random.permutation(n_samples)
            observations_shuffled = observations[perm]
            if associated_data is not None:
                associated_data_shuffled = associated_data[perm]
            
            total_energy = 0
            
            for i in range(0, n_samples, batch_size):
                batch = observations_shuffled[i:i+batch_size]
                
                # Normalize batch
                batch_norms = np.linalg.norm(batch, axis=1, keepdims=True)
                batch_norms = np.where(batch_norms == 0, 1, batch_norms)
                batch_normalized = batch / batch_norms
                
                # Compute similarities: (batch_size, num_patterns)
                similarities = np.dot(batch_normalized, self.memory_normalized.T)
                
                # Scale similarities
                scaled_sim = self.beta * similarities * np.sqrt(self.embedding_dim)
                
                # Compute softmax (attention)
                # (batch_size, num_patterns)
                max_sim = np.max(scaled_sim, axis=1, keepdims=True)
                exp_sim = np.exp(scaled_sim - max_sim)
                sum_exp = np.sum(exp_sim, axis=1, keepdims=True)
                attention = exp_sim / sum_exp
                
                # Compute energy for loss tracking
                # E = -log(sum(exp(scaled_sim)))
                log_sum_exp = max_sim + np.log(sum_exp)
                batch_energy = -np.sum(log_sum_exp)
                total_energy += batch_energy
                
                # Compute gradients for memory
                # The energy function is E = -lse(beta * x^T * M)
                # dE/dM = -beta * x * (softmax(beta * x^T * M))^T
                # We want to minimize energy, so we move in negative gradient direction
                # Update: M = M - lr * dE/dM = M + lr * beta * x * softmax^T
                
                # However, we are optimizing the NORMALIZED memory in the dot product, 
                # but we store unnormalized memory. 
                # For simplicity in this version, we assume we are optimizing the vectors directly
                # and re-normalize after update.
                
                # Gradient update:
                # We want to pull patterns closer to observations that attend to them.
                # Delta M = lr * (batch_normalized.T @ attention).T = lr * attention.T @ batch_normalized
                # Shape: (num_patterns, batch_size) @ (batch_size, dim) -> (num_patterns, dim)
                
                grad = np.dot(attention.T, batch_normalized)
                
                # Update memory
                # We average the gradient by batch size to keep LR stable
                self.memory += learning_rate * grad / batch_size
                
                # Re-normalize memory immediately to keep it on hypersphere
                norms = np.linalg.norm(self.memory, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                self.memory = self.memory / norms # Project back to unit sphere if desired, or just normalize for next step
                self.memory_normalized = self.memory # Since we just normalized it
                
                # Update associated data (positions)
                if self.memory_associated_data is not None:
                    batch_assoc = associated_data_shuffled[i:i+batch_size]
                    # Gradient for associated data: pull towards associated data of attending samples
                    # We want to minimize distance: E = 0.5 * sum(attn * ||data - mem||^2)
                    # dE/dM = -sum(attn * (data - mem)) = sum(attn * data) - sum(attn) * mem
                    
                    weighted_sum_data = np.dot(attention.T, batch_assoc)
                    sum_weights = np.sum(attention, axis=0, keepdims=True).T
                    
                    grad_assoc = weighted_sum_data - sum_weights * self.memory_associated_data
                    
                    self.memory_associated_data += learning_rate * grad_assoc / batch_size
            
            avg_loss = total_energy / n_samples
            loss_history.append(avg_loss)
            
            if progress_callback:
                progress_callback(epoch + 1, epochs, avg_loss)
            elif (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} complete, Loss: {avg_loss:.4f}")
                
        print(f"SGD Training complete. Learned {self.num_patterns} patterns.")
        return loss_history, indices

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
