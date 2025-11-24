## Implementation Section Detail

## Modern Hopfield Network

The core pattern matching is implemented in `ModernHopfieldNetwork` class using the continuous energy formulation. Patterns are stored as an $N \times D$ matrix where $N$ is the number of patterns and $D$ is the embedding dimension.

**Energy Function**: The network uses the energy function $E(x) = -\log\left(\sum_i \exp(\beta \cdot x^T \cdot \xi_i)\right)$ where $x$ is the query pattern, $\xi_i$ are stored patterns (normalized for cosine similarity), and $\beta$ is the inverse temperature parameter.

**Pattern Storage**: During training, patterns are stored directly in memory and normalized for cosine similarity computation. The L2 norm is computed for each pattern vector, avoiding division by zero with a minimum threshold.

**Retrieval Algorithm**: Given a query pattern, the network computes cosine similarities with all stored patterns, scales them by $\beta \cdot \sqrt{D}$ for numerical stability, and applies softmax to obtain attention weights. The top-k patterns with highest attention weights are returned along with their weights for position estimation.

**Temperature Parameter** ($\beta$): Controls retrieval selectivity. Higher values (e.g., $\beta = 50$) make the network focus sharply on the best match. Lower values allow broader consideration of similar patterns. 

## Observation Capture

Visual observations are captured using a raycasting camera simulator implemented in `CameraSimulator`. The robot's field of view is represented as a cone defined by an angle (30-360°) and maximum viewing distance.

**Raycasting**: For each of the 100 angular samples within the viewing cone, a ray is cast from the robot's position outward until hitting a wall (non-white pixel) or map boundary. The ray advances in discrete steps (5 pixels) checking for obstacles.

**Distance-Based Opacity**: Observed pixels are blended with white based on distance using $\text{opacity} = \exp(-d/d_{\text{max}} \cdot v)$ where $d$ is distance, $d_{\text{max}}$ is the maximum viewing distance, and $v$ is the visibility index parameter (0.01-1.0). This simulates distance-dependent visibility.

**Preprocessing**: Optional Gaussian blur is applied to observations to reduce noise sensitivity and improve localization accuracy.

**Output Format**: A 1×100 RGB image representing what the robot "sees" as a horizontal strip of colored pixels.

## Embedding Encoding

Camera observations are converted to fixed-length embedding vectors for the Hopfield network. Two encoding methods are supported:

- **Channel-separated** (default): Concatenates R, G, B channels sequentially: $[R_0, R_1, ..., R_{99}, G_0, G_1, ..., G_{99}, B_0, B_1, ..., B_{99}]$, resulting in a 300-dimensional vector.
- **Interleaved RGB**: Interleaves channels per pixel: $[R_0, G_0, B_0, R_1, G_1, B_1, ..., R_{99}, G_{99}, B_{99}]$, also 300-dimensional.

The encoding method is configurable and must remain consistent between training and querying.

## Localization Engine

Position estimation is handled by `LocalizationEngine`, which manages the full pipeline from sample collection to position prediction.

**Sample Collection**: During setup phase, the robot systematically visits positions on a grid (default 50-pixel stride) with multiple orientations (16 angles, 22.5° apart). At each position, a camera observation is captured and stored with its ground-truth coordinates.

**Training**: Collected observations are converted to embeddings and passed to the Hopfield network for storage. Training is instantaneous as Modern Hopfield Networks require no iterative optimization.

**Position Estimation**: Given a query observation:
1. Convert observation to embedding vector
2. Retrieve top-k closest matching patterns from Hopfield network using attention weights
3. If k=1, return the position of the best match
4. If k>1, compute weighted average position: $\hat{x} = \sum_{i=1}^k w_i x_i$ and $\hat{y} = \sum_{i=1}^k w_i y_i$ where $w_i$ are normalized attention weights and $(x_i, y_i)$ are matched positions
5. Orientation is taken from the best match

**Top-k Weighting**: Combining multiple matches (k=1 to 20, configurable) provides smoother position estimates when the query observation is ambiguous or falls between stored samples.

## Key Parameters

- **Beta ($\beta$)**: Inverse temperature (1.0-200.0, default 50.0). Controls sharpness of pattern retrieval.
- **Sample Stride**: Grid spacing for sample positions (default 50 pixels). Smaller values increase memory size but improve coverage.
- **Sample Rotations**: Number of orientations per position (default 16, equivalent to 22.5° steps). More rotations improve orientation accuracy.
- **Camera Samples**: Number of pixels in observation strip (fixed at 100).
- **Blur Radius**: Gaussian blur applied to observations (0-5.0, default 4.0). Reduces noise sensitivity.
- **Field of View**: Camera viewing angle (30-360°, default 120°). Wider FOV captures more context but may include irrelevant features.
- **Visibility Index**: Distance opacity factor (0.01-1.0, default 0.1). Lower values make distant objects fade more quickly.
- **Top-k**: Number of matches to combine (1-20, default 1). Higher values smooth predictions.

## Sampling Strategy

The `SamplingEngine` generates deterministic grid-based sampling positions. For a map of size 800×450 pixels with stride 50 and 16 rotations, this produces $(16 \times 9 \times 16) = 2,304$ training samples.

**Grid Generation**: Positions are centered in grid cells starting at half-stride offset to avoid map edges.

**Test Positions**: For confidence analysis, a denser grid (half the training stride) is used, providing more evaluation points without requiring full retraining.

**Heatmap Positions**: Separate from training samples, used for confidence visualization across the entire map at multiple angles.

## Data Persistence

Configuration parameters are stored in JSON format (`tmp/sfc_config.json`). This includes user-adjusted parameters (beta, blur, FOV, etc.) and the last used map path for automatic loading on startup. The map image itself is saved/loaded as PNG.

Patterns can optionally be saved to `patterns.npy` (NumPy binary format) for faster loading, avoiding repeated sampling.