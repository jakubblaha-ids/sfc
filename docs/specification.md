# Specification: Robot Localization via Modern Hopfield Networks

## 1. Overview
This project implements a robot localization demo using Modern Hopfield Networks (MHN). The system simulates a robot moving in a 2D environment, capturing visual observations, and using an MHN-based associative memory to determine its location by matching current observations against a stored database of views.

## 2. Functional Requirements

### 2.1. Environment & Simulation
- **Map**: A 2D binary grid (800x450 pixels) representing the environment, loaded from a PNG file (black=wall, white=free).
- **Robot**: An entity capable of movement (x, y) and rotation (angle) within the map.
- **Sensor**: A 1D Raycast sensor with a 90-degree Field of View (FOV). It captures a 1D strip of values and does not see through walls.

### 2.2. Offline Phase (Memory Building)
- **Sampling**: The robot is placed at discrete positions with a stride of 50 pixels (x, y) and 8 discrete rotations (45-degree increments) at each point.
- **Encoding**: The 1D sensor strip is flattened into an embedding vector.
- **Storage**: The embeddings are stored in the Modern Hopfield Network. The memory stores the index of the position.
    -   **Index Mapping**: The index corresponds to the flattened loop iteration.
    -   Logic: `spatial_index = index // n_rotations`, `rotation_idx = index % n_rotations`.
    -   Then: `row = spatial_index // n_cols`, `col = spatial_index % n_cols`.

### 2.3. Online Phase (Localization)
- **Movement**: The user can move the robot manually or start an automatic path simulation.
- **Retrieval**:
    1. Capture current observation.
    2. Generate query embedding.
    3. Feed query into MHN.
    4. Retrieve the closest stored pattern(s).
- **Estimation**: The system estimates the robot's position by matching to the closest stored pattern (Top-1 retrieval). Averaging multiple retrieved positions can be added later.

### 2.4. Visualization (UI)
The user interface should display:
- **Map View**: Top-down view showing the map, the robot's actual position (ground truth), and the estimated position.
- **Camera View**: The current visual input seen by the robot.
- **Retrieval Info**: Visualization of the best matching stored view and confidence/energy levels.
- **Controls**: Options to reset, move robot, or toggle visualization modes.

## 3. Technical Architecture

### 3.1. Data Structures
- **Map**: 2D Numpy array or Image.
- **Memory**: Matrix of stored patterns (N_patterns x Embedding_Dimension).

### 3.2. Algorithms
- **Encoder**: A function $f(image) \rightarrow \mathbb{R}^d$.
- **Hopfield Network**: Implementation of continuous Modern Hopfield Network energy function/update rule for pattern retrieval.

### 3.3. Encoding Strategy Options
The following strategies for encoding visual observations into vectors are listed in order of simplicity. **Option 1 is selected for the initial implementation.**

1.  **Option 1: 1D Sensor Flattening (MVP)**
    *   **Method**: The 1D raycast strip is directly used as the vector (optionally downsampled or blurred).
    *   **Pros**: Extremely simple to implement, requires no training.
    *   **Cons**: Sensitive to large geometric changes.

2.  **Option 2: Random Projections**
    *   **Method**: Flatten the image and project it into a lower-dimensional space using a fixed random matrix (Johnson-Lindenstrauss lemma).
    *   **Pros**: Reduces memory footprint while preserving distances.
    *   **Cons**: Lossy compression.

3.  **Option 3: Small CNN Encoder**
    *   **Method**: Use a lightweight Convolutional Neural Network (trained as an autoencoder or pre-trained) to extract semantic features.
    *   **Pros**: Robust to noise, lighting, and shifts.
    *   **Cons**: Adds complexity (training/inference).

### 3.4. User Interface Design
The application window will be arranged in a split-screen layout to visualize both the physical world and the robot's internal state.

1.  **Left Panel: World View (The "God Mode")**
    *   **Map**: Displays the full 2D environment (walls, free space).
    *   **Ground Truth (Blue Marker)**: Shows the robot's actual position and orientation.
    *   **Estimate (Red Marker)**: Shows the position retrieved by the MHN.
    *   **FOV Cone**: A visual cone indicating what the robot is currently looking at.

2.  **Right Panel: Robot Perception**
    *   **Current Input**: A display of the live visual observation (the pixel strip/patch) being fed into the network.
    *   **Retrieved Memory**: The best-matching image retrieved from the MHN memory.
    *   **Similarity Metric**: A visual indicator (bar or text) of the similarity score/energy between the input and the retrieved memory.

3.  **Toolbar (at the top of the window)**
    *   **Edit Map**: Opens a separate Map Editor window to modify the environment (draw walls/obstacles).
    *   **Import Map**: Load a map from a file.
    *   **Export Map**: Save the current map to a file.
    *   **Auto Sample**: Automatically generates samples from the current map and populates the MHN memory.

4.  **Map Editor Window**
    *   A separate interface allowing the user to draw or erase walls on the grid.
    *   Includes tools for pen size and wall type (if applicable).

5.  **Controls**
    *   **Movement**: Arrow keys or WASD to move and rotate the robot.
    *   **Reset**: A key (e.g., 'R') to respawn the robot at a random location.
    *   **Debug**: Toggle visibility of the estimated position or memory visualization.

### 3.5. Technology Stack
- **Language**: Python 3.10+
- **UI Framework**: Pygame (chosen for its efficient real-time rendering loop and easy integration with Numpy for pixel manipulation).
- **Libraries**: Numpy (pure Numpy implementation for MHN).

## 4. Implementation Plan

1.  **Phase 1: Simulation & Data**: Implement map loading, robot movement, and observation generation.
2.  **Phase 2: Model**: Implement the encoder and MHN memory structure.
3.  **Phase 3: Integration**: Connect the simulation loop with the MHN retrieval.
4.  **Phase 4: UI & Polish**: Build the interactive visualization.
