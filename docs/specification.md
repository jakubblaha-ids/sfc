# Specification: Robot Localization via Modern Hopfield Networks

## 1. Overview
This project implements a robot localization demo using Modern Hopfield Networks (MHN). The system simulates a robot moving in a 2D environment, capturing visual observations, and using an MHN-based associative memory to determine its location by matching current observations against a stored database of views.

## 2. Functional Requirements

### 2.1. Environment & Simulation
- **Map**: A 2D top-down map (grid or image-based) representing the environment (e.g., a floor plan or maze).
- **Robot**: An entity capable of movement (x, y) and rotation (angle) within the map.
- **Sensor**: A simulated camera that captures a local observation (e.g., a 1D strip of pixels or a small 2D patch) based on the robot's current position and orientation.

### 2.2. Offline Phase (Memory Building)
- **Sampling**: The system must generate a dataset of observations by placing the robot at various discrete positions and orientations on the map.
- **Encoding**: Each observation is processed (e.g., via a simple CNN or direct pixel flattening) into an embedding vector.
- **Storage**: These embeddings, along with their corresponding ground-truth coordinates (x, y, angle), are stored in the Modern Hopfield Network.

### 2.3. Online Phase (Localization)
- **Movement**: The user can move the robot manually or start an automatic path simulation.
- **Retrieval**:
    1. Capture current observation.
    2. Generate query embedding.
    3. Feed query into MHN.
    4. Retrieve the closest stored pattern(s).
- **Estimation**: The system estimates the robot's position based on the retrieved memory index.

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

## 4. Implementation Plan

1.  **Phase 1: Simulation & Data**: Implement map loading, robot movement, and observation generation.
2.  **Phase 2: Model**: Implement the encoder and MHN memory structure.
3.  **Phase 3: Integration**: Connect the simulation loop with the MHN retrieval.
4.  **Phase 4: UI & Polish**: Build the interactive visualization.
