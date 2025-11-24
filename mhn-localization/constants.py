
# Screen dimensions
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 600
MAP_WIDTH = 800
MAP_HEIGHT = 450

# Colors (Tkinter uses Hex strings)
COLOR_BLACK = "#000000"
COLOR_WHITE = "#FFFFFF"
COLOR_BG = "#1E1E1E"
COLOR_PANEL_BG = "#323232"
COLOR_TEXT = "#C8C8C8"
COLOR_BUTTON = "#464646"
COLOR_BUTTON_HOVER = "#5A5A5A"
COLOR_ROBOT_GT = "#0000FF"  # Blue
COLOR_ROBOT_EST = "#FF0000"  # Red
COLOR_SAMPLE_DOT = "#FF0000"  # Red dots for sample positions

# UI Layout
TOOLBAR_HEIGHT = 40
PANEL_PADDING = 10

# Sampling parameters
SAMPLE_STRIDE = 50  # Pixels between sample positions
# Number of rotations at each position (45-degree increments)
SAMPLE_ROTATIONS = 16
DEFAULT_NUM_ANGLES = 16  # Default number of angles (4, 8, 16, or 32)
SAMPLE_DOT_RADIUS = 3  # Size of the sample position dots
# Update display every N samples during auto-sampling (set to None to disable updates)
SAMPLE_UPDATE_FREQUENCY = 100

# Camera preprocessing
# Gaussian blur radius for camera observations (0 = no blur, 1.0-2.0 recommended)
CAMERA_BLUR_RADIUS = 4.0

# Camera FOV (field of view in degrees)
CAMERA_FOV = 120  # Default cone angle

# Number of rays for camera capturing (each ray captures RGB = 3 values)
# This controls the embedding dimension: num_rays * 3
CAMERA_NUM_RAYS = 100  # Default number of camera rays

# Embedding encoding method
# True: Interleaved RGB per pixel [R0, G0, B0, R1, G1, B1, ...]
# False: Channel-separated [R0, R1, ..., G0, G1, ..., B0, B1, ...]
INTERLEAVED_RGB = False

# Modern Hopfield Network parameters
# Beta (inverse temperature) controls sharpness of retrieval
# Higher beta = sharper, more confident retrievals
# Lower beta = softer, more distributed retrievals
DEFAULT_BETA = 50.0

# Top-k matches for weighted interpolation
# Number of best matches to combine for final position prediction
DEFAULT_TOP_K = 1

# Noise settings
# Number of random objects to place on the map
DEFAULT_NOISE_AMOUNT = 10

# Heatmap rendering parameters
# Resolution scale for heatmap computation (0.25 = 1/4 resolution)
# Lower values = faster computation but less detail
# The heatmap is smoothly upscaled to full resolution for display
HEATMAP_RESOLUTION_SCALE = 0.25

# Help text
HELP_TEXT = """ROBOT LOCALIZATION VIA MODERN HOPFIELD NETWORKS

PURPOSE:
This program demonstrates robot localization using a Modern Hopfield Network.
The robot moves on a 2D map and uses visual observations to determine its position.
The network stores patterns from known positions and retrieves the closest match
to localize the robot in real-time.

GETTING STARTED:
1. Create or import a map using the toolbar buttons
2. Click "Sample & Train" to build the localization database
3. Use keyboard controls to move the robot around
4. Watch the network predict the robot's position (green dot)

KEYBOARD CONTROLS:
• W/A/S/D - Move robot up/left/down/right
• J/L - Rotate robot left/right

TOOLBAR BUTTONS:
• Edit Map - Open map editor to draw custom maps
• Import Map - Load a map from a PNG file
• Export Map - Save the current map to a PNG file
• Sample & Train - Generate observation samples and train the network
• Help - Show this help dialog

SETTINGS:
• Blur Radius - Amount of blur applied to camera observations
• Field of View - Viewing angle of the robot's camera (30-360°)
• Number of Camera Rays - Number of rays captured (controls embedding dimension: rays × 3)
• Visibility Index - How far the robot can see (0.01-1.0)
• Beta (Inverse Temp) - Sharpness of retrieval (higher = more selective)
• Combine top k matches - Number of closest matches to average for prediction
• Number of Angles per Location - Number of orientations sampled at each training position
• Noise Settings - Add random circular obstacles to the map
• Interleaved RGB encoding - Alternative pattern encoding method

CONFIDENCE STATISTICS:
• Show confidence computation positions - Display test positions (blue dots)
• Show confidence heatmap - Overlay confidence values across the map
• Average heatmap across all angles - Build the heatmap by averaging confidence over all orientations

VISUAL INDICATORS:
• Red dots - Sample positions where patterns were captured
• Blue robot - Ground truth position (your actual position)
• Green dot - Estimated position (network's prediction)
• Purple line - Estimated orientation direction
• Blue viewing cone - Robot's current field of view
• Black lines - Connections between top-k matches and final prediction

DISPLAY PANELS:
• Current Input - What the robot currently sees
• Retrieved Memory - The closest matching stored pattern
• Similarity Metric - Confidence score of the match (0-100%)
"""
