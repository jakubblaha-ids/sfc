
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
SAMPLE_DOT_RADIUS = 3  # Size of the sample position dots
# Update display every N samples during auto-sampling (set to None to disable updates)
SAMPLE_UPDATE_FREQUENCY = 100

# Camera preprocessing
# Gaussian blur radius for camera observations (0 = no blur, 1.0-2.0 recommended)
CAMERA_BLUR_RADIUS = 4.0

# Camera FOV (field of view in degrees)
CAMERA_FOV = 120  # Default cone angle

# Embedding encoding method
# True: Interleaved RGB per pixel [R0, G0, B0, R1, G1, B1, ...]
# False: Channel-separated [R0, R1, ..., G0, G1, ..., B0, B1, ...]
INTERLEAVED_RGB = True

# Modern Hopfield Network parameters
# Beta (inverse temperature) controls sharpness of retrieval
# Higher beta = sharper, more confident retrievals
# Lower beta = softer, more distributed retrievals
DEFAULT_BETA = 50.0

# Heatmap rendering parameters
# Resolution scale for heatmap computation (0.25 = 1/4 resolution)
# Lower values = faster computation but less detail
# The heatmap is smoothly upscaled to full resolution for display
HEATMAP_RESOLUTION_SCALE = 0.25
