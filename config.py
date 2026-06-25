# config.py
# Configuration constants for the StarDance orbital physics simulator

import pygame

# Screen settings
WIDTH, HEIGHT = 1400, 900  # Increased resolution
TARGET_FPS = 120  # Target FPS as requested by user

# Physics constants
G_CONSTANT = 200000  # Gravitational constant (adjusted for nice visuals)
CENTER_MASS = 1000000  # Mass of central body
PLANET_MASS = 10  # Base mass of each planet
PHOTON_MASS = 0.001  # Very small mass for photons (affected by gravity but minimal effect)
SPEED_OF_LIGHT = 100  # Simulated speed of light (adjusted for visualization)
TRAIL_LENGTH = 100  # Number of points to store for trail
PREDICTION_FRAMES = 10  # How many frames ahead to predict
BLACK_HOLE_THRESHOLD = 10000000  # Threshold for black hole formation
PHOTON_COUNT = 0  # No photons are spawned on startup
SOFTENING = 25.0  # Softening factor to prevent numerical instability at close distances
SOFTENING_SQUARED = SOFTENING * SOFTENING
MAX_VELOCITY = 500.0  # Maximum velocity to prevent numerical explosion

# Colors
SUN_COLOR = (255, 220, 100)  # More realistic sun color
PLANET_COLORS = [
    (255, 80, 80),    # Red
    (80, 255, 80),    # Green
    (80, 80, 255),    # Blue
    (255, 255, 80),   # Yellow
    (255, 80, 255),   # Magenta
    (80, 255, 255),   # Cyan
    (255, 180, 80),   # Orange
    (180, 80, 255),   # Purple
]
PHOTON_COLOR = (255, 255, 200)  # Yellow-white for photons
GRID_COLOR = (40, 40, 60)  # Dark blue-gray
GRID_SPACING = 50  # Distance between grid lines in world units
AXIS_COLOR = (60, 60, 80)  # Color for center axes
Z_AXIS_COLOR_POS = (30, 30, 50)  # Color for positive Z-axis lines
Z_AXIS_COLOR_NEG = (50, 30, 30)  # Color for negative Z-axis lines

# Camera settings
INITIAL_CAMERA_POS = [0.0, 0.0, -500.0]  # [x, y, z] position of camera
INITIAL_CAMERA_ROT = [0.5, 0.5, 0.0]  # [pitch, yaw, roll] in radians
MOVE_SPEED = 15.0  # Increased movement speed
ROT_SPEED = 0.08   # Increased rotation speed
MIN_ZOOM = 0.1
MAX_ZOOM = 20.0    # Increased max zoom for better viewing
INITIAL_ZOOM = 1.0

# UI settings
FONT_SIZE_LARGE = 28
FONT_SIZE_MEDIUM = 22
FONT_SIZE_SMALL = 18
UI_TEXT_COLOR = (200, 200, 255)
UI_BACKGROUND_COLOR = (0, 0, 0, 128)  # Semi-transparent black
UI_ACCENT_COLOR = (120, 210, 255)
UI_WARNING_COLOR = (255, 170, 90)
UI_ERROR_COLOR = (255, 110, 110)
AUTO_HIDE_DELAY = 300  # frames (5 seconds at 60fps)
CLICK_DISTANCE_THRESHOLD = 25  # pixels for selection
SELECTION_RADIUS = 8  # Slightly larger than photon radius for visibility
BLINK_RATE = 15  # Frames for selection indicator blink

# Startup behavior
START_WITH_SAMPLE_PLANETS = False
START_WITH_SAMPLE_PHOTONS = False
SAMPLE_PLANET_DISTANCES = [150, 250, 350, 450, 550]

# AI settings
AI_TRAIN_INTERVAL = 180  # Train every 3 seconds instead of every frame
AI_MIN_HISTORY_FOR_TRAINING = 20  # Minimum history points needed for training
AI_MIN_HISTORY_FOR_PREDICTION = 10  # Minimum history points needed for prediction
AI_EPOCHS_PER_TRAINING = 2  # Reduced epochs for better performance
AI_INPUT_SIZE = 40  # Input size for neural network
AI_HIDDEN_SIZE = 64  # Hidden layer size for neural network
AI_OUTPUT_SIZE = 2  # Output size (x, y position) for neural network
AI_LEARNING_RATE = 0.001  # Learning rate for Adam optimizer

# Performance settings
PHYSICS_TIME_STEP = 1 / 60.0  # Physics simulation time step
TRAIL_SURFACE_CACHE_ENABLED = True  # Performance optimization for trails
