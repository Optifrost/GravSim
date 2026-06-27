# config.py
# Central configuration module for the StarDance orbital physics simulator.
#
# This file contains all tunable constants used throughout the simulation,
# including display dimensions, physics parameters, color palettes, camera
# settings, UI parameters, and AI training hyperparameters. Adjusting these
# values allows you to change the visual appearance, simulation behavior, and
# performance characteristics without modifying the core code.

import pygame

# ----------------------------------------------------------------------
# Display and window settings
# ----------------------------------------------------------------------
WIDTH, HEIGHT = 1400, 900            # Window resolution in pixels.
TARGET_FPS = 120                     # Desired frames‑per‑second for the main loop.

# ----------------------------------------------------------------------
# Physical constants (tuned for stable, visually pleasing orbits)
# ----------------------------------------------------------------------
G_CONSTANT = 100                     # Gravitational constant (scaled for visual clarity).
CENTER_MASS = 100000                 # Mass of the central static body (star/black hole).
PLANET_MASS = 10                     # Default mass for newly created planets.
PHOTON_MASS = 0.001                  # Negligible mass for photons (affected by gravity but
                                     # does not noticeably influence other bodies).
SPEED_OF_LIGHT = 100                 # Simulated speed of light (used for photon speed and
                                     # black‑hole capture checks).
TRAIL_LENGTH = 100                   # Number of past positions stored for drawing motion trails.
PREDICTION_FRAMES = 10               # How many steps ahead the AI attempts to predict.
BLACK_HOLE_THRESHOLD = 10000000      # Mass threshold above which the central body acts as a black hole.
PHOTON_COUNT = 0                     # Number of photons to spawn at startup (0 = none).
SOFTENING = 25.0                     # Softening length to prevent divergent forces at very small distances.
SOFTENING_SQUARED = SOFTENING * SOFTENING  # Pre‑computed softening term for efficiency.
MAX_VELOCITY = 500.0                 # Speed cap applied to bodies to avoid numerical instability.

# ----------------------------------------------------------------------
# Color definitions (RGB tuples, values 0‑255)
# ----------------------------------------------------------------------
SUN_COLOR = (255, 220, 100)          # Warm yellow for the central body.
PLANET_COLORS = [                    # Distinct palette for orbiting bodies.
    (255, 80, 80),    # Red
    (80, 255, 80),    # Green
    (80, 80, 255),    # Blue
    (255, 255, 80),   # Yellow
    (255, 80, 255),   # Magenta
    (80, 255, 255),   # Cyan
    (255, 180, 80),   # Orange
    (180, 80, 255),   # Purple
]
PHOTON_COLOR = (255, 255, 200)       # Pale yellow for photons.
GRID_COLOR = (40, 40, 60)            # Dark gray‑blue for the background grid.
GRID_SPACING = 50                    # World‑unit distance between grid lines.
AXIS_COLOR = (60, 60, 80)            # Color for the X and Y axes lines.
Z_AXIS_COLOR_POS = (30, 30, 50)      # Color for positive Z‑axis indicator lines.
Z_AXIS_COLOR_NEG = (50, 30, 30)      # Color for negative Z‑axis indicator lines.

# ----------------------------------------------------------------------
# Camera navigation parameters
# ----------------------------------------------------------------------
INITIAL_CAMERA_POS = [0.0, 0.0, -500.0]  # Starting [x, y, z] position of the camera.
INITIAL_CAMERA_ROT = [0.5, 0.5, 0.0]    # Starting [pitch, yaw, roll] in radians.
MOVE_SPEED = 15.0                       # Speed at which the camera translates (world units/sec).
ROT_SPEED = 0.08                        # Angular speed for camera rotation (radians/pixel).
MIN_ZOOM = 0.1                          # Minimum allowed zoom factor (farther out).
MAX_ZOOM = 20.0                         # Maximum allowed zoom factor (closer in).
INITIAL_ZOOM = 1.0                      # Default zoom level when the simulation starts.

# ----------------------------------------------------------------------
# User interface styling
# ----------------------------------------------------------------------
FONT_SIZE_LARGE = 28                    # Font size for primary UI text.
FONT_SIZE_MEDIUM = 22                   # Font size for secondary labels.
FONT_SIZE_SMALL = 18                    # Font size for detailed/tooltip text.
UI_TEXT_COLOR = (200, 200, 255)         # Main text color (light blue‑white).
UI_BACKGROUND_COLOR = (0, 0, 0, 128)    # Semi‑transparent black backdrop for UI panels.
UI_ACCENT_COLOR = (120, 210, 255)       # Accent color for highlights and active elements.
UI_WARNING_COLOR = (255, 170, 90)       # Color used for warning messages.
UI_ERROR_COLOR = (255, 110, 110)        # Color used for error messages.
AUTO_HIDE_DELAY = 300                   # Frames (≈5 s at 60 FPS) before auto‑hiding the UI.
CLICK_DISTANCE_THRESHOLD = 25          # Pixel radius within which a click selects an entity.
SELECTION_RADIUS = 8                    # Radius of the selection circle drawn around entities.
BLINK_RATE = 15                         # Frame count for the selection indicator blink cycle.

# ----------------------------------------------------------------------
# Simulation startup options
# ----------------------------------------------------------------------
START_WITH_SAMPLE_PLANETS = False       # If True, spawn a predefined set of planets on launch.
START_WITH_SAMPLE_PHOTONS = False       # If True, spawn a predefined set of photons on launch.
SAMPLE_PLANET_DISTANCES = [150, 250, 350, 450, 550]  # Radii (world units) for starter planets.

# ----------------------------------------------------------------------
# AI‑assisted orbit prediction configuration
# ----------------------------------------------------------------------
AI_TRAIN_INTERVAL = 180                 # Frame count between training updates (≈3 s).
AI_MIN_HISTORY_FOR_TRAINING = 20        # Minimum history entries required to train a model.
AI_MIN_HISTORY_FOR_PREDICTION = 10      # Minimum history entries required to make a prediction.
AI_EPOCHS_PER_TRAINING = 2              # Number of training epochs per update.
AI_INPUT_SIZE = 40                      # Dimensionality of the input vector to the predictor.
AI_HIDDEN_SIZE = 64                     # Number of neurons in the hidden layer.
AI_OUTPUT_SIZE = 2                      # Output dimensions (predicted X, Y position).
AI_LEARNING_RATE = 0.001                # Learning rate for the Adam optimizer.

# ----------------------------------------------------------------------
# Performance and simulation integrity settings
# ----------------------------------------------------------------------
PHYSICS_TIME_STEP = 1 / 60.0            # Fixed timestep used by the Pymunk physics engine.
TRAIL_SURFACE_CACHE_ENABLED = True      # Enable surface caching for trails to improve FPS.