# main.py
# Entry point for the StarDance orbital physics simulator.
#
# This script initializes the simulation environment, handles user input,
# updates the physics simulation, and renders the visual output.
# It integrates Pygame for rendering, Pymunk for physics, and optional
# PyTorch for AI-based orbit prediction.

import math
import random
import pygame
import pymunk
import warnings
import traceback
import numpy as np
from pathlib import Path

# Optional PyTorch imports – if torch is not installed, AI features are disabled.
try:
    import torch
    from torch import optim
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    optim = None
    TORCH_AVAILABLE = False

# Import optional AI predictor only if torch is available.
if TORCH_AVAILABLE:
    from orbit_predictor import OrbitPredictor, prepare_training_data, train_planet_predictor, predict_planet_position
else:
    # Placeholders so the rest of the code can reference the names safely.
    class OrbitPass:  # dummy class
        pass
    def prepare_training_data(*args, **kwargs):
        return None, None
    , *args, **kwargs):
        return 0.0
    def predict_planet_position(*args, **kwargs):
        return None

from config import *
from entities import *
from physics import *
from rendering import *

# Initialize Pygame and suppress known warnings.
pygame.init()
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=UserWarning, module='pygame')

# Create the display window using dimensions from config.
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Advanced Orbital Physics Simulator")
clock = pygame.time.Clock()
# Target FPS can be overridden by a global TARGET_FPS; otherwise use configured value.
target_fps = TARGET_FPS if 'TARGET_FPS' in globals() else 120
print(f"Display initialized: {WIDTH}x{HEIGHT}")

ERROR_LOG_PATH = Path("runtime_errors.log")

# --- Physics setup ---
# Create a Pymunk space; we will apply custom gravity forces instead of using space.gravity.
space = pymunk.Space()
space.gravity = (0, 0)  # Nullify built‑in gravity; we apply our own central force.

# Create a static central body (e.g., a star) at the world origin.
center_body = pymunk.Body(body_type=pymunk.Body.STATIC)
center_body.position = (0, 0)
space.add(center_body)

# --- AI prediction structures (only used if torch is available) ---
# Dictionaries that map a planet's Pymunk body ID to its predictor model and optimizer.
predictors = {}          # planet_id -> OrbitPredictor model
optimizers = {}          # planet_id -> Adam optimizer
trained_predictors = set()  # Set of body IDs for which training is complete.

# Black‑hole mode toggle state.
black_hole_mode = False
_stored_center_mass = 0      # Backup of CENTER_MASS when entering black‑hole mode.
_stored_g_constant = 0       # Backup of G_CONSTANT when entering black‑hole mode.

# --- Entity containers ---
planets = []   # List of planet data dictionaries.
photons = []   # List of photon data dictionaries.

# Optionally spawn a predefined set of planets at startup.
if START_WITH_SAMPLE_PLANETS:
    for i, distance in enumerate(SAMPLE_PLANET_DISTANCES):
        angle = (i * 2 * math.pi) / len(SAMPLE_PLANET_DISTANCES)
        planet = create_planet(
            space,
            distance,
            angle,
            color_index=i,
            center_position=center_body.position,
            g_constant=G_CONSTANT,
            center_mass=CENTER_MASS
        )
        planets.append(planet)

# Optionally spawn a predefined set of photons at startup.
if START_WITH_SAMPLE_PHOTONS and PHOTON_COUNT > 0:
    for i in range(PHOTON_COUNT):
        angle = (i * 2 * math.pi) / PHOTON_COUNT
        distance = random.randint(80, 120)
        photon = create_photon(
            space,
            distance,
            angle,
            center_position=center_body.position,
            g_constant=G_CONSTANT,
            center_mass=CENTER_MASS
        )
        photons.append(photon)

print(f"Starting empty scene with {len(planets)} planets and {len(photons)} photons", flush=True)

# --- User interface fonts ---
font = pygame.font.Font(None, FONT_SIZE_LARGE)
small_font = pygame.font.Font(None, FONT_SIZE_MEDIUM)
title_font = pygame.font.Font(None, 36)  # Reserved for the title display.

# --- Camera state ---
camera_pos = INITIAL_CAMERA_POS.copy()  # [x, y, z] position of the camera.
camera_rot = INITIAL_CAMERA_ROT.copy()  # [pitch, yaw, roll] in radians.
move_speed = MOVE_SPEED
rot_speed = ROT_SPEED
zoom_level = INITIAL_ZOOM
min_zoom = MIN_ZOOM
max_zoom = MAX_ZOOM

# --- UI interaction state ---
show_ui = True                 # Whether the UI overlay is currently visible.
ui_hidden_timer = 0            # Timer used to auto‑hide the UI after inactivity.
AUTO_HIDE_DELAY = 300          # Frames (5 seconds at 60 FPS) before auto‑hiding UI.
selected_planet = None         # Currently selected planet (for mass/radius edits).
selected_photon = None         # Currently selected photon (for deletion).
selected_entity = None         # Currently selected entity (planet or photon) for camera follow.
status_message = "Click empty space to add a planet. Right-click to add a photon."
status_timer = target_fps * 5  # How long to show a status message (in frames).
last_error = None              # Stores the last exception for on‑screen display.
error_timer = 0                # Timer for clearing the last error display.

# Sidebar constants for entity list UI.
UI_SIDEBAR_WIDTH = 200
SIDEBAR_PADDING = 10
ENTRY_HEIGHT = 20
HEADER_HEIGHT = 24

# ----- Helper functions -----
def set_status(message, frames=None):
    """Display a transient status message in the UI."""
    global status_message, status_timer
    status_message = message
    status_timer = frames if frames is not None else target_fps * 4


def log_runtime_error(context, exc):
    """Record an exception to the error log and pause the simulation.

    The simulator window remains open so the user can see the error.
    """
    global last_error, error_timer, paused
    last_error = f"{context}: {exc.__class__.__name__}: {exc}"
    error_timer = target_fps * 8
    traced = traceback.format_exc()
    print(last_error, flush=True)
    print(traced, flush=True)
    with ERROR_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{context}] {exc.__class__.__name__}: {exc}\n")
        log_file.write(traced)


def remove_predictor_for_body(body):
    """Remove AI predictor and optimizer associated with a planet's body."""
    planet_id = id(body)
    predictors.pop(planet_id, None)
    optimizers.pop(planet_id, None)
    trained_predictors.discard(planet_id)


def safe_space_remove(entity):
    """Safely remove a Pymunk body and shape from the space."""
    try:
        space.remove(entity['body'], entity['shape'])
    except Exception as exc:
        log_runtime_error("remove entity", exc)


def remove_planet(planet):
    """Remove a planet from the simulation and clean up its AI resources."""
    global selected_planet, selected_photon, selected_entity
    if planet in planets:
        remove_predictor_for_body(planet['body'])
        safe_space_remove(planet)
        planets.remove(planet)
        if selected_planet is planet:
            selected_planet = None
            selected_entity = None
        set_status("Planet removed.")


def remove_photon(photon):
    """Remove a photon from the simulation."""
    global selected_photon, selected_entity
    if photon in photons:
        safe_space_remove(photon)
        photons.remove(photon)
        if selected_photon is photon:
            selected_photon = None
            selected_entity = None
        set_status("Photon removed.")


def clear_scene():
    """Remove all planets and photons, and reset AI tracking structures."""
    global selected_planet, selected_photon, selected_entity, predictors, optimizers, trained_predictors
    for planet in list(planets):
        remove_planet(planet)
    for photon in list(photons):
        remove_photon(photon)
    selected_planet = None
    selected_photon = None
    selected_entity = None
    predictors.clear()
    optimizers.clear()
    trained_predictors.clear()
    set_status("Scene cleared.")


def reset_view():
    """Reset camera position and rotation to their initial values."""
    global camera_pos, camera_rot
    camera_pos[:] = INITIAL_CAMERA_POS.copy()
    camera_rot[:] = INITIAL_CAMERA_ROT.copy()
    set_status("Camera reset.")


def add_planet_at_world(world_pos):
    """Create a planet at the supplied world coordinates where the user clicked.

    Prevents creation too close to the central body.
    Returns the created planet dictionary or None if too close.
    """
    dx = world_pos[0] - center_body.position.x
    dy = world_pos[1] - center_body.position.y
    distance = math.sqrt(dx * dx + dy * dy)
    if distance <= 50:
        set_status("Too close to the center. Click farther out to add a planet.")
        return None

    angle = math.atan2(dy, dx)
    mass = random.uniform(5, 25)
    planet = create_planet(
        space,
        distance,
        angle,
        mass,
        color_index=len(planets),
        center_position=center_body.position,
        g_constant=G_CONSTANT,
        center_mass=CENTER_MASS
    )
    planets.append(planet)
    set_status(f"Added {planet['name']} at radius {int(distance)}.")
    return planet


def add_random_photon():
    """Spawn a photon at a random distance and angle."""
    angle = random.uniform(0, 2 * math.pi)
    distance = random.randint(100, 200)
    photon = create_photon(
        space,
        distance,
        angle,
        center_position=center_body.position,
        g_constant=G_CONSTANT,
        center_mass=CENTER_MASS
    )
    photons.append(photon)
    set_status("Added photon.")
    return photon


def add_photon_at_world(world_pos):
    """Spawn a photon at the world position corresponding to the mouse cursor."""
    dx = world_pos[0] - center_body.position.x
    dy = world_pos[1] - center_body.position.y
    distance = max(math.sqrt(dx * dx + dy * dy), 50)
    angle = math.atan2(dy, dx)
    photon = create_photon(
        space,
        distance,
        angle,
        center_position=center_body.position,
        g_constant=G_CONSTANT,
        center_mass=CENTER_MASS
    )
    photons.append(photon)
    set_status("Added photon.")
    return photon


# ----- Main execution loop -----
try:
    print("Starting main loop", flush=True)
    running = True
    paused = False
    frame_count = 0
    ai_train_counter = 0  # Counter to throttle AI training frequency.

    while running:
        # Clear the screen with a dark space‑like background.
        screen.fill((5, 5, 15))  # Very dark space background

        # Build list of entities for UI sidebar and click detection.
        entities_for_ui = []
        for p in planets:
            entities_for_ui.append((p, p['name']))
        for i, ph in enumerate(photons):
            entities_for_ui.append((ph, f"Photon {i}"))

        # Event handling.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    set_status("Paused." if paused else "Resumed.")
                elif event.key == pygame.K_h:
                    # Toggle UI visibility.
                    show_ui = not show_ui
                    ui_hidden_timer = 0
                    set_status("UI shown." if show_ui else "UI hidden.")
                elif event.key == pygame.K_BACKSPACE:
                    # Remove the most recently added planet, if any.
                    if len(planets) > 0:
                        remove_planet(planets[-1])
                elif event.key == pygame.K_g:
                    # Queue AI training (model update) for better performance.
                    if TORCH_AVAILABLE:
                        ai_train_counter = 0
                        set_status("AI training queued.")
                    else:
                        set_status("AI not available (torch not installed).")
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    # Increase the gravitational constant.
                    G_CONSTANT *= 1.1
                    set_status(f"Gravity increased to {int(G_CONSTANT)}.")
                elif event.key == pygame.K_MINUS:
                    # Decrease the gravitational constant.
                    G_CONSTANT /= 1.1
                    set_status(f"Gravity decreased to {int(G_CONSTANT)}.")
                elif event.key == pygame.K_p:
                    add_random_photon()
                elif event.key == pygame.K_b:
                    # Toggle black‑hole mode.
                    if not black_hole_mode:
                        # Enter black‑hole mode: store current values and apply extreme gravity.
                        _stored_center_mass = CENTER_MASS
                        _stored_g_constant = G_CONSTANT
                        CENTER_MASS = BLACK_HOLE_THRESHOLD * 2  # Well above threshold.
                        G_CONSTANT = 2000000  # Amplify gravity for visual effect.
                        black_hole_mode = True
                        set_status("Black hole mode enabled.")
                    else:
                        # Exit black‑hole mode: restore the original gravity and mass.
                        CENTER_MASS = _stored_center_mass
                        G_CONSTANT = _stored_g_constant
                        black_hole_mode = False
                        set_status("Black hole mode disabled.")
                elif event.key == pygame.K_c:
                    clear_scene()
                elif event.key == pygame.K_HOME:
                    reset_view()
                elif event.key == pygame.K_DELETE:
                    # Prefer deleting a selected photon; otherwise delete selected planet.
                    if selected_photon is not None:
                        remove_photon(selected_photon)
                    elif selected_planet is not None:
                        remove_planet(selected_planet)
                elif event.key == pygame.K_m:
                    # Increase mass of the selected planet.
                    if selected_planet is not None:
                        update_planet_mass(selected_planet, 1.2)
                        set_status(f"{selected_planet['name']} mass: {selected_planet['mass']:.1f}")
                elif event.key == pygame.K_n:
                    # Decrease mass of the selected planet.
                    if selected_planet is not None and selected_planet['mass'] > 1:
                        update_planet_mass(selected_planet, 1.2)
                        set_status(f"{selected_planet['name']} mass: {selected_planet['mass']:.1f}")
                elif event.key == pygame.K_r:
                    # Increase visual radius of the selected planet (visual only).
                    if selected_planet is not None:
                        update_planet_radius(selected_planet, 1.2)
                        set_status(f"{selected_planet['name']} visual size increased.")
                elif event.key == pygame.K_f:
                    # Decrease visual radius of the selected planet (visual only).
                    if selected_planet is not None:
                        update_planet_radius(selected_planet, 1/1.2)
                        set_status(f"{selected_planet['name']} visual size decreased.")
                elif event.key == pygame.K_i:
                    # Zoom in.
                    zoom_level = min(max_zoom, zoom_level * 1.1)
                    set_status(f"Zoom: {zoom_level:.2f}x")
                elif event.key == pygame.K_o:
                    # Zoom out.
                    zoom_level = max(min_zoom, zoom_level / 1.1)
                    set_status(f"Zoom: {zoom_level:.2f}x")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Mouse wheel up → zoom in.
                    zoom_level = min(max_zoom, zoom_level * 1.1)
                elif event.button == 5:  # Mouse wheel down → zoom out.
                    zoom_level = max(min_zoom, zoom_level / 1.1)
                elif event.button == 1:  # Left click: add planet or select existing.
                    # Check if click is in the entity sidebar.
                    if event.pos[0] > WIDTH - UI_SIDEBAR_WIDTH:
                        # Click inside sidebar – select entity from list.
                        y = event.pos[1]
                        index = (y - SIDEBAR_PADDING - HEADER_HEIGHT) // ENTRY_HEIGHT
                        if 0 <= index < len(entities_for_ui):
                            obj, label = entities_for_ui[index]
                            # Set selection based on object type.
                            if obj in planets:
                                selected_planet = obj
                                selected_photon = None
                            else:
                                selected_photon = obj
                                selected_planet = None
                            selected_entity = obj
                            set_status(f"Selected: {label}")
                        # If click outside entries, ignore.
                    else:
                        # Click outside sidebar – use existing selection logic.
                        world_pos = screen_to_world(
                            (event.pos[0], event.pos[1]), camera_pos, camera_rot, zoom_level, WIDTH, HEIGHT
                        )  # Assume z = 0 for clicking.

                        # Check if click hit a photon first (higher priority for selection).
                        clicked_photon = select_entity_at_position(
                            photons, event.pos, camera_pos, camera_rot, zoom_level, WIDTH, HEIGHT, CLICK_DISTANCE_THRESHOLD
                        )
                        if clicked_photon:
                            selected_photon = clicked_photon
                            selected_planet = None
                            selected_entity = clicked_photon
                            set_status("Photon selected. Press Delete to remove it.")
                        else:
                            # Otherwise check for a planet.
                            clicked_planet = select_entity_at_position(
                                planets, event.pos, camera_pos, camera_rot, zoom_level, WIDTH, HEIGHT, CLICK_DISTANCE_THRESHOLD
                            )
                            if clicked_planet:
                                selected_planet = clicked_planet
                                selected_photon = None
                                selected_entity = clicked_planet
                                set_status(f"{selected_planet['name']} selected.")
                            else:
                                # No entity clicked → create a new planet at the cursor location.
                                add_planet_at_world(world_pos)
                elif event.button == 3:  # Right click: add photon at cursor.
                    world_pos = screen_to_world(
                        (event.pos[0], event.pos[1]), camera_pos, camera_rot, zoom_level, WIDTH, HEIGHT
                    )
                    add_photon_at_world(world_pos)
            elif event.type == pygame.MOUSEMOTION:
                if event.buttons[0]:  # Left mouse button dragged → rotate camera.
                    # Convert mouse movement to yaw/pitch adjustments.
                    camera_rot[1] -= event.rel[0] * rot_speed  # Yaw (left/right).
                    camera_rot[0] -= event.rel[1] * rot_speed  # Pitch (up/down).
                    # Clamp pitch to avoid gimbal lock.
                    camera_rot[0] = max(-3.14159 / 2 + 0.1,
                                        min(3.14159 / 2 - 0.1, camera_rot[0]))

        # Continuous keyboard handling for camera movement.
        pressed_keys = pygame.key.get_pressed()
        camera_step = move_speed / max(zoom_level, 0.1)
        if pressed_keys[pygame.K_LEFT]:
            camera_rot[1] += rot_speed
        if pressed_keys[pygame.K_RIGHT]:
            camera_rot[1] -= rot_speed
        if pressed_keys[pygame.K_UP]:
            camera_rot[0] += rot_speed
        if pressed_keys[pygame.K_DOWN]:
            camera_rot[0] -= rot_speed
        if pressed_keys[pygame.K_w]:
            camera_pos[1] += camera_step
        if pressed_keys[pygame.K_s]:
            camera_pos[1] -= camera_step
        if pressed_keys[pygame.K_a]:
            camera_pos[0] -= camera_step
        if pressed_keys[pygame.K_d]:
            camera_pos[0] += camera_step
        # Clamp pitch after movement.
        camera_rot[0] = max(-3.14159 / 2 + 0.1,
                            min(3.14159 / 2 - 0.1, camera_rot[0]))

        # Auto‑hide the UI after a period of inactivity.
        if show_ui and ui_hidden_timer > 0:
            ui_hidden_timer -= 1
            if ui_hidden_timer <= 0:
                show_ui = False

        # Temporarily show UI when the user interacts with controls.
        if not show_ui and (
                pygame.mouse.get_pressed()[0] or any(
                    pressed_keys[key] for key in [
                        pygame.K_SPACE,
                        pygame.K_BACKSPACE,
                        pygame.K_UP,
                        pygame.K_DOWN,
                        pygame.K_LEFT,
                        pygame.K_RIGHT,
                        pygame.K_w,
                        pygame.K_a,
                        pygame.K_s,
                        pygame.K_d,
                        pygame.K_EQUALS,
                        pygame.K_MINUS,
                        pygame.K_g,
                        pygame.K_h,
                        pygame.K_m,
                        pygame.K_n,
                        pygame.K_r,
                        pygame.K_f,
                        pygame.K_DELETE])):
            show_ui = True
            ui_hidden_timer = AUTO_HIDE_DELAY

        # --- Physics update (runs only when not paused) ---
        if not paused:
            try:
                if planets or photons:
                    # Compute gravitational forces acting on each body.
                    forces = calculate_forces(
                        planets,
                        photons,
                        center_body,
                        SOFTENING_SQUARED,
                        G_CONSTANT,
                        CENTER_MASS
                    )
                    # Apply the computed forces.
                    apply_forces(planets, photons, forces)
                    # Integrate positions based on current velocities.
                    update_positions(planets, photons)

                    # --- Optional AI‑based velocity adjustment ---
                    # Blend a predicted position with the simulated one to improve stability.
                    if TORCH_AVAILABLE and frame_count % 60 == 0:  # Run AI logic once per second.
                        for planet_data in planets:
                            if len(planet_data['history']) > AI_MIN_HISTORY_FOR_PREDICTION:
                                planet_id = id(planet_data['body'])
                                # Lazily create predictor and optimizer if needed.
                                if planet_id not in predictors:
                                    predictors[planet_id] = OrbitPredictor(input_size=AI_INPUT_SIZE)
                                    optimizers[planet_id] = optim.Adam(
                                        predictors[planet_id].parameters(), lr=AI_LEARNING_RATE
                                    )

                                predicted_pos = None
                                if planet_id in trained_predictors:
                                    predicted_pos = predict_planet_position(
                                        predictors[planet_id],
                                        planet_data['history']
                                    )
                                if predicted_pos is not None:
                                    # Blend actual and predicted positions (70% physics, 30% AI).
                                    blend_factor = 0.3
                                    current_pos = np.array(
                                        [planet_data['body'].position.x, planet_data['body'].position.y]
                                    )
                                    predicted_pos_np = np.array(predicted_pos)
                                    blended_pos = current_pos * (1 - blend_factor) + predicted_pos_np * blend_factor
                                    # Small corrective nudge toward the blended position.
                                    adjustment = (blended_pos - current_pos) * 0.1
                                    planet_data['body'].velocity += (
                                        adjustment[0] / planet_data['body'].mass,
                                        adjustment[1] / planet_data['body'].mass
                                    )

                # Apply velocity caps to prevent numerical explosion.
                limit_velocities(photons, MAX_VELOCITY)
                for planet_data in planets:
                    speed = math.sqrt(
                        planet_data['body'].velocity.x**2 +
                        planet_data['body'].velocity.y**2
                    )
                    if speed > MAX_VELOCITY:
                        scale = MAX_VELOCITY / speed
                        planet_data['body'].velocity = (
                            planet_data['body'].velocity.x * scale,
                            planet_data['body'].velocity.y * scale
                        )

                # --- Black‑hole capture detection ---
                photons_to_remove = check_black_hole_capture(
                    photons,
                    center_body,
                    CENTER_MASS,
                    BLACK_HOLE_THRESHOLD,
                    SPEED_OF_LIGHT,
                    G_CONSTANT
                )
                for photon_data in photons_to_remove:
                    if photon_data in photons:
                        photons.remove(photon_data)
                    space.remove(photon_data['body'], photon_data['shape'])

                planets_to_remove = check_black_hole_capture_planets(
                    planets,
                    center_body,
                    CENTER_MASS,
                    BLACK_HOLE_THRESHOLD,
                    SPEED_OF_LIGHT,
                    G_CONSTANT
                )
                for planet_data in planets_to_remove:
                    remove_planet(planet_data)

                # Step the Pymunk space forward.
                update_physics_space(space, PHYSICS_TIME_STEP)

                # --- Periodic AI model training ---
                if TORCH_AVAILABLE:
                    ai_train_counter += 1
                    if ai_train_counter >= AI_TRAIN_INTERVAL:
                        ai_train_counter = 0
                        for planet_data in planets:
                            planet_id = id(planet_data['body'])
                            if planet_id not in predictors:
                                predictors[planet_id] = OrbitPredictor(input_size=AI_INPUT_SIZE)
                                optimizers[planet_id] = optim.Adam(
                                    predictors[planet_id].parameters(), lr=AI_LEARNING_RATE
                                )
                            if len(planet_data['history']) > AI_MIN_HISTORY_FOR_TRAINING:
                                orbit_predictor.train_planet_predictor(
                                    predictors[planet_id],
                                    optimizers[planet_id],
                                    planet_data['history'],
                                    epochs=AI_EPOCHS_PER_TRAINING
                                )
                                trained_predictors.add(planet_id)
            except Exception as exc:
                log_runtime_error("physics update", exc)

        # --- Camera follow selected entity ---
        if selected_entity is not None:
            target_x = selected_entity['body'].position.x
            target_y = selected_entity['body'].position.y
            # Smooth follow factor (0.1 = 10% of distance per frame).
            camera_pos[0] += (target_x - camera_pos[0]) * 0.1
            camera_pos[1] += (target_y - camera_pos[1]) * 0.1
            # Keep Z coordinate unchanged (could also follow Z if needed).

        # Frame‑counters and status timers.
        frame_count += 1
        if status_timer > 0:
            status_timer -= 1
        if error_timer > 0:
            error_timer -= 1
        elif last_error is not None:
            last_error = None

        # Periodic console output (once per second).
        if frame_count % target_fps == 0:
            fps = clock.get_fps()
            print(
                f"FPS: {fps:.1f}, Planets: {len(planets)}, Photons: {len(photons)}, Paused: {paused}"
            )

        # --- Rendering (always executed, even when paused) ---
        if frame_count == 1:
            print("About to render...", flush=True)
            print(f"Drawing {len(planets)} planets and {len(photons)} photons", flush=True)

        try:
            # Draw orbital trails.
            draw_trails(screen, planets, photons, camera_pos, camera_rot, zoom_level, WIDTH, HEIGHT)
            # Draw bodies (planets, photons, central mass).
            draw_bodies(screen, planets, photons, center_body, camera_pos, camera_rot, zoom_level, WIDTH, HEIGHT, CENTER_MASS)
            # Draw selection ring around the selected entity.
            draw_selection_indicator(
                screen,
                selected_photon or selected_planet,
                camera_pos,
                camera_rot,
                zoom_level,
                WIDTH,
                HEIGHT,
                frame_count
            )
            # Draw UI overlay (status, data, etc.).
            draw_ui(
                screen,
                font,
                small_font,
                title_font,
                planets,
                photons,
                paused,
                show_ui,
                G_CONSTANT,
                CENTER_MASS,
                BLACK_HOLE_THRESHOLD,
                camera_pos,
                camera_rot,
                zoom_level,
                WIDTH,
                HEIGHT,
                clock.get_fps(),
                selected_photon or selected_planet,
                status_message if status_timer > 0 else None,
                last_error
            )
            # --- Draw entity sidebar ---
            # Semi‑transparent background for sidebar.
            sidebar_surf = pygame.Surface((UI_SIDEBAR_WIDTH, HEIGHT), pygame.SRCALPHA)
            sidebar_surf.fill((0, 0, 0, 180))  # RGBA: black with 70% opacity.
            screen.blit(sidebar_surf, (WIDTH - UI_SIDEBAR_WIDTH, 0))
            # Sidebar title.
            title_surf = font.render("Entities", True, (255, 255, 255))
            screen.blit(title_surf, (WIDTH - UI_SIDEBAR_WIDTH + SIDEBAR_PADDING, SIDEBAR_PADDING))
            # List entries.
            for i, (obj, label) in enumerate(entities_for_ui):
                y = SIDEBAR_PADDING + HEADER_HEIGHT + i * ENTRY_HEIGHT
                # Highlight selected entry.
                if obj == selected_entity:
                    highlight_rect = pygame.Rect(
                        WIDTH - UI_SIDEBAR_WIDTH + SIDEBAR_PADDING,
                        y,
                        UI_SIDEBAR_WIDTH - 2 * SIDEBAR_PADDING,
                        ENTRY_HEIGHT - 2
                    )
                    pygame.draw.rect(screen, (70, 70, 180), highlight_rect)
                # Render text.
                text_surf = small_font.render(label, True, (255, 255, 255))
                screen.blit(text_surf, (WIDTH - UI_SIDEBAR_WIDTH + SIDEBAR_PADDING + 4, y + 2))
            # Present the frame.
            pygame.display.flip()
        except Exception as exc:
            log_runtime_error("render", exc)

        # Limit the loop to the target frame rate.
        clock.tick(target_fps)

except Exception as e:
    log_runtime_error("fatal main loop", e)

finally:
    pygame.quit()