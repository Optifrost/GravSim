# main.py
# Main entry point for the StarDance orbital physics simulator

import torch.optim as optim
import math
import random
import pygame
import pymunk
import warnings
import traceback
import numpy as np
from pathlib import Path
from config import *
from entities import *
from physics import *
from rendering import *
import orbit_predictor

# Initialize pygame
pygame.init()
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=UserWarning, module='pygame')

# Create display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Advanced Orbital Physics Simulator")
clock = pygame.time.Clock()
target_fps = TARGET_FPS if 'TARGET_FPS' in globals() else 120  # Target FPS as requested by user
print(f"Display initialized: {WIDTH}x{HEIGHT}")
ERROR_LOG_PATH = Path("runtime_errors.log")

# Create physics space
space = pymunk.Space()
space.gravity = (0, 0)  # We'll apply our own central forces

# Create central body (static) at world origin
center_body = pymunk.Body(body_type=pymunk.Body.STATIC)
center_body.position = (0, 0)
space.add(center_body)

# Initialize predictor models for each planet (we'll create them as needed)
predictors = {}  # planet_id -> model
optimizers = {}  # planet_id -> optimizer
trained_predictors = set()
# Black hole mode toggle state
black_hole_mode = False
_stored_center_mass = 0
_stored_g_constant = 0


# Entity lists
planets = []
photons = []

if START_WITH_SAMPLE_PLANETS:
    for i, distance in enumerate(SAMPLE_PLANET_DISTANCES):
        angle = (i * 2 * math.pi) / len(SAMPLE_PLANET_DISTANCES)
        planet = create_planet(
            space, distance, angle=angle, color_index=i,
            center_position=center_body.position,
            g_constant=G_CONSTANT, center_mass=CENTER_MASS
        )
        planets.append(planet)

if START_WITH_SAMPLE_PHOTONS and PHOTON_COUNT > 0:
    for i in range(PHOTON_COUNT):
        angle = (i * 2 * math.pi) / PHOTON_COUNT
        distance = random.randint(80, 120)
        photon = create_photon(
            space, distance, angle,
            center_position=center_body.position,
            g_constant=G_CONSTANT, center_mass=CENTER_MASS
        )
        photons.append(photon)

print(f"Starting empty scene with {len(planets)} planets and {len(photons)} photons", flush=True)

# Font for UI
font = pygame.font.Font(None, FONT_SIZE_LARGE)
small_font = pygame.font.Font(None, FONT_SIZE_MEDIUM)
title_font = pygame.font.Font(None, 36)  # Keep original for title

# Camera state
camera_pos = INITIAL_CAMERA_POS.copy()  # x, y, z position
camera_rot = INITIAL_CAMERA_ROT.copy()  # pitch, yaw, roll (in radians)
move_speed = MOVE_SPEED
rot_speed = ROT_SPEED
zoom_level = INITIAL_ZOOM
min_zoom = MIN_ZOOM
max_zoom = MAX_ZOOM

# UI state
show_ui = True
ui_hidden_timer = 0
AUTO_HIDE_DELAY = 300  # frames (5 seconds at 60fps)
selected_planet = None  # Currently selected planet for modification
selected_photon = None  # Currently selected photon for deletion
status_message = "Click empty space to add a planet. Right-click to add a photon."
status_timer = target_fps * 5
last_error = None
error_timer = 0

# Main loop
running = True
paused = False
frame_count = 0
ai_train_counter = 0  # Train AI less frequently for performance


def set_status(message, frames=None):
    """Show a short status message in the UI."""
    global status_message, status_timer
    status_message = message
    status_timer = frames if frames is not None else target_fps * 4


def log_runtime_error(context, exc):
    """Log runtime errors and keep the simulator window alive where possible."""
    global last_error, error_timer, paused
    last_error = f"{context}: {exc.__class__.__name__}: {exc}"
    error_timer = target_fps * 8
    paused = True
    trace = traceback.format_exc()
    print(last_error, flush=True)
    print(trace, flush=True)
    with ERROR_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{context}] {exc.__class__.__name__}: {exc}\n")
        log_file.write(trace)


def remove_predictor_for_body(body):
    planet_id = id(body)
    predictors.pop(planet_id, None)
    optimizers.pop(planet_id, None)
    trained_predictors.discard(planet_id)


def safe_space_remove(entity):
    try:
        space.remove(entity['body'], entity['shape'])
    except Exception as exc:
        log_runtime_error("remove entity", exc)


def remove_planet(planet):
    global selected_planet
    if planet in planets:
        remove_predictor_for_body(planet['body'])
        safe_space_remove(planet)
        planets.remove(planet)
        if selected_planet is planet:
            selected_planet = None
        set_status("Planet removed.")


def remove_photon(photon):
    global selected_photon
    if photon in photons:
        safe_space_remove(photon)
        photons.remove(photon)
        if selected_photon is photon:
            selected_photon = None
        set_status("Photon removed.")


def clear_scene():
    global selected_planet, selected_photon
    for planet in list(planets):
        remove_planet(planet)
    for photon in list(photons):
        remove_photon(photon)
    selected_planet = None
    selected_photon = None
    predictors.clear()
    optimizers.clear()
    trained_predictors.clear()
    set_status("Scene cleared.")


def reset_view():
    camera_pos[:] = INITIAL_CAMERA_POS.copy()
    camera_rot[:] = INITIAL_CAMERA_ROT.copy()
    set_status("Camera reset.")


def add_planet_at_world(world_pos):
    dx = world_pos[0] - center_body.position.x
    dy = world_pos[1] - center_body.position.y
    distance = math.sqrt(dx * dx + dy * dy)
    if distance <= 50:
        set_status("Too close to the center. Click farther out to add a planet.")
        return None

    angle = math.atan2(dy, dx)
    mass = random.uniform(5, 25)
    planet = create_planet(
        space, distance, angle, mass,
        color_index=len(planets),
        center_position=center_body.position,
        g_constant=G_CONSTANT, center_mass=CENTER_MASS
    )
    planets.append(planet)
    set_status(f"Added {planet['name']} at radius {int(distance)}.")
    return planet


def add_random_photon():
    angle = random.uniform(0, 2 * math.pi)
    distance = random.randint(100, 200)
    photon = create_photon(
        space, distance, angle,
        center_position=center_body.position,
        g_constant=G_CONSTANT, center_mass=CENTER_MASS
    )
    photons.append(photon)
    set_status("Added photon.")
    return photon


def add_photon_at_world(world_pos):
    dx = world_pos[0] - center_body.position.x
    dy = world_pos[1] - center_body.position.y
    distance = max(math.sqrt(dx * dx + dy * dy), 50)
    angle = math.atan2(dy, dx)
    photon = create_photon(
        space, distance, angle,
        center_position=center_body.position,
        g_constant=G_CONSTANT, center_mass=CENTER_MASS
    )
    photons.append(photon)
    set_status("Added photon.")
    return photon


try:
    # Run until user quits (remove frame limit for normal operation)
    print("Starting main loop", flush=True)
    while running:
        # Clear screen at start of loop with dark space background
        screen.fill((5, 5, 15))  # Very dark space background

        # Draw gridlines to help visualize dimensions
        # draw_gridlines(screen, camera_pos, camera_rot, zoom_level, WIDTH, HEIGHT)

        # Handle events
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
                    # Toggle UI visibility
                    show_ui = not show_ui
                    ui_hidden_timer = 0
                    set_status("UI shown." if show_ui else "UI hidden.")
                elif event.key == pygame.K_BACKSPACE:
                    # Remove last planet (if any exist)
                    if len(planets) > 0:
                        remove_planet(planets[-1])
                elif event.key == pygame.K_g:
                    # Generate training data and train models (do this less
                    # frequently for performance)
                    ai_train_counter = 0  # Reset counter to trigger training
                    set_status("AI training queued.")
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    # Increase gravitational constant
                    G_CONSTANT *= 1.1
                    set_status(f"Gravity increased to {int(G_CONSTANT)}.")
                elif event.key == pygame.K_MINUS:
                    # Decrease gravitational constant
                    G_CONSTANT /= 1.1
                    set_status(f"Gravity decreased to {int(G_CONSTANT)}.")
                elif event.key == pygame.K_p:
                    add_random_photon()
                elif event.key == pygame.K_b:
                    if not black_hole_mode:
                        # Enter black hole mode
                        _stored_center_mass = CENTER_MASS
                        _stored_g_constant = G_CONSTANT
                        CENTER_MASS = BLACK_HOLE_THRESHOLD * 2  # Well above threshold
                        G_CONSTANT = 2000000  # Increase gravitational constant for stronger effect
                        black_hole_mode = True
                        set_status("Black hole mode enabled.")
                    else:
                        # Exit black hole mode
                        CENTER_MASS = _stored_center_mass
                        G_CONSTANT = _stored_g_constant
                        black_hole_mode = False
                        set_status("Black hole mode disabled.")
                elif event.key == pygame.K_c:
                    clear_scene()
                elif event.key == pygame.K_HOME:
                    reset_view()
                elif event.key == pygame.K_DELETE:
                    # Remove selected photon first (higher priority)
                    if selected_photon is not None:
                        remove_photon(selected_photon)
                    # Then remove selected planet if no photon selected
                    elif selected_planet is not None:
                        remove_planet(selected_planet)
                elif event.key == pygame.K_m:
                    # Increase mass of selected planet
                    if selected_planet is not None:
                        update_planet_mass(selected_planet, 1.2)
                        set_status(f"{selected_planet['name']} mass: {selected_planet['mass']:.1f}")
                elif event.key == pygame.K_n:
                    # Decrease mass of selected planet
                    if selected_planet is not None and selected_planet['mass'] > 1:
                        update_planet_mass(selected_planet, 1/1.2)
                        set_status(f"{selected_planet['name']} mass: {selected_planet['mass']:.1f}")
                elif event.key == pygame.K_r:
                    # Increase radius of selected planet (visual only)
                    if selected_planet is not None:
                        update_planet_radius(selected_planet, 1.2)
                        set_status(f"{selected_planet['name']} visual size increased.")
                elif event.key == pygame.K_f:
                    # Decrease radius of selected planet (visual only)
                    if selected_planet is not None:
                        update_planet_radius(selected_planet, 1/1.2)
                        set_status(f"{selected_planet['name']} visual size decreased.")
                elif event.key == pygame.K_i:
                    # Increase zoom (move closer)
                    zoom_level = min(max_zoom, zoom_level * 1.1)
                    set_status(f"Zoom: {zoom_level:.2f}x")
                elif event.key == pygame.K_o:
                    # Decrease zoom (move farther)
                    zoom_level = max(min_zoom, zoom_level / 1.1)
                    set_status(f"Zoom: {zoom_level:.2f}x")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Mouse wheel up
                    zoom_level = min(max_zoom, zoom_level * 1.1)
                elif event.button == 5:  # Mouse wheel down
                    zoom_level = max(min_zoom, zoom_level / 1.1)
                elif event.button == 1:  # Left click - add planet or select existing
                    world_pos = screen_to_world(
                        (event.pos[0], event.pos[1]), camera_pos, camera_rot, zoom_level, WIDTH, HEIGHT)  # Assume z=0 for clicking

                    # First check if we clicked on an existing photon
                    clicked_photon = select_entity_at_position(
                        photons, event.pos, camera_pos, camera_rot, zoom_level, WIDTH, HEIGHT, CLICK_DISTANCE_THRESHOLD)

                    if clicked_photon:
                        # Select the photon for deletion
                        selected_photon = clicked_photon
                        selected_planet = None  # Deselect any planet
                        set_status("Photon selected. Press Delete to remove it.")
                    else:
                        # First check if we clicked on an existing planet
                        clicked_planet = select_entity_at_position(
                            planets, event.pos, camera_pos, camera_rot, zoom_level, WIDTH, HEIGHT, CLICK_DISTANCE_THRESHOLD)

                        if clicked_planet:
                            # Select the planet for modification
                            selected_planet = clicked_planet
                            selected_photon = None  # Deselect any photon
                            set_status(f"{selected_planet['name']} selected.")
                        else:
                            add_planet_at_world(world_pos)
                elif event.button == 3:  # Right click - add photon at cursor
                    world_pos = screen_to_world(
                        (event.pos[0], event.pos[1]), camera_pos, camera_rot, zoom_level, WIDTH, HEIGHT)
                    add_photon_at_world(world_pos)
            elif event.type == pygame.MOUSEMOTION:
                if event.buttons[0]:  # Left mouse button dragged for panning
                    # Rotate camera instead of panning for 3D feel
                    camera_rot[1] -= event.rel[0] * \
                        rot_speed  # Yaw (left/right)
                    camera_rot[0] -= event.rel[1] * \
                        rot_speed  # Pitch (up/down)
                    # Limit pitch to avoid flipping
                    camera_rot[0] = max(-3.14159 / 2 + 0.1,
                                        min(3.14159 / 2 - 0.1, camera_rot[0]))

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
        camera_rot[0] = max(-3.14159 / 2 + 0.1,
                            min(3.14159 / 2 - 0.1, camera_rot[0]))

        # Auto-hide UI after delay
        if show_ui and ui_hidden_timer > 0:
            ui_hidden_timer -= 1
            if ui_hidden_timer <= 0:
                show_ui = False

        # Show UI temporarily when interacting
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

        if not paused:
            try:
                if planets or photons:
                    forces = calculate_forces(
                        planets,
                        photons,
                        center_body,
                        SOFTENING_SQUARED,
                        G_CONSTANT,
                        CENTER_MASS
                    )
                    apply_forces(planets, photons, forces)
                    update_positions(planets, photons)

                    # Occasionally use AI prediction to nudge velocity.
                    if frame_count % 60 == 0:
                        for planet_data in planets:
                            if len(planet_data['history']) > AI_MIN_HISTORY_FOR_PREDICTION:
                                planet_id = id(planet_data['body'])
                                if planet_id not in predictors:
                                    predictors[planet_id] = orbit_predictor.OrbitPredictor(input_size=AI_INPUT_SIZE)
                                    optimizers[planet_id] = optim.Adam(
                                        predictors[planet_id].parameters(), lr=AI_LEARNING_RATE)

                                predicted_pos = None
                                if planet_id in trained_predictors:
                                    predicted_pos = orbit_predictor.predict_planet_position(
                                        predictors[planet_id],
                                        planet_data['history'])
                                if predicted_pos is not None:
                                    blend_factor = 0.3
                                    current_pos = np.array(
                                        [planet_data['body'].position.x, planet_data['body'].position.y])
                                    predicted_pos_np = np.array(predicted_pos)
                                    blended_pos = current_pos * \
                                        (1 - blend_factor) + predicted_pos_np * blend_factor
                                    adjustment = (blended_pos - current_pos) * 0.1
                                    planet_data['body'].velocity += (
                                        adjustment[0] / planet_data['body'].mass,
                                        adjustment[1] / planet_data['body'].mass)

                limit_velocities(photons, MAX_VELOCITY)
                # Limit planet velocities to prevent numerical explosion
                for planet_data in planets:
                    speed = math.sqrt(
                        planet_data['body'].velocity.x**2 +
                        planet_data['body'].velocity.y**2)
                    if speed > MAX_VELOCITY:
                        scale = MAX_VELOCITY / speed
                        planet_data['body'].velocity = (
                            planet_data['body'].velocity.x * scale,
                            planet_data['body'].velocity.y * scale)
                photons_to_remove = check_black_hole_capture(
                    photons,
                    center_body,
                    CENTER_MASS,
                    BLACK_HOLE_THRESHOLD,
                    SPEED_OF_LIGHT,
                    G_CONSTANT
                )
                # Remove photons that have fallen into the black hole
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
                # Remove planets that have fallen into the black hole
                for planet_data in planets_to_remove:
                    remove_planet(planet_data)
                update_physics_space(space, PHYSICS_TIME_STEP)

                # Train AI models less frequently for better performance.
                ai_train_counter += 1
                if ai_train_counter >= AI_TRAIN_INTERVAL:
                    ai_train_counter = 0
                    for planet_data in planets:
                        planet_id = id(planet_data['body'])
                        if planet_id not in predictors:
                            predictors[planet_id] = orbit_predictor.OrbitPredictor(input_size=AI_INPUT_SIZE)
                            optimizers[planet_id] = optim.Adam(
                                predictors[planet_id].parameters(), lr=AI_LEARNING_RATE)
                        if len(planet_data['history']) > AI_MIN_HISTORY_FOR_TRAINING:
                            orbit_predictor.train_planet_predictor(
                                predictors[planet_id],
                                optimizers[planet_id],
                                planet_data['history'],
                                epochs=AI_EPOCHS_PER_TRAINING)
                            trained_predictors.add(planet_id)
            except Exception as exc:
                log_runtime_error("physics update", exc)

        frame_count += 1
        if status_timer > 0:
            status_timer -= 1
        if error_timer > 0:
            error_timer -= 1
        elif last_error is not None:
            last_error = None

        # Print FPS and object counts to console every second (at target_fps)
        if frame_count % target_fps == 0:
            fps = clock.get_fps()
            print(
                f"FPS: {fps:.1f}, Planets: {len(planets)}, Photons: {len(photons)}, Paused: {paused}")

        # Render (always render, even when paused)
        if frame_count == 1:
            print("About to render...", flush=True)
            print(f"Drawing {len(planets)} planets and {len(photons)} photons", flush=True)

        try:
            # Draw everything
            draw_trails(screen, planets, photons, camera_pos, camera_rot, zoom_level, WIDTH, HEIGHT)
            draw_bodies(screen, planets, photons, center_body, camera_pos, camera_rot, zoom_level, WIDTH, HEIGHT, CENTER_MASS)
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

            draw_ui(screen, font, small_font, title_font, planets, photons, paused, show_ui,
                    G_CONSTANT, CENTER_MASS, BLACK_HOLE_THRESHOLD, camera_pos, camera_rot,
                    zoom_level, WIDTH, HEIGHT, clock.get_fps(), selected_photon or selected_planet,
                    status_message if status_timer > 0 else None, last_error)

            pygame.display.flip()
        except Exception as exc:
            log_runtime_error("render", exc)
        clock.tick(target_fps)  # Run at stable target FPS

except Exception as e:
    log_runtime_error("fatal main loop", e)

finally:
    pygame.quit()