# rendering.py
# Rendering engine for the StarDance orbital physics simulator

import pygame
import math
from config import *
import numpy as np


# Sun texture - will be loaded after display is initialized
SUN_TEXTURE = None
# Black hole texture - will be generated if not already loaded
BLACKHOLE_TEXTURE = None


def load_sun_texture():
    """Load the sun texture if not already loaded."""
    global SUN_TEXTURE
    if SUN_TEXTURE is not None:
        return
    try:
        # Try to load a sun texture image from the current directory
        SUN_TEXTURE = pygame.image.load("sun_texture.png").convert_alpha()
    except Exception as e:
        # If loading fails, we'll fall back to drawing a circle
        print(f"Could not load sun texture: {e}")
        SUN_TEXTURE = None


def load_blackhole_texture():
    """Create a simple black hole texture if not already created."""
    global BLACKHOLE_TEXTURE
    if BLACKHOLE_TEXTURE is not None:
        return
    size = 256
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    # Fill with transparent black (fully opaque black)
    surf.fill((0, 0, 0, 255))
    center = (size // 2, size // 2)
    # Outer ring radius
    outer_radius = size // 2 - 5
    # Inner radius for the black hole 'shadow' (just a smaller circle)
    inner_radius = outer_radius // 2
    # Draw a bright ring (accretion disk) - yellowish
    ring_color = (255, 165, 0, 255)  # orange
    pygame.draw.circle(surf, ring_color, center, outer_radius, 3)
    # Optionally, add a inner darker circle to represent the event horizon
    pygame.draw.circle(surf, (0, 0, 0, 255), center, inner_radius)
    BLACKHOLE_TEXTURE = surf


def load_blackhole_texture():
    """Create a black hole texture if not already created."""
    global BLACKHOLE_TEXTURE
    if BLACKHOLE_TEXTURE is not None:
        return
    # Create a simple black hole texture: black background with a white/accretion ring
    size = 256
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    # Fill with transparent black (fully opaque black)
    surf.fill((0, 0, 0, 255))
    # Draw a white/accretion ring
    center = (size // 2, size // 2)
    radius_outer = size // 2 - 5
    # Ring thickness
    thickness = max(2, size // 32)
    # Use a yellow-orange color for the accretion disk
    ring_color = (255, 165, 0, 255)  # orange
    # Draw a circle for the outer edge
    pygame.draw.circle(surf, ring_color, center, radius_outer, thickness)
    # Optionally add inner dark circle (event horizon) - already black
    # For a more pronounced effect, draw a darker inner circle
    inner_radius = radius_outer // 2
    pygame.draw.circle(surf, (20, 20, 20, 255), center, inner_radius)
    BLACKHOLE_TEXTURE = surf


def world_to_screen(world_pos, camera_pos, camera_rot, zoom_level, width, height):
    """Convert world coordinates to screen coordinates with 3D rotation"""
    # Handle both 2D and 3D positions
    if hasattr(world_pos, "x") and hasattr(world_pos, "y"):
        x, y = world_pos.x, world_pos.y
        z = getattr(world_pos, "z", 0)
    elif len(world_pos) == 2:
        x, y = world_pos
        z = 0
    else:
        x, y, z = world_pos

    # Apply camera position (translate world relative to camera)
    x_rel = x - camera_pos[0]
    y_rel = y - camera_pos[1]
    z_rel = z - camera_pos[2]

    # Apply camera rotation (pitch, yaw, roll)
    # Rotate around Y axis (yaw)
    cos_yaw = math.cos(camera_rot[1])
    sin_yaw = math.sin(camera_rot[1])
    x_rot = x_rel * cos_yaw - z_rel * sin_yaw
    z_rot = x_rel * sin_yaw + z_rel * cos_yaw
    y_rot = y_rel

    # Rotate around X axis (pitch)
    cos_pitch = math.cos(camera_rot[0])
    sin_pitch = math.sin(camera_rot[0])
    y_final = y_rot * cos_pitch - z_rot * sin_pitch
    z_final = y_rot * sin_pitch + z_rot * cos_pitch
    x_final = x_rot

    # Simple perspective projection
    if z_final <= 0:
        z_final = 0.1  # Avoid division by zero

    scale = min(width, height) / 1000 * zoom_level * 300 / z_final
    screen_x = int(x_final * scale + width // 2)
    # Negative Y for screen coordinates
    screen_y = int(-y_final * scale + height // 2)

    return (screen_x, screen_y)


def screen_to_world(screen_pos, camera_pos, camera_rot, zoom_level, width, height):
    """Convert screen coordinates to world coordinates (returns 3D point with z=0)"""
    screen_x, screen_y = screen_pos

    projection_scale = min(width, height) / 1000 * zoom_level * 300
    if projection_scale <= 0:
        return (camera_pos[0], camera_pos[1], 0)

    # Build a ray in camera space, undo pitch/yaw, then intersect world z=0.
    ray_x = (screen_x - width // 2) / projection_scale
    ray_y = -(screen_y - height // 2) / projection_scale
    ray_z = 1.0

    cos_pitch = math.cos(-camera_rot[0])
    sin_pitch = math.sin(-camera_rot[0])
    y_unpitched = ray_y * cos_pitch - ray_z * sin_pitch
    z_unpitched = ray_y * sin_pitch + ray_z * cos_pitch
    x_unpitched = ray_x

    cos_yaw = math.cos(-camera_rot[1])
    sin_yaw = math.sin(-camera_rot[1])
    dir_x = x_unpitched * cos_yaw - z_unpitched * sin_yaw
    dir_z = x_unpitched * sin_yaw + z_unpitched * cos_yaw
    dir_y = y_unpitched

    if abs(dir_z) < 1e-6:
        return (camera_pos[0], camera_pos[1], 0)

    t = -camera_pos[2] / dir_z
    world_x = camera_pos[0] + dir_x * t
    world_y = camera_pos[1] + dir_y * t
    world_z = 0

    return (world_x, world_y, world_z)


def draw_gridlines(screen, camera_pos, camera_rot, zoom_level, width, height):
    """Draw gridlines to help visualize dimensions"""
    # Only draw gridlines if zoomed in close enough to see them
    if zoom_level > 0.1:  # Only draw when zoomed in reasonably close
        # Grid color - dim white/gray
        grid_color = GRID_COLOR
        grid_spacing = GRID_SPACING  # Distance between grid lines in world units

        # Calculate visible world bounds
        world_left, world_right = -width//2 // zoom_level + camera_pos[0], width//2 // zoom_level + camera_pos[0]
        world_top, world_bottom = -height//2 // zoom_level + camera_pos[1], height//2 // zoom_level + camera_pos[1]

        # Draw horizontal lines (constant y)
        y_start = int((world_bottom) // grid_spacing) * grid_spacing
        y_end = int((world_top) // grid_spacing) * grid_spacing
        for y in range(y_start, y_end + 1, grid_spacing):
            # Convert world coordinates to screen coordinates
            start_pos = world_to_screen((world_left, y, 0), camera_pos, camera_rot, zoom_level, width, height)
            end_pos = world_to_screen((world_right, y, 0), camera_pos, camera_rot, zoom_level, width, height)

            # Only draw if both points are visible on screen
            if (0 <= start_pos[0] <= width and 0 <= start_pos[1] <= height) or \
               (0 <= end_pos[0] <= width and 0 <= end_pos[1] <= height):
                # Make the center axes (x=0, y=0) more visible
                if y == 0:
                    pygame.draw.line(screen, AXIS_COLOR, start_pos, end_pos, 2)
                else:
                    pygame.draw.line(screen, grid_color, start_pos, end_pos, 1)

        # Draw vertical lines (constant x)
        x_start = int((world_left) // grid_spacing) * grid_spacing
        x_end = int((world_right) // grid_spacing) * grid_spacing
        for x in range(x_start, x_end + 1, grid_spacing):
            # Convert world coordinates to screen coordinates
            start_pos = world_to_screen((x, world_bottom, 0), camera_pos, camera_rot, zoom_level, width, height)
            end_pos = world_to_screen((x, world_top, 0), camera_pos, camera_rot, zoom_level, width, height)

            # Only draw if both points are visible on screen
            if (0 <= start_pos[0] <= width and 0 <= start_pos[1] <= height) or \
               (0 <= end_pos[0] <= width and 0 <= end_pos[1] <= height):
                # Make the center axes (x=0, y=0) more visible
                if x == 0:
                    pygame.draw.line(screen, AXIS_COLOR, start_pos, end_pos, 2)
                else:
                    pygame.draw.line(screen, grid_color, start_pos, end_pos, 1)

                # Optional: Draw Z-axis grid lines (for 3D depth perception)
                # Draw lines at different Z depths to show 3D space
                for z_offset in [-100, 0, 100]:  # Different Z levels
                    if z_offset != 0:  # Skip Z=0 as it's the main plane
                        z_start_pos = world_to_screen((x, world_bottom, z_offset), camera_pos, camera_rot, zoom_level, width, height)
                        z_end_pos = world_to_screen((x, world_top, z_offset), camera_pos, camera_rot, zoom_level, width, height)
                        # Only draw if reasonably close to screen
                        if -100 <= z_start_pos[1] <= height + 100 and -100 <= z_end_pos[1] <= height + 100:
                            # Dimmer color for Z-axis lines
                            z_color = Z_AXIS_COLOR_POS if z_offset > 0 else Z_AXIS_COLOR_NEG
                            pygame.draw.line(screen, z_color, z_start_pos, z_end_pos, 1)


def draw_trails(screen, planets, photons, camera_pos, camera_rot, zoom_level, width, height):
    """Draw trails for planets and photons with fading effect"""
    # Draw planet trails
    for planet_data in planets:
        if len(planet_data['trail']) > 1:
            points = []
            for i, pos in enumerate(planet_data['trail']):
                # Calculate alpha based on position in trail (newer = more opaque)
                alpha = int(255 * (i + 1) / len(planet_data['trail']))
                color = (*planet_data['color'], alpha)  # Add alpha channel
                screen_pos = world_to_screen(pos, camera_pos, camera_rot, zoom_level, width, height)
                points.append((screen_pos[0], screen_pos[1]))

            # Draw as lines with varying width/alpha
            if len(points) >= 2:
                # Draw multiple layers for a glow effect
                for i in range(len(points) - 1):
                    alpha = int(100 * (i + 1) / len(points))  # Fade out older points
                    if alpha > 0:
                        width_factor = 1 + (i / len(points))  # Thicker for newer points
                        pygame.draw.line(
                            screen,
                            planet_data['color'],
                            points[i],
                            points[i + 1],
                            max(1, int(width_factor))
                        )

    # Draw photon trails
    for photon_data in photons:
        if len(photon_data['trail']) > 1:
            points = []
            for i, pos in enumerate(photon_data['trail']):
                # Calculate alpha based on position in trail (newer = more opaque)
                alpha = int(155 * (i + 1) / len(photon_data['trail']))  # Photons are dimmer
                color = (*PHOTON_COLOR, alpha)  # Add alpha channel
                screen_pos = world_to_screen(pos, camera_pos, camera_rot, zoom_level, width, height)
                points.append((screen_pos[0], screen_pos[1]))

            # Draw as lines with varying width/alpha
            if len(points) >= 2:
                # Draw multiple layers for a glow effect
                for i in range(len(points) - 1):
                    alpha = int(100 * (i + 1) / len(points))  # Fade out older points
                    if alpha > 0:
                        width_factor = 0.5 + (i / len(points))  # Thinner for photons
                        pygame.draw.line(
                            screen,
                            PHOTON_COLOR,
                            points[i],
                            points[i + 1],
                            max(1, int(width_factor))
                        )


def draw_bodies(screen, planets, photons, center_body, camera_pos, camera_rot, zoom_level, width, height, center_mass):
    """Draw all celestial bodies"""
    # Ensure sun texture is loaded if possible
    load_sun_texture()
    # Ensure black hole texture is generated if needed
    load_blackhole_texture()

    # Determine if we should render as a black hole
    is_black_hole = center_mass > BLACK_HOLE_THRESHOLD

    # Draw central body (sun or black hole)
    sun_pos = world_to_screen(center_body.position, camera_pos, camera_rot, zoom_level, width, height)
    if is_black_hole:
        # Use black hole texture if available, else draw a black circle with optional accretion ring
        if BLACKHOLE_TEXTURE is not None:
            bh_size = 30  # diameter in pixels
            scaled_bh = pygame.transform.scale(BLACKHOLE_TEXTURE, (bh_size, bh_size))
            bh_rect = scaled_bh.get_rect(center=sun_pos)
            screen.blit(scaled_bh, bh_rect)
        else:
            # Fallback: simple black circle
            pygame.draw.circle(screen, (0, 0, 0), sun_pos, 15)
    else:
        # Sun rendering
        if SUN_TEXTURE is not None:
            # Calculate the size to draw the sun (fixed diameter of 30 pixels to match the original circle radius of 15)
            sun_size = 30  # diameter in pixels
            scaled_sun = pygame.transform.scale(SUN_TEXTURE, (sun_size, sun_size))
            # Blit the image centered at sun_pos
            sun_rect = scaled_sun.get_rect(center=sun_pos)
            screen.blit(scaled_sun, sun_rect)
        else:
            # Fallback to the circle
            pygame.draw.circle(screen, SUN_COLOR, sun_pos, 15)

    # Draw planets
    for planet_data in planets:
        pos = world_to_screen(planet_data['body'].position, camera_pos, camera_rot, zoom_level, width, height)
        # Apply radius multiplier if available (for visual scaling)
        radius = 8
        if 'radius_multiplier' in planet_data:
            radius = int(8 * planet_data['radius_multiplier'])
        pygame.draw.circle(screen, planet_data['color'], pos, radius)

    # Draw photons
    for photon_data in photons:
        pos = world_to_screen(photon_data['body'].position, camera_pos, camera_rot, zoom_level, width, height)
        pygame.draw.circle(screen, PHOTON_COLOR, pos, 3)


def draw_selection_indicator(screen, selected_entity, camera_pos, camera_rot, zoom_level, width, height, frame_count):
    """Draw selection indicator for the selected body."""
    if selected_entity is not None:
        entity_body = selected_entity['body']
        entity_pos = world_to_screen(entity_body.position, camera_pos, camera_rot, zoom_level, width, height)
        if (frame_count // 15) % 2:  # Blink every 15 frames
            selection_radius = selected_entity.get('radius_multiplier', 1.0) * 10
            pygame.draw.circle(screen, (255, 255, 255), entity_pos, int(selection_radius), 2)


def draw_ui(screen, font, small_font, title_font, planets, photons, paused, show_ui,
            g_constant, center_mass, black_hole_threshold, camera_pos, camera_rot, zoom_level,
            width, height, current_fps=0, selected_entity=None, status_message=None, last_error=None):
    """Draw user interface elements"""
    if not show_ui:
        return

    panel = pygame.Surface((390, 292), pygame.SRCALPHA)
    panel.fill(UI_BACKGROUND_COLOR)
    screen.blit(panel, (16, 16))
    pygame.draw.rect(screen, (80, 95, 130), (16, 16, 390, 292), 1)

    title = title_font.render("StarDance", True, UI_ACCENT_COLOR)
    screen.blit(title, (30, 28))

    state_text = "PAUSED" if paused else "RUNNING"
    state_color = UI_WARNING_COLOR if paused else (150, 240, 180)
    state = small_font.render(state_text, True, state_color)
    screen.blit(state, (315, 34))

    stats = [
        f"Planets {len(planets)}   Photons {len(photons)}   FPS {int(current_fps)}",
        f"G {int(g_constant)}   Center mass {int(center_mass)}",
        f"Zoom {zoom_level:.2f}x   Camera {camera_pos[0]:.0f}, {camera_pos[1]:.0f}",
    ]

    y = 72
    for line in stats:
        text = small_font.render(line, True, UI_TEXT_COLOR)
        screen.blit(text, (30, y))
        y += 24

    if selected_entity is not None:
        name = selected_entity.get("name", "Photon")
        mass = selected_entity["body"].mass
        selection = f"Selected: {name}   mass {mass:.2f}"
    else:
        selection = "Selected: none"
    screen.blit(small_font.render(selection, True, UI_ACCENT_COLOR), (30, y + 4))
    y += 36

    controls = [
        "Click planet   Right-click photon   Delete selected",
        "Space pause   C clear   Home reset view   H hide UI",
        "Arrows rotate   WASD pan   Mouse wheel zoom",
        "+/- gravity   M/N mass   R/F size   B black hole",
    ]
    for line in controls:
        text = small_font.render(line, True, (210, 215, 235))
        screen.blit(text, (30, y))
        y += 22

    if status_message:
        pygame.draw.rect(screen, (12, 18, 30), (16, height - 52, min(width - 32, 720), 36))
        pygame.draw.rect(screen, (60, 75, 105), (16, height - 52, min(width - 32, 720), 36), 1)
        screen.blit(small_font.render(status_message, True, UI_TEXT_COLOR), (30, height - 43))

    if last_error:
        error_box_width = min(width - 32, 900)
        pygame.draw.rect(screen, (42, 14, 18), (16, height - 92, error_box_width, 34))
        pygame.draw.rect(screen, UI_ERROR_COLOR, (16, height - 92, error_box_width, 34), 1)
        screen.blit(small_font.render(last_error[:95], True, UI_ERROR_COLOR), (30, height - 84))