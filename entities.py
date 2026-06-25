# entities.py
# Entity management for the StarDance orbital physics simulator

import pymunk
import math
import random
from collections import deque
from config import *
from rendering import world_to_screen


def unpack_position(position):
    """Return an (x, y) tuple for tuple/list or Pymunk Vec2d positions."""
    if hasattr(position, "x") and hasattr(position, "y"):
        return position.x, position.y
    return position


def create_planet(space, distance, angle=None, mass=None, color_index=None, name=None,
                  center_position=(0.0, 0.0), g_constant=G_CONSTANT, center_mass=CENTER_MASS):
    """Create a planet at given distance from center"""
    if mass is None:
        mass = PLANET_MASS
    if color_index is None:
        color_index = 0
    if angle is None:
        angle = random.uniform(0, 2 * math.pi)
    distance = max(float(distance), 1.0)

    # Position in world coordinates relative to the central body.
    center_x, center_y = unpack_position(center_position)
    x = center_x + distance * math.cos(angle)
    y = center_y + distance * math.sin(angle)

    # Create planet body
    planet_body = pymunk.Body(
        mass, pymunk.moment_for_circle(
            mass, 0, 12, (0, 0)))
    planet_body.position = (x, y)

    # Calculate orbital velocity for circular orbit: v = sqrt(G * M / r)
    orbital_speed = math.sqrt(max(g_constant * center_mass / distance, 0.0))
    # Velocity vector: tangential (perpendicular to radial vector)
    # For counter-clockwise orbit: (-sin(angle), cos(angle)) * orbital_speed
    vx = -math.sin(angle) * orbital_speed
    vy = math.cos(angle) * orbital_speed
    planet_body.velocity = (vx, vy)

    # Create shape
    planet_shape = pymunk.Circle(planet_body, 12)
    planet_shape.elasticity = 0.95
    planet_shape.friction = 0.05
    planet_shape.collision_type = 1

    space.add(planet_body, planet_shape)

    # Store planet data
    planet_data = {
        'body': planet_body,
        'shape': planet_shape,
        'color': PLANET_COLORS[color_index % len(PLANET_COLORS)],
        'name': name if name is not None else f"Planet {color_index + 1}",
        'mass': mass,
        'trail': deque(maxlen=TRAIL_LENGTH),  # Limited trail history
        # For AI predictions
        'predicted_positions': deque(maxlen=PREDICTION_FRAMES),
        'history': deque(maxlen=50)  # For training data
    }

    return planet_data


def create_photon(space, distance, angle=None, center_position=(0.0, 0.0),
                  g_constant=G_CONSTANT, center_mass=CENTER_MASS):
    """Create a photon (light particle) at given distance from center"""
    if angle is None:
        angle = random.uniform(0, 2 * math.pi)
    distance = max(float(distance), 1.0)

    # Position in world coordinates relative to the central body.
    center_x, center_y = unpack_position(center_position)
    x = center_x + distance * math.cos(angle)
    y = center_y + distance * math.sin(angle)

    # Create photon body (very small mass)
    photon_body = pymunk.Body(
        PHOTON_MASS, pymunk.moment_for_circle(
            PHOTON_MASS, 0, 2, (0, 0)))
    photon_body.position = (x, y)

    # Photons move at speed of light (simulated)
    # Tangential velocity for orbit
    # Slightly faster than orbital
    photon_speed = math.sqrt(max(g_constant * center_mass / distance, 0.0)) * 1.2
    vx = -math.sin(angle) * photon_speed
    vy = math.cos(angle) * photon_speed
    photon_body.velocity = (vx, vy)

    # Create shape (smaller than planets)
    photon_shape = pymunk.Circle(photon_body, 2)
    photon_shape.elasticity = 0.99  # Very bouncy
    photon_shape.friction = 0.01   # Very slippery
    photon_shape.collision_type = 2  # Different collision type

    space.add(photon_body, photon_shape)

    # Store photon data
    photon_data = {
        'body': photon_body,
        'shape': photon_shape,
        'color': PHOTON_COLOR,
        'trail': deque(maxlen=TRAIL_LENGTH // 2),  # Shorter trail for photons
        'history': deque(maxlen=30)  # Shorter history
    }

    return photon_data


def create_random_planet(space, min_distance=100, max_distance=600):
    """Create a planet at a random position"""
    distance = random.uniform(min_distance, max_distance)
    angle = random.uniform(0, 2 * math.pi)
    mass = random.uniform(5, 25)
    return create_planet(space, distance, angle, mass)


def create_random_photon(space, min_distance=80, max_distance=200):
    """Create a photon at a random position"""
    distance = random.randint(min_distance, max_distance)
    angle = random.uniform(0, 2 * math.pi)
    return create_photon(space, distance, angle)


def remove_entity(space, entities, entity_to_remove):
    """Remove an entity from the simulation"""
    if entity_to_remove in entities:
        space.remove(entity_to_remove['body'], entity_to_remove['shape'])
        entities.remove(entity_to_remove)
        return True
    return False


def select_entity_at_position(entities, screen_pos, camera_pos, camera_rot, zoom_level, width, height, threshold=25):
    """Select an entity at the given screen position"""
    for entity in entities:
        entity_screen_pos = world_to_screen(
            entity['body'].position, camera_pos, camera_rot, zoom_level, width, height)
        dx = screen_pos[0] - entity_screen_pos[0]
        dy = screen_pos[1] - entity_screen_pos[1]
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < threshold:
            return entity
    return None


def update_planet_mass(planet, multiplier):
    """Update the mass and moment of a planet without replacing its body."""
    if planet is None:
        return False

    planet['mass'] = max(0.1, planet['mass'] * multiplier)
    body = planet['body']
    body.mass = planet['mass']
    body.moment = pymunk.moment_for_circle(planet['mass'], 0, 12, (0, 0))

    return True


def update_planet_radius(planet, multiplier):
    """Update the visual radius of a planet (visual only)"""
    if planet is None:
        return False

    # Store radius multiplier in planet data
    if 'radius_multiplier' not in planet:
        planet['radius_multiplier'] = 1.0
    planet['radius_multiplier'] *= multiplier

    # Minimum radius multiplier
    if planet['radius_multiplier'] < 0.3:
        planet['radius_multiplier'] = 0.3

    return True
