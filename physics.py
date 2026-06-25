# physics.py
# Physics engine for the StarDance orbital physics simulator

import pymunk
import math
import numpy as np
from config import *


def add_central_force(forces, body, center_body, softening_squared, g_constant, center_mass):
    """Accumulate gravity from the central body onto a dynamic body."""
    dx = center_body.position.x - body.position.x
    dy = center_body.position.y - body.position.y
    distance_squared = dx * dx + dy * dy
    softened_distance_squared = distance_squared + softening_squared

    if distance_squared <= 0:
        return

    force_magnitude = g_constant * center_mass * body.mass / softened_distance_squared
    distance = math.sqrt(distance_squared)
    forces[id(body)][0] += (dx / distance) * force_magnitude
    forces[id(body)][1] += (dy / distance) * force_magnitude


def calculate_forces(planets, photons, center_body, softening_squared,
                     g_constant=G_CONSTANT, center_mass=CENTER_MASS):
    """Calculate gravitational forces between all bodies"""
    # Initialize forces for all planets and photons
    forces = {}

    for planet_data in planets:
        forces[id(planet_data['body'])] = [0.0, 0.0]
    for photon_data in photons:
        forces[id(photon_data['body'])] = [0.0, 0.0]

    # Calculate planet-planet interactions (optimized - only half the matrix)
    for i, planet_data1 in enumerate(planets):
        body1 = planet_data1['body']
        pos1 = (body1.position.x, body1.position.y)

        add_central_force(forces, body1, center_body, softening_squared, g_constant, center_mass)

        # Forces from other planets (only calculate half and mirror results)
        for j in range(i + 1, len(planets)):
            planet_data2 = planets[j]
            body2 = planet_data2['body']
            pos2 = (body2.position.x, body2.position.y)

            # Calculate force between planet1 and planet2
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            distance_squared = dx * dx + dy * dy

            # Force magnitude with softening: G * m1 * m2 / (r^2 + EPSILON^2)
            force_magnitude = g_constant * body1.mass * \
                body2.mass / (distance_squared + softening_squared)

            # Force direction (unit vector from planet1 to planet2)
            distance = math.sqrt(distance_squared)
            if distance > 0:  # Avoid division by zero
                fx = (dx / distance) * force_magnitude
                fy = (dy / distance) * force_magnitude
            else:
                fx = 0.0
                fy = 0.0

            # Add to forces (equal and opposite)
            forces[id(body1)][0] += fx
            forces[id(body1)][1] += fy
            forces[id(body2)][0] -= fx
            forces[id(body2)][1] -= fy

    # Calculate planet-photon interactions
    for planet_data in planets:
        planet_body = planet_data['body']
        planet_pos = (planet_body.position.x, planet_body.position.y)

        for photon_data in photons:
            photon_body = photon_data['body']
            photon_pos = (
                photon_body.position.x,
                photon_body.position.y)

            # Calculate force between planet and photon
            dx = photon_pos[0] - planet_pos[0]
            dy = photon_pos[1] - planet_pos[1]
            distance_squared = dx * dx + dy * dy

            # Force magnitude with softening: G * m_planet * m_photon / (r^2 + EPSILON^2)
            force_magnitude = g_constant * planet_body.mass * \
                photon_body.mass / (distance_squared + softening_squared)

            # Force direction (unit vector from planet to photon)
            distance = math.sqrt(distance_squared)
            if distance > 0:  # Avoid division by zero
                fx = (dx / distance) * force_magnitude
                fy = (dy / distance) * force_magnitude
            else:
                fx = 0.0
                fy = 0.0

            # Add to forces
            forces[id(planet_body)][0] += fx
            forces[id(planet_body)][1] += fy
            forces[id(photon_body)][0] -= fx
            forces[id(photon_body)][1] -= fy

    for photon_data in photons:
        add_central_force(
            forces,
            photon_data['body'],
            center_body,
            softening_squared,
            g_constant,
            center_mass
        )

    return forces


def apply_forces(planets, photons, forces):
    """Apply calculated forces to bodies"""
    # Apply forces and update positions
    for planet_data in planets:
        # Apply accumulated force
        fx, fy = forces.get(id(planet_data['body']), (0.0, 0.0))
        planet_data['body'].apply_force_at_local_point((fx, fy))

    # Apply forces to photons
    for photon_data in photons:
        # Apply accumulated force
        fx, fy = forces.get(id(photon_data['body']), (0.0, 0.0))
        photon_data['body'].apply_force_at_local_point((fx, fy))


def update_positions(planets, photons):
    """Update positions and store history for training/prediction"""
    # Store current state for history/training and update trails
    for planet_data in planets:
        # Store current state for history/training
        pos = (planet_data['body'].position.x, planet_data['body'].position.y)
        vel = (planet_data['body'].velocity.x, planet_data['body'].velocity.y)
        planet_data['history'].append((pos, vel))

        # Store position for trail
        planet_data['trail'].append(
            (planet_data['body'].position.x, planet_data['body'].position.y))

    for photon_data in photons:
        # Store current state for history/training
        pos = (photon_data['body'].position.x, photon_data['body'].position.y)
        vel = (photon_data['body'].velocity.x, photon_data['body'].velocity.y)
        photon_data['history'].append((pos, vel))

        # Store position for trail
        photon_data['trail'].append(
            (photon_data['body'].position.x, photon_data['body'].position.y))


def apply_ai_adjustments(planets, frame_count, predict_planet_position_func):
    """Apply AI predictions to adjust planet velocities"""
    # Occasionally use AI prediction to adjust (blend with physics)
    # Reduced frequency for better performance
    if frame_count % 60 == 0:  # Every 1 second instead of 0.5s
        for planet_data in planets:
            if len(planet_data['history']) > 15:  # Need sufficient history
                predicted_pos = predict_planet_position_func(planet_data)
                if predicted_pos is not None:
                    # Blend prediction with actual position (70% physics, 30% AI)
                    blend_factor = 0.3
                    current_pos = np.array(
                        [planet_data['body'].position.x, planet_data['body'].position.y])
                    predicted_pos_np = np.array(predicted_pos)
                    blended_pos = current_pos * \
                        (1 - blend_factor) + predicted_pos_np * blend_factor
                    # Small adjustment toward prediction
                    adjustment = (blended_pos - current_pos) * 0.1
                    planet_data['body'].velocity += (
                        adjustment[0] / planet_data['body'].mass,
                        adjustment[1] / planet_data['body'].mass)


def limit_velocities(photons, max_velocity):
    """Limit velocity to prevent numerical explosion"""
    for photon_data in photons:
        # Limit velocity to prevent numerical explosion
        speed = math.sqrt(
            photon_data['body'].velocity.x**2 +
            photon_data['body'].velocity.y**2)
        if speed > max_velocity:
            scale = max_velocity / speed
            photon_data['body'].velocity = (
                photon_data['body'].velocity.x * scale,
                photon_data['body'].velocity.y * scale)


def check_black_hole_capture(photons, center_body, center_mass, black_hole_threshold,
                             speed_of_light, g_constant=G_CONSTANT):
    """Check for black hole formation and adjust photon velocities accordingly"""
    if center_mass > black_hole_threshold:
        # Calculate Schwarzschild radius (simplified)
        schwarzschild_radius = 2 * g_constant * \
            center_mass / (speed_of_light ** 2)

        for photon_data in photons:
            # If photon is within schwarzschild radius, it's trapped (simplified)
            dx = photon_data['body'].position.x - center_body.position.x
            dy = photon_data['body'].position.y - center_body.position.y
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < schwarzschild_radius:
                # Photon is trapped - reduce velocity significantly (simulating capture)
                photon_data['body'].velocity *= 0.99


def update_physics_space(space, time_step):
    """Step the physics simulation forward"""
    space.step(time_step)
