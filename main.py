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
                                # No entity clicked → create a new planet at cursor location.
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