# StarDance

StarDance is an interactive orbital physics sandbox built with Pygame and Pymunk.

## Run

```powershell
python main.py
```

## Controls

- Left click empty space: add a planet
- Right click empty space: add a photon
- Click a planet or photon: select it
- Delete: remove selected body
- Backspace: remove the latest planet
- Space: pause or resume
- Mouse wheel or `I` / `O`: zoom
- Arrow keys: rotate the camera
- `WASD`: pan the camera
- `+` / `-`: adjust gravity
- `M` / `N`: adjust selected planet mass
- `R` / `F`: adjust selected planet visual size
- `C`: clear the scene
- `Home`: reset the camera
- `B`: enable black hole mode
- `H`: hide or show the UI

The simulator starts with an empty scene. Optional sample bodies can be enabled in `config.py`.

## Crash Handling

Recoverable runtime errors are logged to `runtime_errors.log`, shown in the UI, and pause the simulation so the window stays open for inspection.
