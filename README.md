# GravSim

GravSim is an interactive orbital physics sandbox built with Pygame and Pymunk.

## Quick Start (very simple)

1. **Make sure you have Python 3.7+ installed**  
   Download from https://www.python.org/downloads/

2. **Open a terminal (Command Prompt / PowerShell / Bash)**  
   Navigate to where you want to put the project.

3. **Download the project**  
   ```powershell
   git clone https://github.com/Optifrost/GravSim.git
   cd GravSim
   ```

4. **Install the required Python packages**  
   ```powershell
   pip install pygame pymunk
   ```

5. **Run the simulation**  
   ```powershell
   python main.py
   ```

The window will open showing a dark space background.  
- Left‑click empty space → add a planet  
- Right‑click empty space → add a photon  
- Press **B** to toggle black‑hole mode  
- Press **Space** to pause/un‑pause  
- Scroll mouse wheel or press **I/O** to zoom  

## Controls (summary)

| Key / Mouse | Action |
|-------------|--------|
| Left click | Add planet |
| Right click | Add photon |
| Click a body | Select it |
| Delete | Remove selected body |
| Backspace | Remove the most recent planet |
| Space | Pause / resume |
| Mouse wheel / I / O | Zoom |
| Arrow keys | Rotate camera |
| WASD | Pan camera |
| + / - | Increase / decrease gravity |
| M / N | Increase / decrease selected planet mass |
| R / F | Increase / decrease selected planet visual size |
| C | Clear the scene |
| Home | Reset camera view |
| B | Enable / disable black‑hole mode |
| H | Hide / show the UI |

## Crash Handling

Recoverable runtime errors are logged to `runtime_errors.log`, shown in the UI, and pause the simulation so the window stays open for inspection.

## Configuration

Optional sample bodies can be enabled in `config.py` by setting `START_WITH_SAMPLE_PLANETS = True` or `START_WITH_SAMPLE_PHOTONS = True`.

Enjoy experimenting with orbits!