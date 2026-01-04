import fastplotlib as fpl
import cupy as cp
from fastplotlib.ui import EdgeWindow
from imgui_bundle import imgui
from typing import Callable, Dict


# Waveforms. Their formula, their offset, and their colour
WaveFunc = Callable[[cp.ndarray, float, float], cp.ndarray]

WAVEFORMS: Dict[str, dict] = {
    "Sine": {
        "func": lambda t, f, a: a * cp.sin(2 * cp.pi * f * t),
        "offset": 0.0,
        "color": "blue",
    },
    "Square": {
        "func": lambda t, f, a: a * cp.sign(cp.sin(2 * cp.pi * f * t)),
        "offset": 20.0,
        "color": "red",
    },
    "Triangle": {
        "func": lambda t, f, a: a * 2 * cp.arcsin(cp.sin(2 * cp.pi * f * t)) / cp.pi,
        "offset": 40.0,
        "color": "green",
    },
    "Sawtooth": {
        "func": lambda t, f, a: a * 2 * (t * f - cp.floor(0.5 + t * f)),
        "offset": 60.0,
        "color": "orange",
    },
}

defaults = {
    "freq": 5.0,
    "amplitude": 5.0,
    "speed": 5.0,
    "n_points": 10_000,
}

# Create evenly spaced points between 0 and 1, as a float32 CuPy array
t = cp.linspace(0, 1, defaults["n_points"], dtype=cp.float32)

freq = defaults["freq"]
amplitude = defaults["amplitude"]
speed = defaults["speed"]  # speed modifier

# Phase offset for animation (decouples speed from point count)
_phase = 0.0

# Guard to avoid updating while rebuilding graphics
_is_rebuilding = False

_current_n_points = defaults["n_points"]


# Delete all existing lines from subplot
def delete_all_lines(subplot, lines_dict):
    for line in list(lines_dict.values()):
        try:
            subplot.delete_graphic(line)
        except Exception as e:
            print(f"Failed to delete line: {e}")
    lines_dict.clear()


# Generate waveforms + their y offsets
def create_waveforms(freq_: float, amplitude_: float, phase_: float):
    t_eval = t + phase_
    return {
        name: wf["func"](t_eval, freq_, amplitude_) + wf["offset"]
        for name, wf in WAVEFORMS.items()
    }


# Create figure
fig = fpl.Figure(size=(900, 600))
fig.canvas.set_title("Waveform Visualiser")
subplot = fig[0, 0]
subplot.title = "Waveform Animation"
subplot.axes.visible = False  # hide axes

# Dict of all existing lines
lines = {}


# Create xy positions from x and y cupy arrays, return as numpy array
def _positions_xy_numpy(x_cp: cp.ndarray, y_cp: cp.ndarray):
    return cp.stack([x_cp, y_cp], axis=1).get()


# Creates lines, or recreates them if they already exist
def _create_or_recreate_lines():
    global lines
    waves_local = create_waveforms(freq, amplitude, _phase)

    delete_all_lines(subplot, lines)

    for name, y in waves_local.items():
        data = _positions_xy_numpy(t, y)
        lines[name] = subplot.add_line(
            data=data,
            name=name,
            colors=WAVEFORMS[name]["color"],
        )


# Rebuild lines with a different number of points
def _rebuild_with_points(new_n_points: int):
    global t, _is_rebuilding, _current_n_points

    new_n_points = int(new_n_points)
    new_n_points = max(2, new_n_points)

    if new_n_points == _current_n_points:
        return

    _is_rebuilding = True
    try:
        _current_n_points = new_n_points
        t = cp.linspace(0, 1, new_n_points, dtype=cp.float32)
        _create_or_recreate_lines()
    finally:
        _is_rebuilding = False


_create_or_recreate_lines()


# Update animation
def update(_subplot):
    global freq, amplitude, speed, _phase

    if _is_rebuilding:
        return

    # Speed is now independent of point count
    increment = 0.001 * float(speed)
    _phase += increment

    for name, line in list(lines.items()):
        wf = WAVEFORMS.get(name)
        if wf is None:
            continue

        y_shifted = wf["func"](t + _phase, freq, amplitude) + wf["offset"]

        try:
            line.data[:, 1] = y_shifted.get()
        except Exception as e:
            print(f"Line update failed ({name}): {e}")


# Add animation to subplot
subplot.add_animations(update)


# Create UI
class WaveformUI(EdgeWindow):
    def __init__(self, figure, size=300, location="right", title="Waveform Controls"):
        super().__init__(figure=figure, size=size, location=location, title=title)
        self._amplitude = float(defaults["amplitude"])
        self._freq = float(defaults["freq"])
        self._speed = float(defaults["speed"])
        self._points = int(defaults["n_points"])
        self._pending_points = None

    def update(self):
        global amplitude, freq, speed, _phase

        # Amplitude
        changed_slider, new_amp = imgui.slider_float("Amplitude", float(self._amplitude), 0.0, 10.0)
        changed_text, text_amp = imgui.input_float("##Amplitude", float(self._amplitude), step=0.5, format="%.2f")
        if changed_slider:
            self._amplitude = float(new_amp)
        if changed_text:
            self._amplitude = float(text_amp)
        amplitude = self._amplitude

        # Frequency
        changed_slider, new_freq = imgui.slider_float("Frequency", float(self._freq), 1.0, 100.0)
        changed_text, text_freq = imgui.input_float("##Frequency", float(self._freq), step=5.0, format="%.2f")
        if changed_slider:
            self._freq = float(new_freq)
        if changed_text:
            self._freq = float(text_freq)
        freq = self._freq

        # Speed
        changed_slider, new_speed = imgui.slider_float("Speed", float(self._speed), 0.0, 50.0)
        changed_text, text_speed = imgui.input_float("##Speed", float(self._speed), step=0.5, format="%.2f")
        if changed_slider:
            self._speed = float(new_speed)
        if changed_text:
            self._speed = float(text_speed)
        speed = self._speed

        # Points
        changed_p_slider, new_points = imgui.slider_int("Points", int(self._points), 128, 200_000)
        changed_p_text, text_points = imgui.input_int("##Points", int(self._points), step=256)

        if changed_p_slider:
            self._points = int(new_points)
            self._pending_points = self._points
        if changed_p_text:
            self._points = int(text_points)
            self._pending_points = self._points

        if self._pending_points is not None and self._pending_points != _current_n_points:
            _rebuild_with_points(self._pending_points)
            self._pending_points = None

        # Reset
        if imgui.button("Reset"):
            self._amplitude = defaults["amplitude"]
            self._freq = defaults["freq"]
            self._speed = defaults["speed"]
            self._points = defaults["n_points"]

            amplitude = self._amplitude
            freq = self._freq
            speed = self._speed
            _phase = 0.0

            _rebuild_with_points(self._points)


gui = WaveformUI(fig)
fig.add_gui(gui)

# Show figure
fig.show(maintain_aspect=False)
fpl.loop.run()
