import time
import fastplotlib as fpl
import cupy as cp
from fastplotlib.ui import EdgeWindow
from imgui_bundle import imgui
from typing import Callable, Dict


# Waveforms. Their formula, their offset, and their colour
WaveFunc = Callable[[cp.ndarray, float, float], cp.ndarray]

defaults = {
    "freq": 5.0,
    "amplitude": 5.0,
    "speed_hz": 0.5,
    "n_points": 10_000,
}

WAVEFORMS: Dict[str, dict] = {
    "Sine": {
        "func": lambda t, f, a: a * cp.sin(2 * cp.pi * f * t),
        "offset": 0.0,
        "color": "blue",
        "enabled": True,
        "freq": defaults["freq"],
        "amplitude": defaults["amplitude"],
        "speed_hz": defaults["speed_hz"],
    },
    "Square": {
        "func": lambda t, f, a: a * cp.sign(cp.sin(2 * cp.pi * f * t)),
        "offset": 20.0,
        "color": "red",
        "enabled": True,
        "freq": defaults["freq"],
        "amplitude": defaults["amplitude"],
        "speed_hz": defaults["speed_hz"],
    },
    "Triangle": {
        "func": lambda t, f, a: a * 2 * cp.arcsin(cp.sin(2 * cp.pi * f * t)) / cp.pi,
        "offset": 40.0,
        "color": "green",
        "enabled": True,
        "freq": defaults["freq"],
        "amplitude": defaults["amplitude"],
        "speed_hz": defaults["speed_hz"],
    },
    "Sawtooth": {
        "func": lambda t, f, a: a * 2 * (t * f - cp.floor(0.5 + t * f)),
        "offset": 60.0,
        "color": "orange",
        "enabled": True,
        "freq": defaults["freq"],
        "amplitude": defaults["amplitude"],
        "speed_hz": defaults["speed_hz"],
    },
}

waveform_defaults = {
    name: {
        "freq": wf["freq"],
        "amplitude": wf["amplitude"],
        "offset": wf["offset"],
        "speed_hz": wf["speed_hz"],
    }
    for name, wf in WAVEFORMS.items()
}

# Create evenly spaced points between 0 and 1, as a float32 CuPy array
t = cp.linspace(0, 1, defaults["n_points"], dtype=cp.float32)

# Per-waveform phase offsets
_phases: Dict[str, float] = {name: 0.0 for name in WAVEFORMS.keys()}

# Guard to avoid updating while rebuilding graphics
_is_rebuilding = False

_current_n_points = defaults["n_points"]
_last_time = None  # monotonic timestamp for dt


# Delete all existing lines from subplot
def delete_all_lines(subplot, lines_dict):
    for line in list(lines_dict.values()):
        try:
            subplot.delete_graphic(line)
        except Exception as e:
            print(f"Failed to delete line: {e}")
    lines_dict.clear()


# Generate waveforms + their y offsets
def create_waveforms(phases: Dict[str, float]):
    return {
        name: wf["func"](t + phases.get(name, 0.0), wf["freq"], wf["amplitude"]) + wf["offset"]
        for name, wf in WAVEFORMS.items()
    }


# Create figure
fig = fpl.Figure(size=(900, 600))
fig.canvas.set_title("Waveform Visualiser")
subplot = fig[0, 0]
subplot.title = "Waveform Animation"
subplot.axes.visible = False  # hide axes


# Persistent contiguous Y buffers for fast updates
_line_y_buffers: Dict[str, cp.ndarray] = {}

# Dict of all existing lines
lines = {}


# Create xy positions from x and y cupy arrays, return as numpy array
def _positions_xy_numpy(x_cp: cp.ndarray, y_cp: cp.ndarray):
    return cp.stack([x_cp, y_cp], axis=1).get()


# Creates lines, or recreates them if they already exist
def _create_or_recreate_lines():
    global lines
    waves_local = create_waveforms(_phases)

    delete_all_lines(subplot, lines)
    _line_y_buffers.clear()

    for name, y in waves_local.items():
        data = _positions_xy_numpy(t, y)
        line = subplot.add_line(
            data=data,
            name=name,
            colors=WAVEFORMS[name]["color"],
        )

        line.visible = WAVEFORMS[name]["enabled"]

        lines[name] = line
        _line_y_buffers[name] = line.data[:, 1].copy()


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
    global _last_time

    if _is_rebuilding:
        return

    now = time.monotonic()
    if _last_time is None:
        _last_time = now
        return
    dt = now - _last_time
    _last_time = now

    for name, line in list(lines.items()):
        wf = WAVEFORMS.get(name)

        if not wf["enabled"]:
            continue

        # Advance phase by speed (Hz) * elapsed seconds
        _phases[name] += dt * wf["speed_hz"]

        y_shifted = wf["func"](t + _phases[name], wf["freq"], wf["amplitude"]) + wf["offset"]

        try:
            y_buf = _line_y_buffers[name]
            y_shifted.get(out=y_buf)  # contiguous GPU → CPU copy
            line.data[:, 1] = y_buf  # fast NumPy memcpy
        except Exception as e:
            print(f"Line update failed ({name}): {e}")


# Add animation to subplot
subplot.add_animations(update)


# Create UI
class WaveformUI(EdgeWindow):
    def __init__(self, figure, size=320, location="right", title="Waveform Controls"):
        super().__init__(figure=figure, size=size, location=location, title=title)
        self._points = int(defaults["n_points"])
        self._pending_points = None

    def update(self):
        global _phases, _last_time

        imgui.separator()
        imgui.text("Waveforms")

        if imgui.begin_tab_bar("WaveformTabs"):
            for name, wf in WAVEFORMS.items():
                opened, _ = imgui.begin_tab_item(name)
                if opened:
                    changed, enabled = imgui.checkbox(f"Enabled##{name}", wf["enabled"])
                    if changed:
                        wf["enabled"] = enabled
                        lines[name].visible = enabled

                    # Amplitude
                    changed_slider, new_amp = imgui.slider_float(
                        f"Amplitude##{name}", float(wf["amplitude"]), 0.0, 10.0
                    )
                    changed_text, text_amp = imgui.input_float(
                        f"##AmplitudeInput{name}", float(wf["amplitude"]), step=0.5, format="%.2f"
                    )
                    if changed_slider:
                        wf["amplitude"] = float(new_amp)
                    if changed_text:
                        wf["amplitude"] = float(text_amp)

                    # Frequency
                    changed_slider_f, new_freq = imgui.slider_float(
                        f"Frequency##{name}", float(wf["freq"]), 1.0, 100.0
                    )
                    changed_text_f, text_freq = imgui.input_float(
                        f"##FrequencyInput{name}", float(wf["freq"]), step=5.0, format="%.2f"
                    )
                    if changed_slider_f:
                        wf["freq"] = float(new_freq)
                    if changed_text_f:
                        wf["freq"] = float(text_freq)

                    # Offset
                    changed_offset, new_offset = imgui.slider_float(
                        f"Offset##{name}", float(wf["offset"]), -100.0, 100.0
                    )
                    changed_offset_text, text_offset = imgui.input_float(
                        f"##OffsetInput{name}", float(wf["offset"]), step=5, format="%.2f"
                    )
                    if changed_offset:
                        wf["offset"] = float(new_offset)
                    if changed_offset_text:
                        wf["offset"] = float(text_offset)

                    # Speed (Hz)
                    changed_speed, new_speed = imgui.slider_float(
                        f"Speed (Hz)##{name}", float(wf["speed_hz"]), -2.0, 2.0
                    )
                    changed_speed_text, text_speed = imgui.input_float(
                        f"##SpeedInput{name}", float(wf["speed_hz"]), step=0.5, format="%.2f"
                    )
                    if changed_speed:
                        wf["speed_hz"] = float(new_speed)
                    if changed_speed_text:
                        wf["speed_hz"] = float(text_speed)

                    # Per-waveform reset
                    if imgui.button(f"Reset {name}"):
                        wf["freq"] = waveform_defaults[name]["freq"]
                        wf["amplitude"] = waveform_defaults[name]["amplitude"]
                        wf["offset"] = waveform_defaults[name]["offset"]
                        wf["speed_hz"] = waveform_defaults[name]["speed_hz"]
                        wf["enabled"] = True
                        lines[name].visible = True
                        _phases[name] = 0.0
                        _last_time = None  # resync timing after reset

                    imgui.end_tab_item()
            imgui.end_tab_bar()

        imgui.separator()
        imgui.text("Global")

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

        # Global reset
        if imgui.button("Reset All"):
            self._points = defaults["n_points"]
            for name, wf in WAVEFORMS.items():
                wf["freq"] = waveform_defaults[name]["freq"]
                wf["amplitude"] = waveform_defaults[name]["amplitude"]
                wf["offset"] = waveform_defaults[name]["offset"]
                wf["speed_hz"] = waveform_defaults[name]["speed_hz"]
                wf["enabled"] = True
                lines[name].visible = True
                _phases[name] = 0.0
            _last_time = None  # resync timing
            _rebuild_with_points(self._points)


gui = WaveformUI(fig)
fig.add_gui(gui)

# Show figure
fig.show(maintain_aspect=False)
fpl.loop.run()