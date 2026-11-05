"""Render saved simulation states to a video without resimulating physics."""

from pathlib import Path
import hashlib
import json
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from graspforge.contacts import MODEL_PATH, load_model


def frame_indices(times, fps):
    """Nearest prior recorded sample for uniformly spaced video timestamps."""
    times = np.asarray(times, dtype=float)
    if not isinstance(fps, int) or not 1 <= fps <= 120:
        raise ValueError("fps must be an integer between 1 and 120")
    if (
        times.ndim != 1
        or len(times) == 0
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("episode times must be finite and strictly increasing")
    target = times[0] + np.arange(int(np.floor((times[-1] - times[0]) * fps)) + 1) / fps
    return np.clip(np.searchsorted(times, target, side="right") - 1, 0, len(times) - 1)


def render_video(
    episode_path, output_path, *, fps=30, camera="overview", width=640, height=480
):
    """Replay qpos/qvel for visual inspection; does not simulate or modify the log."""
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            'Install video support with: python -m pip install -e ".[video]"'
        ) from exc
    episode_path, output_path = Path(episode_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"video already exists: {output_path}")
    if (
        output_path.suffix.lower() != ".mp4"
        or width <= 0
        or height <= 0
        or width % 2
        or height % 2
    ):
        raise ValueError(
            "output must be .mp4 and dimensions must be positive even numbers"
        )
    manifest_path = episode_path.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected = manifest.get("model", {}).get("sha256")
        if expected and expected != hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest():
            raise ValueError("episode model hash differs from the installed model")
    with np.load(episode_path, allow_pickle=False) as loaded:
        log = {key: loaded[key] for key in loaded.files}
    model = load_model()
    data = mujoco.MjData(model)
    indices = frame_indices(log["t"], fps)
    count = len(log["t"])
    if log["qpos"].shape != (count, model.nq) or log["qvel"].shape != (count, model.nv):
        raise ValueError("episode state dimensions do not match model")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
    except Exception as exc:
        raise RuntimeError(
            "MuJoCo could not create a graphics context. Run this command in a "
            "desktop terminal or configure a supported headless OpenGL backend."
        ) from exc
    proc = None
    try:
        proc = subprocess.Popen(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
        )
        for k in indices:
            data.qpos[:] = log["qpos"][k]
            data.qvel[:] = log["qvel"][k]
            data.time = float(log["t"][k])
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame, "RGBA")
            draw.rectangle((0, 0, width, 60), fill=(0, 0, 0, 175))
            phase = str(log["phase_names"][int(log["phase"][k])])
            force = f"L {log['touch_left'][k]:.1f} N  R {log['touch_right'][k]:.1f} N"
            slip = log["slip"][k]
            slip_text = f"{slip * 1000:.1f} mm" if np.isfinite(slip) else "--"
            push = "  PUSH" if log["dist_active"][k] else ""
            draw.text((8, 6), f"t {log['t'][k]:.2f} s   {phase}{push}", fill="white")
            draw.text((8, 28), f"{force}   slip {slip_text}", fill="white")
            proc.stdin.write(np.asarray(frame).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while writing video")
    except BaseException:
        if proc is not None:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            proc.wait()
        output_path.unlink(missing_ok=True)
        raise
    finally:
        renderer.close()
    return output_path
