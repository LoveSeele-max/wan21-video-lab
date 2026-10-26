"""Build fixed counterexamples; preserve originals and verify lossless pixels."""
import json
from pathlib import Path

import av
import cv2
import numpy as np

from metrics import ROOT, baseline_jobs, read_video, sha256, write_json

# x0, y0, x1, y1; fixed after viewing source frame 35, before metric results.
REGIONS = {
    "P01_s42": (350, 60, 650, 380),
    "P04_s42": (370, 155, 455, 410),
    "P07_s42": (260, 220, 660, 430),
}


def transform(frames, kind, region):
    result = frames.copy()
    if kind == "static":
        result[:] = frames[0]
        return result
    for index in range(30, 41):
        if kind == "flicker":
            gain = 1.15 if index % 2 == 0 else 0.85
            result[index] = np.rint(np.clip(frames[index].astype(np.float32) * gain,
                                           0, 255)).astype(np.uint8)
        elif kind == "blur":
            result[index] = cv2.GaussianBlur(
                frames[index], (19, 19), 3, sigmaY=3, borderType=cv2.BORDER_REFLECT_101)
        elif kind == "local_warp":
            x0, y0, x1, y1 = region
            patch = frames[index, y0:y1, x0:x1]
            y, x = np.mgrid[:y1-y0, :x1-x0].astype(np.float32)
            # Taper displacement at patch edges; alternate phase across frames.
            envelope = np.sin(np.pi*x/(x1-x0-1)) * np.sin(np.pi*y/(y1-y0-1))
            offset = 12 * envelope * np.sin(2*np.pi*y/32 + (index-30)*np.pi/2)
            result[index, y0:y1, x0:x1] = cv2.remap(
                patch, (x + offset).astype(np.float32), y,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        else:
            raise ValueError(kind)
    return result


def write_lossless(path, frames):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=16)
        stream.width, stream.height = 832, 480
        stream.pix_fmt = "bgr0"
        for array in frames:
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    if not np.array_equal(read_video(path), frames):
        raise RuntimeError(f"Lossless pixel roundtrip failed: {path}")


def main():
    outputs = ROOT / "outputs/eval/probes"
    notes = ROOT / "notes/eval/probes"
    if outputs.exists() or notes.exists():
        raise FileExistsError("Probe directories already exist; preserve prior artifacts")
    jobs = {j["id"]: j for j in baseline_jobs()}
    # Validate all source checksums before creating output directories.
    for key in (*REGIONS, "P02_s42"):
        if sha256(jobs[key]["video_path"]) != jobs[key]["video_sha256"]:
            raise ValueError(f"Source hash mismatch: {key}")
    outputs.mkdir(parents=True)
    notes.mkdir(parents=True)
    write_json(notes / "construction.json", {
        "version": 1, "script_sha256": sha256(__file__),
        "protocol_sha256": sha256(ROOT / "notes/week3_metrics.md"),
        "frame_start": 30, "frame_end_inclusive": 40, "frame_index_base": 0,
        "flicker_gains": [1.15, 0.85], "blur_kernel": [19, 19], "blur_sigma": 3,
        "regions_xyxy": REGIONS, "warp_amplitude_pixels": 12,
        "warp_period_pixels": 32, "warp_phase_step": "pi/2",
        "warp_edge_envelope": "sin(pi*x/(width-1))*sin(pi*y/(height-1))",
        "codec": "ffv1", "pixel_format": "bgr0", "container": "matroska",
    })
    with (notes / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for key, region in REGIONS.items():
            source = jobs[key]
            frames = read_video(source["video_path"])
            for kind in ("static", "flicker", "blur", "local_warp"):
                identifier = f"{key}_{kind}"
                path = outputs / f"{identifier}.mkv"
                changed = transform(frames, kind, region)
                write_lossless(path, changed)
                record = {
                    "id": identifier, "kind": kind, "source_key": key,
                    "prompt_id": source["prompt_id"], "prompt": source["prompt"],
                    "video_path": str(path), "video_sha256": sha256(path),
                    "source_sha256": source["video_sha256"],
                    "lossless_pixel_check": True,
                    "changed_frames": [i for i in range(81)
                                       if not np.array_equal(frames[i], changed[i])],
                }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                print(f"{identifier}: saved and pixel-verified", flush=True)
        source = jobs["P02_s42"]
        record = dict(source, id="P02_s42_text_P01", kind="text_mismatch",
                      source_key="P02_s42", prompt_id="P01", prompt=jobs["P01_s42"]["prompt"],
                      requires_human_action_check=True)
        handle.write(json.dumps(record) + "\n")
    print(f"Probe manifest saved: {notes / 'manifest.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
