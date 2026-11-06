"""Plot the audited allocator CSV using the separate report Python environment."""

import argparse
import csv
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path)
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError(options.output)
    if options.dependencies:
        sys.path.insert(0, str(options.dependencies))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audit = json.loads((options.analysis / "audit.json").read_text(encoding="utf-8"))
    with (options.analysis / "allocator_timeline.csv").open(encoding="utf-8", newline="") as handle:
        points = list(csv.DictReader(handle))
    peak = audit["peak"]
    before, after = peak["before_checkpoint"], peak["after_checkpoint"]
    x = [int(p["event"]) for p in points]
    y = [int(p["allocated_bytes"]) / 1024**2 for p in points]
    fig, (ax, detail) = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    ax.plot(x, y, color="#087f8c", linewidth=0.8, label="Allocated blocks")
    ax.plot(x, [int(p["reserved_bytes"]) / 1024**2 for p in points],
            color="#996300", linewidth=0.8, label="Reserved segments")
    ax.scatter([peak["event"]], [peak["allocated_bytes"] / 1024**2], color="#c43e4d", zorder=5)
    ax.annotate(
        f'Peak: {peak["allocated_bytes"] / 1024**2:.2f} MiB\nEvent {peak["event"]}, decoder call {before["decoder_call"]}',
        (peak["event"], peak["allocated_bytes"] / 1024**2),
        xytext=(-230, -25), textcoords="offset points", arrowprops={"arrowstyle": "->"},
    )
    ax.set(xlabel="Allocator event index (not time)", ylabel="MiB",
           title="Fixed P01/s42 VAE decode: allocator history")
    ax.legend(loc="upper left")
    start, stop = before["first_matching_event"], after["last_matching_event"]
    detail.plot(x[start:stop + 1], y[start:stop + 1], color="#087f8c", linewidth=1)
    detail.axvline(peak["event"], color="#c43e4d", linewidth=0.8)
    detail.set(xlabel="Allocator event index (not time)", ylabel="Allocated MiB",
               title=f'Decoder call {before["decoder_call"]}: peak within convolution')
    fig.savefig(options.output, dpi=160)
    plt.close(fig)
    print(options.output)


if __name__ == "__main__":
    main()
