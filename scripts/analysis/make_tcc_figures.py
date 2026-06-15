"""
Generate the figures used in the TCC (stage 2) from the real project data.

Outputs PNGs into tcc/ibagens/ (Portuguese labels, host matplotlib, no GPU):
  - fig_dataset_noise.png   : clips per curated noise type (train vs val)
  - fig_training_curve.png  : Park training/validation loss per epoch
  - fig_results_metrics.png : Delta-SDR / PESQ / STOI per input SNR (Park vs baselines)
  - fig_spectrograms.png    : clean/noisy/specsub/wiener/park log-magnitude spectrograms
  - fig_waveforms.png       : noisy/park/clean waveforms of one example

Reads:
  dataset/dataset_manifest.json
  models/runs/park_rced_park_data/training.csv
  models/eval_compare_park/metrics.json (+ audio/)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import soundfile as sf           # noqa: E402
import librosa                   # noqa: E402
import librosa.display           # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import Config, compute_stft  # noqa: E402

OUT_DIR = REPO_ROOT / "tcc" / "ibagens"
MANIFEST = REPO_ROOT / "dataset" / "dataset_manifest.json"
TRAIN_CSV = REPO_ROOT / "models" / "runs" / "park_rced_park_data" / "training.csv"
EVAL_DIR = REPO_ROOT / "models" / "eval_compare_park"

# Consistent display names / colours for the four "methods".
METHOD_LABEL = {
    "noisy": "Ruidoso",
    "specsub": "Subtração espectral",
    "wiener": "Filtro de Wiener",
    "park": "Park R-CED",
}
METHOD_COLOR = {
    "noisy": "#9e9e9e",
    "specsub": "#4c8bf5",
    "wiener": "#f5a14c",
    "park": "#2ca02c",
}
SNR_ORDER = ["0", "5", "10", "15"]

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})


def fig_dataset_noise() -> None:
    manifest = json.loads(MANIFEST.read_text())
    train = Counter(e["noise_source"].split("/")[0] for e in manifest["splits"]["train"])
    val = Counter(e["noise_source"].split("/")[0] for e in manifest["splits"]["val"])
    types = sorted(set(train) | set(val))
    x = np.arange(len(types))
    tr = [train[t] for t in types]
    va = [val[t] for t in types]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x, tr, label="Treino", color="#2ca02c")
    ax.bar(x, va, bottom=tr, label="Validação", color="#1f4e79")
    for xi, (a, b) in enumerate(zip(tr, va)):
        ax.text(xi, a + b + 8, str(a + b), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=20, ha="right")
    ax.set_ylabel("Número de clipes (10 s)")
    ax.set_xlabel("Tipo de ruído (MS-SNSD)")
    ax.legend()
    ax.set_axisbelow(True)
    fig.savefig(OUT_DIR / "fig_dataset_noise.png")
    plt.close(fig)
    print("wrote fig_dataset_noise.png")


def fig_training_curve() -> None:
    data = np.genfromtxt(TRAIN_CSV, delimiter=",", names=True)
    ep = data["epoch"]
    loss = data["loss"]
    val_loss = data["val_loss"]
    best = int(np.argmin(val_loss))

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(ep, loss, label="Treino (MSE)", color="#2ca02c", lw=1.8)
    ax.plot(ep, val_loss, label="Validação (MSE)", color="#1f4e79", lw=1.8)
    ax.axvline(best, color="#d62728", ls="--", lw=1.2,
               label=f"Melhor época ({best})")
    ax.set_yscale("log")
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda MSE (log-magnitude)")
    ax.legend()
    fig.savefig(OUT_DIR / "fig_training_curve.png")
    plt.close(fig)
    print("wrote fig_training_curve.png")


def _grouped_bars(ax, methods, snr_vals, value_fn, ylabel):
    n = len(methods)
    width = 0.8 / n
    x = np.arange(len(SNR_ORDER))
    for i, m in enumerate(methods):
        vals = [value_fn(s, m) for s in SNR_ORDER]
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width,
               label=METHOD_LABEL[m], color=METHOD_COLOR[m])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} dB" for s in SNR_ORDER])
    ax.set_xlabel("SNR de entrada")
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)


def fig_results_metrics() -> None:
    metrics = json.loads((EVAL_DIR / "metrics.json").read_text())
    per_snr = metrics["per_snr"]

    def get(snr, method, key):
        return per_snr[snr][method][key]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    # (a) Delta-SDR -- only the methods that process (noisy delta is 0 by def.)
    _grouped_bars(axes[0], ["specsub", "wiener", "park"], None,
                  lambda s, m: get(s, m, "delta_sdr"), "ΔSDR (dB)")
    axes[0].set_title("(a) Melhoria de SDR")
    # (b) PESQ -- include noisy reference
    _grouped_bars(axes[1], ["noisy", "specsub", "wiener", "park"], None,
                  lambda s, m: get(s, m, "pesq"), "PESQ")
    axes[1].set_title("(b) PESQ")
    # (c) STOI -- include noisy reference
    _grouped_bars(axes[2], ["noisy", "specsub", "wiener", "park"], None,
                  lambda s, m: get(s, m, "stoi"), "STOI")
    axes[2].set_title("(c) STOI")

    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_results_metrics.png")
    plt.close(fig)
    print("wrote fig_results_metrics.png")


def _pick_example() -> dict:
    metrics = json.loads((EVAL_DIR / "metrics.json").read_text())
    examples = metrics.get("audio_examples", [])
    if not examples:
        raise SystemExit("no audio_examples in metrics.json")
    # prefer the lowest-SNR example (most visually informative)
    return sorted(examples, key=lambda e: e["snr"])[0]


def _load(base: str, kind: str) -> np.ndarray:
    a, _ = sf.read(str(EVAL_DIR / "audio" / f"{base}_{kind}.wav"), dtype="float32")
    return a


def fig_spectrograms() -> None:
    ex = _pick_example()
    base, snr = ex["base"], ex["snr"]
    order = [("clean", "Limpo (referência)"), ("noisy", f"Ruidoso ({snr:.0f} dB)"),
             ("park", "Park R-CED"), ("specsub", "Subtração espectral"),
             ("wiener", "Filtro de Wiener")]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.2))
    axes = axes.ravel()
    img = None
    for ax, (kind, title) in zip(axes, order):
        audio = _load(base, kind)
        S_db = librosa.amplitude_to_db(np.abs(compute_stft(audio)), ref=np.max)
        img = librosa.display.specshow(
            S_db, sr=Config.SAMPLE_RATE, hop_length=Config.HOP_LENGTH,
            x_axis="time", y_axis="hz", ax=ax, vmin=-80, vmax=0, cmap="magma")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Frequência (Hz)")
    # last cell: colorbar
    axes[-1].axis("off")
    cbar = fig.colorbar(img, ax=axes[-1], fraction=0.6, pad=0.05)
    cbar.set_label("Magnitude (dB)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_spectrograms.png")
    plt.close(fig)
    print(f"wrote fig_spectrograms.png (example {base})")


def fig_waveforms() -> None:
    ex = _pick_example()
    base, snr = ex["base"], ex["snr"]
    rows = [("noisy", f"Ruidoso ({snr:.0f} dB)", "#9e9e9e"),
            ("park", "Park R-CED", "#2ca02c"),
            ("clean", "Limpo (referência)", "#1f4e79")]
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 4.8), sharex=True, sharey=True)
    for ax, (kind, title, color) in zip(axes, rows):
        a = _load(base, kind)
        t = np.arange(len(a)) / Config.SAMPLE_RATE
        ax.plot(t, a, color=color, lw=0.6)
        ax.set_ylabel("Amplitude")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylim(-1.05, 1.05)
    axes[-1].set_xlabel("Tempo (s)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_waveforms.png")
    plt.close(fig)
    print("wrote fig_waveforms.png")


def fig_architecture() -> None:
    """Schematic of the 2D R-CED filter pyramid with mirror skip connections."""
    from matplotlib.patches import Rectangle, FancyArrowPatch
    filters = [12, 16, 20, 24, 32, 32, 32, 32, 24, 24, 20, 20, 16, 16, 12, 12]
    n = len(filters)
    maxf = max(filters)

    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    for i, f in enumerate(filters):
        h = f / maxf
        color = "#2ca02c" if i < n // 2 else "#1f4e79"
        ax.add_patch(Rectangle((i - 0.35, -h / 2), 0.7, h, color=color, alpha=0.85))
        ax.text(i, -h / 2 - 0.05, str(f), ha="center", va="top", fontsize=7)
    # mirror skip connections: decoder layer i <- encoder layer n-1-i
    for i in range(n // 2 + 1, n):
        j = n - 1 - i
        arc = FancyArrowPatch((j, filters[j] / maxf / 2), (i, filters[i] / maxf / 2),
                              connectionstyle="arc3,rad=-0.45", arrowstyle="-|>",
                              color="#d62728", lw=1.0, alpha=0.6, mutation_scale=9)
        ax.add_patch(arc)
    # input / output blocks
    ax.text(-1.7, 0, "Espectrograma\nruidoso\n(257×251×1)", ha="center", va="center", fontsize=8)
    ax.text(n + 0.7, 0, "Espectrograma\nestimado\n(257×251×1)", ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(-0.45, 0), xytext=(-0.95, 0), arrowprops=dict(arrowstyle="-|>", color="k"))
    ax.annotate("", xy=(n - 0.1, 0), xytext=(n - 0.55, 0), arrowprops=dict(arrowstyle="-|>", color="k"))

    handles = [Rectangle((0, 0), 1, 1, color="#2ca02c", alpha=0.85),
               Rectangle((0, 0), 1, 1, color="#1f4e79", alpha=0.85),
               FancyArrowPatch((0, 0), (1, 0), color="#d62728")]
    ax.legend(handles, ["Codificador (8 camadas)", "Decodificador (8 camadas)",
                        "Conexão de salto (skip)"], loc="lower center",
              ncol=3, bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.text(n / 2 - 0.5, 0.62, "16 camadas Conv2D (padding \"same\") + BatchNorm + ReLU",
            ha="center", fontsize=9)
    ax.set_xlim(-2.7, n + 1.7)
    ax.set_ylim(-0.8, 0.75)
    ax.axis("off")
    fig.savefig(OUT_DIR / "fig_architecture.png")
    plt.close(fig)
    print("wrote fig_architecture.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_architecture()
    fig_dataset_noise()
    fig_training_curve()
    fig_results_metrics()
    fig_spectrograms()
    fig_waveforms()
    print(f"\nAll figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
