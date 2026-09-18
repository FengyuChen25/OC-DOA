"""
M16 benchmark: parameter count + inference time.
Models: OBCNN, CNN, HMCViT .
"""
import time, os, sys, gc, numpy as np, torch
from scipy.signal import find_peaks
from simulation_utils import *
from lowcnn_model import *
from hmc_vit_model import cnn_trans
from OB_model import OrthoBasisCNN1D
from OB_utils import estimate_angles_from_coefficients_ESPRIT
import os
import platform


# 然后再 import numpy 和其他库
DEVICE = torch.device("cpu")
torch.set_num_threads(1)  # pin single thread, reduce timing variance
DOA = np.array([-34.64, -8.31])
K, M = len(DOA), 16
wavelength = 0.3
N_snap, SNR = 400, 0
N_TRIALS = 100
dd = (np.arange(M) * wavelength / 2).reshape(-1, 1)
derad = np.pi / 180

# ═══════════════════════════
# Load models
# ═══════════════════════════
obcnn = OrthoBasisCNN1D(input_length=120, output_dim=32).to(DEVICE).eval()
obcnn.load_state_dict(torch.load('obcnn/best_model_obmodel1.pth', map_location=DEVICE))
base_info = np.load(os.path.join("orthobasis_data_M16", 'base_info.npy'), allow_pickle=True).item()

lowsnr = lowsnr_cnn().to(DEVICE).eval()
lowsnr.load_state_dict(torch.load('lowsnrcnn/net.pth', map_location=DEVICE))

hmcvit = cnn_trans(in_channels=3, patch_size_x=1, patch_size_y=6,
                   emb_size=128, img_size=120, depth=4, num_cls_tokens=8).to(DEVICE).eval()
ckpt = torch.load('best.pth', map_location=DEVICE, weights_only=True)
hmcvit.load_state_dict(ckpt.get('model_state', ckpt))
hmcvit_grid = np.linspace(-60, 60, 121)

# ═══════════════════════════
# Parameter count
# ═══════════════════════════
# ═══════════════════════════
# Trainable parameter count
# ═══════════════════════════
def count_trainable_params(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

param_data = [
    ('OC-DOA', count_trainable_params(obcnn)),
    ('CNN', count_trainable_params(lowsnr)),
    ('HMC-ViT', count_trainable_params(hmcvit)),
]

print(f"\n{'='*55}")
print(f"  Trainable Parameter Count")
print(f"{'='*55}")
for name, p in param_data:
    print(f"  {name:12s} {p:>12,d}")
print(f"{'='*55}")

# ═══════════════════════════
# Prepare data
# ═══════════════════════════
Rx = generate_unified_Rx(DOA, M, N_snap, SNR)

data_low = extract_lowsnrcnn_features(Rx, M)
data_t_low = torch.tensor(np.array(data_low['input']), dtype=torch.float32).to(DEVICE)

data_ob = extract_obcnn_features(Rx, M)
data_t_ob = torch.from_numpy(np.array(data_ob['input'])).float().reshape(-1, 2, 1, 120).to(DEVICE)

# HMC-ViT: trace norm + 上三角 3 通道 (和仿真一致)
tr = np.trace(np.abs(Rx)) + 1e-12
Rx_n = Rx * (M / tr)
hmc_in = np.stack([Rx_n.real, Rx_n.imag, np.arctan2(Rx_n.imag, Rx_n.real)], axis=0)
idx_ut = np.triu_indices(M, k=1)
hmc_in = hmc_in[:, idx_ut[0], idx_ut[1]]
data_t_hmc = torch.from_numpy(hmc_in).float().unsqueeze(0).unsqueeze(2).to(DEVICE)

ev, I = np.linalg.eig(Rx)
si = np.argsort(ev)  # 升序
U_mat = np.asmatrix(I[:, si[:-K]])  # 噪声子空间 = M-K 个最小特征向量

results = {}

def cooldown(sec=0.5):
    """每段计时之间释放内存+短暂休眠, 避免热缓存遗留影响下一段计时"""
    gc.collect()
    time.sleep(sec)

# ═══════════════════════════
# OBCNN
# ═══════════════════════════
t0 = time.time()
for _ in range(N_TRIALS):
    with torch.no_grad():
        coeff = obcnn(data_t_ob).cpu().numpy().squeeze()
        coeff = coeff / (np.linalg.norm(coeff) + 1e-8)
        est_ang, _, _ = estimate_angles_from_coefficients_ESPRIT(coeff, base_info, max_sources=K)
results['OCDOA'] = time.time() - t0
cooldown()

# ═══════════════════════════
# CNN
# ═══════════════════════════
t0 = time.time()
for _ in range(N_TRIALS):
    with torch.no_grad():
        r = lowsnr(data_t_low).cpu().numpy().squeeze()
        peaks, _ = find_peaks(r)
        if len(peaks) < K:
            peaks = np.argsort(r)[::-1][:K]
        est = np.sort(np.array(sorted(peaks, key=lambda x: r[x], reverse=True)[:K]) - 60)
results['CNN'] = time.time() - t0
cooldown()

# ═══════════════════════════
# HMCViT
# ═══════════════════════════
t0 = time.time()
for _ in range(N_TRIALS):
    with torch.no_grad():
        out = torch.sigmoid(hmcvit(data_t_hmc)).detach().cpu().numpy().flatten()
        peaks, _ = find_peaks(out)
        if len(peaks) < K:
            peaks = np.argsort(out)[::-1][:K]
        est = np.sort(hmcvit_grid[sorted(peaks, key=lambda x: out[x], reverse=True)[:K]])
results['HMCViT'] = time.time() - t0
cooldown()

# ═══════════════════════════
# MUSIC with grid 1°
# ═══════════════════════════
t0 = time.time()
for _ in range(N_TRIALS):
    P = np.zeros(120)
    for i in range(120):
        a = np.exp(-1j * 2 * np.pi * dd / wavelength * np.sin((1 * i - 60) * derad))
        P[i] = (1 / np.abs(np.matrix.getH(a) @ U_mat @ U_mat.H @ a)).item()
    P = P / np.max(P)
    order = np.diff(P)
    est, _ = spectral_peak_search(order, P, DOA, -60, 1, WF=True)
results['MUSIC 1°'] = time.time() - t0
cooldown()

# ═══════════════════════════
# MUSIC with grid 0.1°
# ═══════════════════════════
t0 = time.time()
for _ in range(N_TRIALS):
    P = np.zeros(1200)
    for i in range(1200):
        a = np.exp(-1j * 2 * np.pi * dd / wavelength * np.sin((0.1 * i - 60) * derad))
        P[i] = (1 / np.abs(np.matrix.getH(a) @ U_mat @ U_mat.H @ a)).item()
    P = P / np.max(P)
    order = np.diff(P)
    est, _ = spectral_peak_search(order, P, DOA, -60, 0.1, WF=True)
results['MUSIC 0.1°'] = time.time() - t0
cooldown()

# ═══════════════════════════
# CBF with grid 1°
# ═══════════════════════════
t0 = time.time()
for _ in range(N_TRIALS):
    P = np.zeros(120)
    for i in range(120):
        a = np.exp(-1j * 2 * np.pi * dd / wavelength * np.sin((1 * i - 60) * derad))
        P[i] = np.abs(np.matrix.getH(a) @ Rx @ a).item()
    P = P / np.max(P)
    order = np.diff(P)
    est, _ = spectral_peak_search(order, P, DOA, -60, 1, WF=True)
results['CBF 1°'] = time.time() - t0
cooldown()

# ═══════════════════════════
# CBF with grid 0.1°
# ═══════════════════════════
t0 = time.time()
for _ in range(N_TRIALS):
    P = np.zeros(1200)
    for i in range(1200):
        a = np.exp(-1j * 2 * np.pi * dd / wavelength * np.sin((0.1 * i - 60) * derad))
        P[i] = np.abs(np.matrix.getH(a) @ Rx @ a).item()
    P = P / np.max(P)
    order = np.diff(P)
    est, _ = spectral_peak_search(order, P, DOA, -60, 0.1, WF=True)
results['CBF 0.1°'] = time.time() - t0

# ═══════════════════════════
# Table
# ═══════════════════════════
print(f"\n{'='*60}")
print(f"  Inference Time ({N_TRIALS} trials, SNR={SNR}dB, CPU)")
print(f"{'='*60}")
fastest = min(results.values())
for name, t in results.items():
    ratio = t / fastest
    print(f"  {name:20s} {t:7.2f}s  {t/N_TRIALS*1000:7.2f}ms  x{ratio:.1f}")
print(f"{'='*60}")

# ═══════════════════════════
# Plot: paper-style two-panel figure
# ═══════════════════════════
## ═══════════════════════════════════════════════════════════
# Plot: paper-style two-panel figure
# ═══════════════════════════════════════════════════════════
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

os.makedirs('pic_final', exist_ok=True)

# ============================================================
# Display-name mapping
# ============================================================
name_map = {
    'OCDOA': 'OC-DOA',
    'CNN': 'CNN',
    'HMCViT': 'HMC-ViT',
    'CBF 1°': r'CBF ($1^\circ$)',
    'MUSIC 1°': r'MUSIC ($1^\circ$)',
    'CBF 0.1°': r'CBF ($0.1^\circ$)',
    'MUSIC 0.1°': r'MUSIC ($0.1^\circ$)',
}

# ============================================================
# Colors
# ============================================================
# Keep the same color for the same DL model in both panels
color_ocdoa = '#4C72B0'     # blue
color_cnn = '#DD8452'       # orange
color_hmcvit = '#55A868'    # green

color_cbf1 = '#8172B3'      # purple
color_music1 = '#C44E52'    # red
color_cbf01 = '#64B5CD'     # cyan
color_music01 = '#CCB974'   # yellow/brown

time_color_map = {
    'OCDOA': color_ocdoa,
    'CNN': color_cnn,
    'HMCViT': color_hmcvit,
    'CBF 1°': color_cbf1,
    'MUSIC 1°': color_music1,
    'CBF 0.1°': color_cbf01,
    'MUSIC 0.1°': color_music01,
}

# ============================================================
# Create two-panel figure
# ============================================================
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(7.25, 2.85),
    gridspec_kw={
        'width_ratios': [0.90, 1.35]
    }
)

# ============================================================
# (a) Number of Trainable Parameters
# ============================================================
names = [d[0] for d in param_data]
counts = [d[1] for d in param_data]

param_colors = [
    color_ocdoa,
    color_cnn,
    color_hmcvit
]

bars1 = ax1.bar(
    names,
    counts,
    width=0.60,
    color=param_colors,
    edgecolor='black',
    linewidth=0.55
)

# Value labels
for bar, c in zip(bars1, counts):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(counts) * 0.020,
        f'{c / 1e6:.2f}M',
        ha='center',
        va='bottom',
        fontsize=8
    )

ax1.set_ylabel('Number of Trainable Parameters')

ax1.set_ylim(
    0,
    max(counts) * 1.14
)

# Display y-axis in millions
ax1.yaxis.set_major_formatter(
    FuncFormatter(
        lambda x, pos: f'{x / 1e6:.0f}M'
    )
)

# Paper-style axes
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax1.grid(
    axis='y',
    linestyle='--',
    linewidth=0.45,
    alpha=0.30
)

ax1.set_axisbelow(True)

# ============================================================
# (b) Average Running Time
# ============================================================

# Original results contain total runtime over N_TRIALS.
# Convert to average runtime for ONE trial and express in ms.
avg_time_ms = {
    name: total_time / N_TRIALS * 1000.0
    for name, total_time in results.items()
}

# Sort from fastest to slowest
sorted_items = sorted(
    avg_time_ms.items(),
    key=lambda x: x[1]
)

raw_names_t = [name for name, _ in sorted_items]

names_t = [
    name_map.get(name, name)
    for name in raw_names_t
]

times_t = [
    t for _, t in sorted_items
]

colors_t = [
    time_color_map[name]
    for name in raw_names_t
]

bars2 = ax2.barh(
    names_t,
    times_t,
    height=0.58,
    color=colors_t,
    edgecolor='black',
    linewidth=0.55
)

# Fastest method displayed at the top
ax2.invert_yaxis()

# Value labels: one decimal place
for bar, t in zip(bars2, times_t):
    ax2.text(
        bar.get_width() + max(times_t) * 0.018,
        bar.get_y() + bar.get_height() / 2,
        f'{t:.1f}',
        va='center',
        ha='left',
        fontsize=8
    )


ax2.set_xlim(
    0,
    max(times_t) * 1.15
)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.grid(
    axis='x',
    linestyle='--',
    linewidth=0.45,
    alpha=0.30
)

ax2.set_axisbelow(True)

# ============================================================
# Subfigure captions BELOW the panels
# ============================================================

ax1.text(
    0.5,
    -0.245,
    '(a) Trainable Parameters',
    transform=ax1.transAxes,
    ha='center',
    va='top',
    fontsize=9
)

ax2.text(
    0.5,
    -0.245,
    '(b) Average Running Time (ms)',
    transform=ax2.transAxes,
    ha='center',
    va='top',
    fontsize=9
)

# ============================================================
# Layout
# ============================================================

fig.subplots_adjust(
    left=0.085,
    right=0.985,
    bottom=0.285,
    top=0.97,
    wspace=0.47
)

# ============================================================
# Save
# ============================================================

output_path = 'pic_final/complexity_comparison.pdf'

fig.savefig(
    output_path,
    dpi=600,
    bbox_inches='tight'
)

plt.close(fig)

print(f"Saved: {output_path}")