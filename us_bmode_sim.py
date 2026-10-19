# -*- coding: utf-8 -*-
"""
Курсова работа: Методи за обработка на ултразвукови диагностични сигнали
Компютърна симулация на верига за обработка в B-режим (B-mode).

Етапи:
  1) Синтез на RF сигнали по линеен модел на пулс-ехото (точкови разсейватели)
  2) Лентов FIR филтър
  3) Времева компенсация на усилването (TGC)
  4) Детекция на обвивката чрез аналитичен сигнал (Хилбертова трансформация)
  5) Децимация и преобразуване на растера (scan conversion)
  6) Логаритмична компресия до зададен динамичен обхват
  7) Потискане на спекъла с филтър на Lee (локални статистики)
  8) Количествена оценка: SNR на спекъла, контраст, CNR, разделителна способност

Изисква: Python 3, NumPy, SciPy, Matplotlib
Стартиране:  python us_bmode_sim.py
"""
import json
import numpy as np
from scipy import signal, ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
rng = np.random.default_rng(2026)  # фиксирано семе -> възпроизводими резултати

# ----------------------------------------------------------------------------
# 1. Параметри на системата
# ----------------------------------------------------------------------------
c = 1540.0            # скорост на звука в мека тъкан [m/s]
f0 = 5e6              # централна честота на преобразувателя [Hz]
B_frac = 0.6          # относителна честотна лента (-6 dB) на импулса
fs = 50e6             # честота на дискретизация [Hz]
alpha0_db = 0.5       # коеф. на затихване [dB/(cm*MHz)]
z_min, z_max = 5e-3, 60e-3     # дълбочина на изображението [m]
x_width = 40e-3                # ширина на изображението [m]
n_lines = 128                  # брой линии на сканиране
beam_fwhm = 1.0e-3             # латерална ширина на лъча (FWHM) [m]
dr_db = 50.0                   # динамичен обхват на дисплея [dB]
tgc_max_db = 40.0              # макс. TGC усилване (обхват на VCAT в AFE5809)
noise_rms_rel = 2e-4           # ниво на електронния шум (отн. на макс. ехо)

lam = c / f0                    # дължина на вълната [m]
x_lines = np.linspace(-x_width / 2, x_width / 2, n_lines)

# ----------------------------------------------------------------------------
# 2. Предаван импулс: синусоида с гаусова обвивка
#    p(t) = exp(-t^2 / (2 sigma_t^2)) * cos(2 pi f0 t)
# ----------------------------------------------------------------------------
# връзка между -6 dB ширина на спектъра и sigma_f: FWHM_f = 2 sqrt(2 ln2) sigma_f
sigma_f = B_frac * f0 / (2 * np.sqrt(2 * np.log(2)))
sigma_t = 1 / (2 * np.pi * sigma_f)
t_p = np.arange(-4 * sigma_t, 4 * sigma_t, 1 / fs)
pulse = np.exp(-t_p**2 / (2 * sigma_t**2)) * np.cos(2 * np.pi * f0 * t_p)

# ----------------------------------------------------------------------------
# 3. Фантом от точкови разсейватели (2D, равнина x-z)
# ----------------------------------------------------------------------------
dens = 25           # разсейватели на mm^2 (>> 1 на клетка -> развит спекъл)
n_sc = int(dens * (x_width + 6e-3) * 1e3 * (z_max - z_min + 4e-3) * 1e3)
xs = rng.uniform(-x_width / 2 - 3e-3, x_width / 2 + 3e-3, n_sc)
zs = rng.uniform(z_min - 2e-3, z_max + 2e-3, n_sc)
amp = rng.normal(0, 1, n_sc)

cysts = [  # (x, z, радиус, мащаб на амплитудата)
    (-9e-3, 25e-3, 4e-3, 0.0),     # анехогенна киста (без разсейватели)
    (9e-3, 25e-3, 4e-3, 2.0),      # хиперехогенна област (+6 dB)
    (-9e-3, 47e-3, 4e-3, 0.0),     # дълбока анехогенна киста
]
for (cx, cz, r, k) in cysts:
    inside = (xs - cx) ** 2 + (zs - cz) ** 2 < r ** 2
    amp[inside] *= k
# точкови цели (силни отражатели) по оста x = 0 и x = 9 mm
pts = [(0.0, z) for z in (10e-3, 20e-3, 30e-3, 40e-3, 50e-3)] + [(9e-3, 47e-3)]
xs = np.concatenate([xs, [p[0] for p in pts]])
zs = np.concatenate([zs, [p[1] for p in pts]])
amp = np.concatenate([amp, np.full(len(pts), 30.0)])

# ----------------------------------------------------------------------------
# 4. Синтез на RF сигналите (линеен модел на пулс-ехото), в честотна област
#    e_l(t) = sum_k a_k s(x_k - x_l) exp(-2 alpha z_k) p(t - 2 z_k / c) + n(t)
# ----------------------------------------------------------------------------
t0 = 2 * (z_min - 2e-3) / c
t1 = 2 * (z_max + 2e-3) / c
n_t = int(np.ceil((t1 - t0) * fs))
t = t0 + np.arange(n_t) / fs
nfft = int(2 ** np.ceil(np.log2(n_t + len(pulse))))
f = np.fft.rfftfreq(nfft, 1 / fs)
P = np.fft.rfft(pulse, nfft) * np.exp(-1j * 2 * np.pi * f * t_p[0])  # центриран

sigma_x = beam_fwhm / (2 * np.sqrt(2 * np.log(2)))
alpha_np = alpha0_db * (f0 / 1e6) / 8.686 * 100     # [Np/m] при f0

# ограничаване на честотите до лентата на импулса (ускорява сметките)
band = np.abs(P) > 1e-6 * np.abs(P).max()
fb = f[band]
rf = np.zeros((n_t, n_lines))
for l, xl in enumerate(x_lines):
    sel = np.abs(xs - xl) < 3 * sigma_x
    a = amp[sel] * np.exp(-(xs[sel] - xl) ** 2 / (2 * sigma_x ** 2)) \
        * np.exp(-2 * alpha_np * zs[sel])
    tau = 2 * zs[sel] / c - t0
    E = np.zeros(len(f), complex)
    # сума от закъснени копия: E(f) = P(f) sum_k a_k exp(-j 2 pi f tau_k)
    for i0 in range(0, len(a), 4000):
        E[band] += np.exp(-2j * np.pi * np.outer(fb, tau[i0:i0 + 4000])) @ a[i0:i0 + 4000]
    rf[:, l] = np.fft.irfft(E * P, nfft)[:n_t]

rf /= np.abs(rf).max()
rf_clean = rf.copy()
rf = rf + noise_rms_rel * rng.normal(0, 1, rf.shape)   # адитивен термичен шум
z_t = c * t / 2                                        # дълбочина на отчета [m]

# ----------------------------------------------------------------------------
# 5. Лентов FIR филтър (метод на прозореца, прозорец на Хеминг), нулева фаза
# ----------------------------------------------------------------------------
n_taps = 65
h_bp = signal.firwin(n_taps, [f0 * (1 - 0.5), f0 * (1 + 0.5)], pass_zero=False,
                     fs=fs, window="hamming")
rf_f = signal.filtfilt(h_bp, [1.0], rf, axis=0)

# ----------------------------------------------------------------------------
# 6. TGC: G(z) = 10^(min(2 alpha0 f0 z, Gmax)/20)
# ----------------------------------------------------------------------------
tgc_db = np.minimum(2 * alpha0_db * (f0 / 1e6) * (z_t * 100), tgc_max_db)
tgc_db -= tgc_db[np.argmin(np.abs(z_t - z_min))]      # 0 dB в началото
tgc_db = np.clip(tgc_db, 0, None)
G = 10 ** (tgc_db / 20)
rf_tgc = rf_f * G[:, None]

# ----------------------------------------------------------------------------
# 7. Обвивка: A(t) = |x_a(t)|, x_a = x + j H{x}  (аналитичен сигнал чрез FFT)
# ----------------------------------------------------------------------------
env = np.abs(signal.hilbert(rf_tgc, axis=0))
env_notgc = np.abs(signal.hilbert(rf_f, axis=0))

# ----------------------------------------------------------------------------
# 8. Децимация по оста z и scan conversion към квадратни пиксели 0,1 mm
# ----------------------------------------------------------------------------
def to_grid(e):
    dz = 0.1e-3
    zg = np.arange(z_min, z_max, dz)
    xg = np.arange(-x_width / 2, x_width / 2 + 1e-9, dz)
    # интерполация първо по z (по всяка линия), после по x (линейна)
    ez = np.array([np.interp(zg, z_t, e[:, l]) for l in range(n_lines)]).T
    exz = np.array([np.interp(xg, x_lines, ez[i, :]) for i in range(len(zg))])
    return zg, xg, exz

# ниско-честотно изглаждане на обвивката преди децимацията (антиалиасинг)
def smooth(e):
    return ndimage.uniform_filter1d(e, size=5, axis=0)

zg, xg, img = to_grid(smooth(env))
_, _, img_notgc = to_grid(smooth(env_notgc))

# ----------------------------------------------------------------------------
# 9. Логаритмична компресия: I_dB = 20 log10(A/A_max), клипиране до [-DR, 0]
# ----------------------------------------------------------------------------
def log_compress(a, ref):
    idb = 20 * np.log10(np.maximum(a, 1e-12) / ref)
    return np.clip(idb, -dr_db, 0)

ref = img.max()
b_tgc = log_compress(img, ref)
b_notgc = log_compress(img_notgc, img_notgc.max())

# ----------------------------------------------------------------------------
# 10. Филтър на Lee (локални статистики), прилаган върху лог-изображението
#     x_hat = m + k (z - m),  k = max(0, (v - sn2) / v)
# ----------------------------------------------------------------------------
def lee_filter(im, win, sn2):
    m = ndimage.uniform_filter(im, win)
    m2 = ndimage.uniform_filter(im * im, win)
    v = np.maximum(m2 - m * m, 1e-12)
    k = np.clip((v - sn2) / v, 0, 1)
    return m + k * (im - m)

def roi_mask(cx, cz, r):
    X, Z = np.meshgrid(xg, zg)
    return (X - cx) ** 2 + (Z - cz) ** 2 < r ** 2

def box_mask(x_a, x_b, z_a, z_b):
    X, Z = np.meshgrid(xg, zg)
    return (X >= x_a) & (X <= x_b) & (Z >= z_a) & (Z <= z_b)

# хомогенна (фонова) област за оценка на дисперсията на спекъла
bg = box_mask(-4e-3, 4e-3, 28e-3, 36e-3) & ~roi_mask(0, 30e-3, 1.5e-3)
sn2 = np.var(b_tgc[bg])
win = 7
b_lee = lee_filter(b_tgc, win, sn2)
b_med = ndimage.median_filter(b_tgc, size=win)

# ----------------------------------------------------------------------------
# 11. Количествени показатели
# ----------------------------------------------------------------------------
cyst_in = roi_mask(-9e-3, 25e-3, 3e-3)
cyst_bg = box_mask(-15e-3, -3e-3, 18e-3, 32e-3) & ~roi_mask(-9e-3, 25e-3, 5e-3)
hyp_in = roi_mask(9e-3, 25e-3, 3e-3)
hyp_bg = box_mask(3e-3, 15e-3, 18e-3, 32e-3) & ~roi_mask(9e-3, 25e-3, 5e-3) \
    & ~roi_mask(9e-3, 25e-3, 5e-3)

def snr_speckle(a, m):
    v = a[m]
    return v.mean() / v.std()

def cnr_power(a, mi, mo):
    pi, po = a[mi] ** 2, a[mo] ** 2
    return abs(pi.mean() - po.mean()) / np.sqrt(pi.var() + po.var())

def cnr_db(b, mi, mo):
    return abs(b[mi].mean() - b[mo].mean()) / np.sqrt(b[mi].var() + b[mo].var())

def contrast_db(a, mi, mo):
    return 10 * np.log10((a[mi] ** 2).mean() / (a[mo] ** 2).mean())

# разделителна способност по точкова цел при z = 30 mm (x = 0)
def fwhm(profile, step):
    p = profile / profile.max()
    above = np.where(p >= 0.5)[0]
    # линейна интерполация на границите
    i0, i1 = above[0], above[-1]
    left = i0 - 1 + (0.5 - p[i0 - 1]) / (p[i0] - p[i0 - 1])
    right = i1 + (p[i1] - 0.5) / (p[i1] - p[i1 + 1])
    return (right - left) * step

il = np.argmin(np.abs(x_lines - 0.0))
sel_z = (z_t > 29e-3) & (z_t < 31e-3)
ax_prof = env[sel_z, il]
axial_res = fwhm(ax_prof, c / (2 * fs))
iz = np.argmin(np.abs(z_t - 30e-3))
iz = np.where(sel_z)[0][np.argmax(ax_prof)]
lat_prof = env[iz, :]
lat_res = fwhm(lat_prof[il - 10:il + 11], x_lines[1] - x_lines[0])

# теоретична оценка на аксиалната разделителна способност: FWHM на обвивката
# на импулса в пулс-ехо (обвивка exp(-t^2/(2 sigma_t^2)) -> FWHM_t = 2.355 sigma_t)
axial_theory = c * 2 * np.sqrt(2 * np.log(2)) * sigma_t / 2

# SNR на електронния шум спрямо дълбочината (TGC усилва и шума)
sig_rms = np.sqrt(np.mean((signal.filtfilt(h_bp, [1.0], rf_clean, axis=0)) ** 2, axis=1))
noise_rms_f = np.std(rf_f - signal.filtfilt(h_bp, [1.0], rf_clean, axis=0))

res = {
    "lambda_mm": lam * 1e3,
    "sigma_t_ns": sigma_t * 1e9,
    "pulse_len_us": (t_p[-1] - t_p[0]) * 1e6,
    "n_scatterers": int(n_sc),
    "alpha_np_per_m": alpha_np,
    "tgc_max_applied_db": float(tgc_db.max()),
    "tgc_slope_db_per_cm": 2 * alpha0_db * f0 / 1e6,
    "speckle_snr_env": snr_speckle(img, bg),
    "contrast_cyst_db": contrast_db(img, cyst_in, cyst_bg),
    "contrast_hyper_db": contrast_db(img, hyp_in, hyp_bg),
    "cnr_power_cyst": cnr_power(img, cyst_in, cyst_bg),
    "cnr_db_cyst_raw": cnr_db(b_tgc, cyst_in, cyst_bg),
    "cnr_db_cyst_lee": cnr_db(b_lee, cyst_in, cyst_bg),
    "cnr_db_cyst_med": cnr_db(b_med, cyst_in, cyst_bg),
    "cnr_db_hyper_raw": cnr_db(b_tgc, hyp_in, hyp_bg),
    "cnr_db_hyper_lee": cnr_db(b_lee, hyp_in, hyp_bg),
    "cnr_db_hyper_med": cnr_db(b_med, hyp_in, hyp_bg),
    "std_bg_raw_db": float(b_tgc[bg].std()),
    "std_bg_lee_db": float(b_lee[bg].std()),
    "std_bg_med_db": float(b_med[bg].std()),
    "axial_res_mm": axial_res * 1e3,
    "axial_theory_mm": axial_theory * 1e3,
    "lateral_res_mm": lat_res * 1e3,
    "mean_bg_notgc_shallow_db": float(b_notgc[box_mask(-4e-3, 4e-3, 12e-3, 16e-3)].mean()),
    "mean_bg_notgc_deep_db": float(b_notgc[box_mask(4e-3, 18e-3, 52e-3, 56e-3)].mean()),
    "mean_bg_tgc_shallow_db": float(b_tgc[box_mask(-4e-3, 4e-3, 12e-3, 16e-3)].mean()),
    "mean_bg_tgc_deep_db": float(b_tgc[box_mask(4e-3, 18e-3, 52e-3, 56e-3)].mean()),
    "fir_taps": n_taps,
}
print(json.dumps(res, indent=2, ensure_ascii=False))
json.dump(res, open("results.json", "w"), indent=2)

# ----------------------------------------------------------------------------
# 12. Фигури
# ----------------------------------------------------------------------------
ext = [xg[0] * 1e3, xg[-1] * 1e3, zg[-1] * 1e3, zg[0] * 1e3]

# Фиг. А: импулс и спектър + честотна характеристика на FIR
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3))
t_fine = np.linspace(t_p[0], t_p[-1], 2000)   # само за гладко изчертаване
axs[0].plot(t_fine * 1e6, np.exp(-t_fine**2 / (2 * sigma_t**2)) * np.cos(2 * np.pi * f0 * t_fine),
            color="#1f4e79", lw=1)
axs[0].plot(t_fine * 1e6, np.exp(-t_fine**2 / (2 * sigma_t**2)), "--", color="#c00000", lw=0.8)
axs[0].set_xlabel("t, µs"); axs[0].set_ylabel("p(t)"); axs[0].set_title("а) предаван импулс")
axs[0].grid(alpha=0.3)
Pn = np.abs(P) / np.abs(P).max()
wf, Hf = signal.freqz(h_bp, worN=4096, fs=fs)
axs[1].plot(f / 1e6, 20 * np.log10(Pn + 1e-12), color="#1f4e79", lw=1, label="|P(f)| импулс")
axs[1].plot(wf / 1e6, 20 * np.log10(np.abs(Hf) + 1e-12), color="#c00000", lw=1, label="|H(f)| FIR филтър")
axs[1].set_xlim(0, 15); axs[1].set_ylim(-60, 5)
axs[1].set_xlabel("f, MHz"); axs[1].set_ylabel("dB"); axs[1].set_title("б) спектри")
axs[1].legend(fontsize=7, loc="lower center"); axs[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig("fig_pulse.png", dpi=200); plt.close(fig)

# Фиг. Б: една линия (A-линия) по етапите
fig, axs = plt.subplots(3, 1, figsize=(7.0, 4.2), sharex=True)
zz = z_t * 1e3
axs[0].plot(zz, rf[:, il], lw=0.4, color="#1f4e79")
axs[0].set_ylabel("RF"); axs[0].set_title("а) RF сигнал на линия x = 0 (преди TGC)", fontsize=8)
axs[1].plot(zz, rf_tgc[:, il], lw=0.4, color="#1f4e79")
axs[1].plot(zz, env[:, il], lw=0.8, color="#c00000")
ax2 = axs[1].twinx(); ax2.plot(zz, tgc_db, "--", color="#548235", lw=0.9); ax2.set_ylabel("TGC, dB", color="#548235")
axs[1].set_ylabel("RF·G(z)"); axs[1].set_title("б) след филтър и TGC (синьо), обвивка (червено), крива TGC (зелено)", fontsize=8)
axs[2].plot(zz, log_compress(env[:, il], env.max()), lw=0.6, color="#1f4e79")
axs[2].set_ylabel("dB"); axs[2].set_xlabel("дълбочина z, mm")
axs[2].set_title("в) обвивка след логаритмична компресия (DR = 50 dB)", fontsize=8)
for a in axs: a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("fig_aline.png", dpi=200); plt.close(fig)

# Фиг. В: B-mode изображения
fig, axs = plt.subplots(1, 3, figsize=(7.0, 3.6), sharey=True)
for a, im, ttl in zip(axs, [b_notgc, b_tgc, b_lee],
                      ["а) без TGC", "б) с TGC", "в) с TGC + филтър на Lee"]):
    h = a.imshow(im, cmap="gray", extent=ext, vmin=-dr_db, vmax=0, aspect="equal")
    a.set_title(ttl, fontsize=8); a.set_xlabel("x, mm")
axs[0].set_ylabel("z, mm")
cb = fig.colorbar(h, ax=axs, shrink=0.8, pad=0.02); cb.set_label("dB")
fig.savefig("fig_bmode.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# Фиг. Г: ROI-та за метриките (за документацията)
fig, a = plt.subplots(figsize=(3.2, 3.6))
a.imshow(b_tgc, cmap="gray", extent=ext, vmin=-dr_db, vmax=0)
for m, col in [(cyst_in, "r"), (cyst_bg, "y"), (hyp_in, "c"), (bg, "g")]:
    a.contour(xg * 1e3, zg * 1e3, m.astype(float), levels=[0.5], colors=col, linewidths=0.7)
a.set_xlabel("x, mm"); a.set_ylabel("z, mm")
fig.tight_layout(); fig.savefig("fig_roi.png", dpi=200); plt.close(fig)
print("OK")
