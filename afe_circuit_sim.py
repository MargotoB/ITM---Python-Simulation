# -*- coding: utf-8 -*-
"""
Курсова работа: Методи за обработка на ултразвукови диагностични сигнали
Симулация на схемите (т. 2.5) изцяло на Python:

  (б) генератор на TGC управлението по референтния дизайн TI TIDA-01427:
      14-битов ЦАП + I/V  ->  MFB филтър 2-ри ред (THS4130)  ->  атенюатор  ->  VCNTL
  (а) канал на приемника по модела на TI AFE5809:
      LNA (24 dB) -> VCA/TGC (0...-40 dB, управляван от VCNTL) -> PGA (24 dB) -> LPF 3-ти ред

Всяко звено е описано с предавателната си функция H(s), изчислена от стойностите на
елементите (ур. 14-21 от документа), и се симулира с scipy.signal (lsim, freqs).

Изисква: Python 3, NumPy, SciPy, Matplotlib
Стартиране:  python afe_circuit_sim.py   -> fig_circuit.png, circuit_results.json
"""
import json
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

# ----------------------------------------------------------------------------
# Стойности на елементите
# ----------------------------------------------------------------------------
# LNA и PGA: неинвертиращи усилватели, ур. (14): G = 1 + Rf/Rg
Rf, Rg = 1500.0, 100.0
G_amp = 1 + Rf / Rg                       # = 16 (24,1 dB)
GBW = 1.5e9                               # честотна лента на ОУ [Hz]
# Sallen-Key LPF (K = 1), ур. (16)-(17): C1 - към маса, C2 - обратна връзка
R1sk, R2sk, C1sk, C2sk = 806.0, 806.0, 10e-12, 39e-12
R3rc, C3rc = 806.0, 20e-12                # пасивно RC звено (3-ти полюс)
# TIDA-01427: MFB филтър, ур. (19)-(20); атенюатор, ур. (21)
R1m, R2m, R3m = 1000.0, 1000.0, 499.0
C1m, C2m = 3000e-12, 750e-12              # C1m = 2 x 1500 pF (диференциално)
Ra1, Ra2 = 348.0, 28.0
k_att = Ra2 / (Ra1 + Ra2)                 # = 1/13,4
VFS = 20.0                                # диференциален размах на ЦАП + I/V [V]
NBITS = 14
# VCA, ур. (15): G_VCA[dB] = -26,67 * VCNTL
k_vca = 40.0 / 1.5                        # dB/V

# ----------------------------------------------------------------------------
# Предавателни функции H(s) = num(s)/den(s) (коефициенти по намаляващи степени на s)
# ----------------------------------------------------------------------------
def tf_noninv(G, gbw):
    """Неинвертиращ усилвател с ОУ с еднополюсна характеристика:
    H(s) = G / (1 + s G / (2 pi GBW))  (честотна лента GBW/G)."""
    return [G], [G / (2 * np.pi * gbw), 1.0]

def tf_sallen_key(R1, R2, C1, C2):
    """Ур. (16): H(s) = 1 / (s^2 R1 R2 C1 C2 + s C1 (R1 + R2) + 1)."""
    return [1.0], [R1 * R2 * C1 * C2, C1 * (R1 + R2), 1.0]

def tf_rc(R, C):
    """Пасивно RC звено: H(s) = 1 / (s R C + 1)."""
    return [1.0], [R * C, 1.0]

def tf_mfb(R1, R2, R3, C1, C2):
    """Ур. (19): H(s) = -(R2/R1) / (s^2 R2 R3 C1 C2 + s C2 (R2 + R3 + R2 R3 / R1) + 1)."""
    return [-R2 / R1], [R2 * R3 * C1 * C2, C2 * (R2 + R3 + R2 * R3 / R1), 1.0]

def series(*tfs):
    """Последователно свързване: умножение на числителите и знаменателите."""
    num, den = np.array([1.0]), np.array([1.0])
    for n, d in tfs:
        num, den = np.polymul(num, n), np.polymul(den, d)
    return num, den

H_amp = tf_noninv(G_amp, GBW)
H_lpf = series(tf_sallen_key(R1sk, R2sk, C1sk, C2sk), tf_rc(R3rc, C3rc))
H_mfb = tf_mfb(R1m, R2m, R3m, C1m, C2m)

# аналитични стойности от ур. (17) и (20)
fc_sk = 1 / (2 * np.pi * np.sqrt(R1sk * R2sk * C1sk * C2sk))
Q_sk = np.sqrt(R1sk * R2sk * C1sk * C2sk) / (C1sk * (R1sk + R2sk))
f0_m = 1 / (2 * np.pi * np.sqrt(R2m * R3m * C1m * C2m))
Q_m = np.sqrt(R2m * R3m * C1m * C2m) / (C2m * (R2m + R3m + R2m * R3m / R1m))
tau_g = 1 / (2 * np.pi * f0_m * Q_m)

# ----------------------------------------------------------------------------
# Времева ос и входен сигнал: 5 ехосигнала от 10...50 mm (t = 2z/c), f0 = 5 MHz
# амплитудата намалява с 5 dB/cm (0,5 dB/(cm MHz) x 5 MHz x 2); 50/50 Ohm делител на входа
# ----------------------------------------------------------------------------
dt = 2e-9
t = np.arange(0, 80e-6, dt)
f0, sg = 5e6, 125e-9
t_echo = np.array([13.0, 26.0, 39.0, 51.9, 64.9]) * 1e-6
a_echo = 20e-3 * 10 ** (-5 * np.arange(5) / 20)          # 20 mV ... 2 mV (на източника)
v_src = sum(a * np.exp(-0.5 * ((t - te) / sg) ** 2) * np.sin(2 * np.pi * f0 * (t - te))
            for a, te in zip(a_echo, t_echo))
v_in = v_src * 50 / (50 + 50)                             # Rs = Rterm = 50 Ohm

# ----------------------------------------------------------------------------
# (б) TGC генератор по TIDA-01427
# ----------------------------------------------------------------------------
# желана крива: VCNTL спада с 14,4 mV/us (TGC 5 dB/cm), от 1,0877 V до 0,0769 V
def vcntl_target(tt):
    return np.clip(1.0877 - 14444 * tt, 0.0769, None)

# ЦАП: кодът се обновява на всяка 1 us и се подава с tau_g по-рано (компенсация на закъснението)
t_upd = np.floor(t / 1e-6) * 1e-6
D = np.round(2 ** NBITS * vcntl_target(t_upd + tau_g) / (VFS * k_att))
v_dac = -VFS * D / 2 ** NBITS                             # ур. (18), диференциален еквивалент

def lsim_steady(tf, u, tt):
    """lsim с начално състояние, равно на установения режим при u[0] (като DC работна точка)."""
    A, B, C, Dm = signal.tf2ss(*tf)
    x0 = np.linalg.solve(-A, B.flatten() * u[0])
    _, y, _ = signal.lsim((A, B, C, Dm), u, tt, X0=x0)
    return y

v_mfb = lsim_steady(H_mfb, v_dac, t)
v_cntl = v_mfb * k_att                                    # ур. (21)

# ----------------------------------------------------------------------------
# (а) Канал на приемника по AFE5809
# ----------------------------------------------------------------------------
v_lna = lsim_steady(H_amp, v_in, t)
g_vca = 10 ** (-k_vca * v_cntl / 20)                      # ур. (15)
v_vca = v_lna * g_vca
v_vca_fix = v_lna * 10 ** (-14 / 20)                      # сравнение: постоянно -14 dB (без TGC)
v_pga = lsim_steady(H_amp, v_vca, t)
v_out = lsim_steady(H_lpf, v_pga, t)

# ----------------------------------------------------------------------------
# Честотни характеристики (АЧХ)
# ----------------------------------------------------------------------------
f_lpf = np.logspace(5, np.log10(2e8), 2000)
_, h_lpf = signal.freqs(*H_lpf, worN=2 * np.pi * f_lpf)
f_m = np.logspace(3, 7, 2000)
_, h_m = signal.freqs(*H_mfb, worN=2 * np.pi * f_m)
db_lpf, db_m = 20 * np.log10(np.abs(h_lpf)), 20 * np.log10(np.abs(h_m))

def f3db(f, db):
    i = np.where(db <= db[0] - 3)[0][0]
    return np.interp(db[0] - 3, [db[i], db[i - 1]], [f[i], f[i - 1]])

# ----------------------------------------------------------------------------
# Резултати
# ----------------------------------------------------------------------------
def peaks(v):
    return np.array([np.abs(v[(t > te - 1e-6) & (t < te + 1.5e-6)]).max() for te in t_echo])

p_in, p_out, p_fix = peaks(v_in), peaks(v_out), peaks(v_vca_fix)
sel = (t > 8e-6) & (t < 68e-6)
err = np.max(np.abs(v_cntl[sel] - vcntl_target(t[sel])))
res = {
    "G_amp_db": 20 * np.log10(G_amp),
    "fc_sk_MHz": fc_sk / 1e6, "Q_sk": Q_sk, "f_rc_MHz": 1 / (2 * np.pi * R3rc * C3rc) / 1e6,
    "lpf_f3db_MHz": f3db(f_lpf, db_lpf) / 1e6,
    "lpf_att_5MHz_db": float(np.interp(5e6, f_lpf, db_lpf)),
    "lpf_att_20MHz_db": float(np.interp(20e6, f_lpf, db_lpf)),
    "mfb_f0_kHz": f0_m / 1e3, "mfb_Q": Q_m, "mfb_tau_g_us": tau_g * 1e6,
    "mfb_dc_gain_db": float(db_m[0]), "mfb_f3db_kHz": f3db(f_m, db_m) / 1e3,
    "k_att": k_att, "vcntl_fullscale_V": VFS * k_att,
    "vcntl_err_mV": err * 1e3, "vcntl_err_dB": err * k_vca,
    "peak_in_mV": (p_in * 1e3).tolist(), "peak_out_mV": (p_out * 1e3).tolist(),
    "in_spread_db": 20 * np.log10(p_in.max() / p_in.min()),
    "out_spread_db": 20 * np.log10(p_out.max() / p_out.min()),
    "fix_spread_db": 20 * np.log10(p_fix.max() / p_fix.min()),
    "gain_echo1_db": 20 * np.log10(p_out[0] / p_in[0]),
    "gain_echo5_db": 20 * np.log10(p_out[-1] / p_in[-1]),
}
print(json.dumps(res, indent=2, ensure_ascii=False))
json.dump(res, open("circuit_results.json", "w"), indent=2)

# ----------------------------------------------------------------------------
# Фигура
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(7.0, 4.4))
gs = fig.add_gridspec(3, 2, width_ratios=[2.2, 1])
ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
ax4 = fig.add_subplot(gs[0:2, 1]); ax5 = fig.add_subplot(gs[2, 1])
us = t * 1e6
ax1.plot(us, v_in * 1e3, lw=0.5, color="#1f4e79"); ax1.set_ylabel("V(in), mV")
ax1.set_title("а) вход на LNA (ехо от 10…50 mm)", fontsize=8)
ax2.plot(us, -v_dac * k_att, color="#bf9000", lw=0.7, label="ЦАП (стъпала, мащаб.)")
ax2.plot(us, v_cntl, color="#548235", lw=1.1, label="VCNTL след MFB и атенюатор")
ax2.set_ylabel("V"); ax2.legend(fontsize=6.5, loc="upper right")
ax2.set_title("б) TGC управление по TIDA-01427", fontsize=8)
ax3.plot(us, v_out, lw=0.5, color="#c00000"); ax3.set_ylabel("V(out), V")
ax3.set_title("в) изход на AFE канала (към АЦП)", fontsize=8); ax3.set_xlabel("t, µs")
for ax in (ax1, ax2, ax3):
    ax.grid(alpha=0.3)
ax4.semilogx(f_lpf / 1e6, db_lpf, color="#1f4e79")
ax4.axvline(res["lpf_f3db_MHz"], ls="--", color="gray", lw=0.7)
ax4.set_xlim(0.1, 200); ax4.set_ylim(-80, 5); ax4.set_xlabel("f, MHz"); ax4.set_ylabel("dB")
ax4.set_title(f"г) LPF на AFE, f(-3 dB) = {res['lpf_f3db_MHz']:.2f} MHz", fontsize=8)
ax4.grid(alpha=0.3, which="both")
ax5.semilogx(f_m / 1e3, db_m, color="#548235")
ax5.axvline(res["mfb_f3db_kHz"], ls="--", color="gray", lw=0.7)
ax5.set_xlim(1, 10000); ax5.set_ylim(-80, 5); ax5.set_xlabel("f, kHz"); ax5.set_ylabel("dB")
ax5.set_title(f"д) MFB филтър, f(-3 dB) = {res['mfb_f3db_kHz']:.0f} kHz", fontsize=8)
ax5.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig("fig_circuit.png", dpi=200); plt.close(fig)
print("OK")
