# -*- coding: utf-8 -*-
"""Блокова схема на алгоритъма за обработка (B-mode)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
steps = [
    ("Начало", "term"),
    ("Параметри: c, f0, fs, α0,\nфантом от разсейватели", "io"),
    ("Синтез на RF линии\n(линеен модел, ур. 6–7)", "proc"),
    ("Лентов FIR филтър\n2,5–7,5 MHz (ур. 8)", "proc"),
    ("TGC: G(z) = 2·α0·f0·z [dB]\n(ур. 9)", "proc"),
    ("Обвивка: |x + jH{x}|\n(ур. 10–11)", "proc"),
    ("Децимация и scan\nconversion (0,1 mm)", "proc"),
    ("Лог. компресия,\nDR = 50 dB (ур. 12)", "proc"),
    ("Филтър на Lee 7×7\n(ур. 13)", "proc"),
    ("Метрики: SNR, C, CNR,\nразделит. способност (ур. 5)", "proc"),
    ("Изобразяване / край", "term"),
]
# две колони за компактност
fig, ax = plt.subplots(figsize=(6.6, 3.3))
ax.set_xlim(0, 10); ax.set_ylim(-0.1, 6.4); ax.axis("off")
w, h = 3.4, 0.66
pos = []
for i in range(6):
    pos.append((0.4, 5.3 - i * 1.0))
for i in range(5):
    pos.append((6.0, 5.0 - i * 1.0))
colors = {"term": "#d9e2f3", "io": "#fff2cc", "proc": "#e2efda"}
for (txt, kind), (x, y) in zip(steps, pos):
    style = "round,pad=0.02,rounding_size=0.38" if kind == "term" else "square,pad=0.02"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=colors[kind], ec="black", lw=0.8))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center")
for i in range(len(pos) - 1):
    (x0, y0), (x1, y1) = pos[i], pos[i + 1]
    if x0 == x1:
        ax.annotate("", xy=(x1 + w / 2, y1 + h), xytext=(x0 + w / 2, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8))
    else:  # преход между колоните
        xm, ytop = 4.9, y1 + h + 0.3
        ax.plot([x0 + w, xm, xm, x1 + w / 2], [y0 + h / 2, y0 + h / 2, ytop, ytop], color="black", lw=0.8)
        ax.annotate("", xy=(x1 + w / 2, y1 + h), xytext=(x1 + w / 2, ytop),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8))
fig.savefig("fig_flowchart.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("OK")
