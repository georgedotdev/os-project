"""Draws docs/architecture.png / .svg / .pdf in the same visual style as the team's reference."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

W, H = 1600, 1110
INK, GREY, SOFT = "#1d1d1d", "#6b6b6b", "#8c8c8c"
ORANGE, RED, FILL = "#b35a1f", "#98203c", "#f6e6d6"
MONO = "DejaVu Sans Mono"

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, lw=2.0, ec=INK, ls="-", fc="none", r=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                lw=lw, ec=ec, fc=fc, ls=ls if ls == "-" else (0, (5, 3))))


def t(x, y, s, size=8.4, color=INK, weight="normal", ha="center", family=None):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va="center",
            family=family or "DejaVu Sans")


def arrow(pts, color=INK, lw=1.6, dashed=False, head=True):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, ls=(0, (5, 3)) if dashed else "-",
            solid_capstyle="butt", zorder=2)
    if head:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        dx, dy = x1 - x0, y1 - y0
        n = (dx ** 2 + dy ** 2) ** 0.5
        ux, uy = dx / n, dy / n
        s = 9
        tip = (x1, y1)
        a = (x1 - ux * s - uy * s * 0.5, y1 - uy * s + ux * s * 0.5)
        b = (x1 - ux * s + uy * s * 0.5, y1 - uy * s - ux * s * 0.5)
        ax.add_patch(Polygon([tip, a, b], closed=True, fc=color, ec=color, zorder=3))


# ======================= LEFT COLUMN =======================
LX, LW = 80, 655
CX = LX + LW / 2

# Docker host
box(LX, 32, LW, 172, lw=1.3, ec="#777")
t(LX + 28, 50, "DOCKER HOST  (no Kubernetes)", 7.6, "#555", "bold", "left")
t(CX, 80, "Four containers share one Docker engine and one Linux kernel")
t(CX, 106, "web-frontend   ·   api-service   ·   batch-worker   ·   cache-db", 7.6, GREY, family=MONO)
box(LX + 34, 125, 283, 62)
t(LX + 175, 144, "Docker Stats API", 9.2, weight="bold")
t(LX + 175, 169, "deployed capture path", 7.6, GREY)
box(LX + 339, 125, 283, 62, ec=RED, ls="--")
t(LX + 480, 144, "cgroup v2 / eBPF reader", 9.2, RED, "bold")
t(LX + 480, 169, "designed, not built", 7.6, RED)

arrow([(LX + 175, 187), (LX + 175, 257)])
t(LX + 196, 219, "cpu, throttled time, memory, network", 7.9, ha="left")
t(LX + 196, 241, "40,320 rows across 4 containers (7 days, 1 per minute)", 7.6, GREY, ha="left")

# L1
box(LX, 258, LW, 90)
t(CX, 284, "L1  ·  COLLECTOR", 9.6, weight="bold")
t(CX, 316, "usage  +  throttled time   →   one true-demand sample per container")

arrow([(CX, 348), (CX, 418)])
t(CX + 22, 383, "samples, per container, in time order", 7.9, ha="left")

# L2
box(LX, 419, LW, 107)
t(CX, 447, "L2  ·  FEATURE EXTRACTOR", 9.6, weight="bold")
t(CX, 478, "rolling 10-tick window    ·    stateful, one per container")
t(CX, 504, "uses only past samples, so the same code runs offline and live", 7.6, GREY)

arrow([(CX, 526), (CX, 606)])
t(CX + 22, 553, "ONE row  ·  9 numbers  ·  per container per tick", 7.9, ha="left")
t(CX + 22, 579, "cpu, mem, net, Δcpu, mean, std, Δmem, time-of-day sin/cos", 7.6, GREY, ha="left")

# Stage 1
box(LX, 607, LW, 188)
t(LX + 28, 631, "STAGE 1", 9.6, weight="bold", ha="left")
t(LX + 162, 631, "memoryless: sees this one row, nothing before it", 7.6, GREY, ha="left")
box(LX + 34, 661, 283, 108, lw=1.1, ec="#666")
t(LX + 175, 688, "Isolation Forest", 9.0)
t(LX + 175, 716, "how easy is this row to isolate", 7.6, GREY)
t(LX + 175, 736, "from this container's normal rows?", 7.6, GREY)
t(LX + 175, 757, "150 trees · one per container", 7.4, GREY, family=MONO)
box(LX + 339, 661, 283, 108, lw=1.1, ec="#666")
t(LX + 480, 688, "Score normaliser", 9.0)
t(LX + 480, 716, "raw score  →  0 to 1", 7.6, GREY)
t(LX + 480, 736, "burst line = 97th pct of training", 7.6, GREY)
t(LX + 480, 757, "threshold also sent to L3", 7.4, GREY, family=MONO)

# offline training (bottom-left)
box(LX, 852, 300, 104, lw=1.1, ec="#666")
t(LX + 150, 874, "OFFLINE TRAINING", 8.4, weight="bold")
t(LX + 150, 900, "days 1-4 train  ·  day 5 validate", 7.6, GREY)
t(LX + 150, 920, "days 6-7 held-out test", 7.6, GREY)
t(LX + 150, 941, "fits Stage 1 and Stage 2", 7.4, GREY, family=MONO)

# relay
RX = 405
arrow([(RX, 795), (RX, 870), (803, 870), (803, 128), (862, 128)], color=ORANGE, lw=2.4)
t(605, 892, "one burst score, appended as the 10th column", 8.8, ORANGE, "bold")
t(605, 914, "not a vote; it becomes input", 8.0, ORANGE)

# ======================= RIGHT COLUMN =======================
RXL, RW = 870, 657
RC = RXL + RW / 2

box(RXL, 70, RW, 135)
t(RXL + 29, 91, "SLIDING BUFFER", 9.6, weight="bold", ha="left")
t(RXL + 232, 91, "last 30 rows, the oldest drops off as each new one arrives", 7.4, GREY, ha="left")
t(RXL + 44, 150, "out", 7.4, GREY)
for i in range(9):
    x = RXL + 68 + i * 52
    if i == 4:
        t(x + 20, 150, "· · ·", 10, GREY)
        continue
    box(x, 122, 41, 58, lw=1.2, r=6)
box(RXL + 68 + 9 * 52, 122, 41, 58, lw=2.2, ec=ORANGE, fc=FILL, r=6)
t(RXL + 612, 138, "newest", 7.8, ORANGE, "bold")
t(RXL + 612, 162, "10 cols", 7.8, ORANGE, family=MONO)

arrow([(RC, 205), (RC, 268)])
t(RC + 20, 236, "all 30 rows together, in order", 7.9, ha="left")

box(RXL, 269, RW, 110)
t(RXL + 29, 295, "STAGE 2  ·  LSTM", 9.6, weight="bold", ha="left")
t(RXL + 300, 295, "temporal: reads the sequence, so it sees ramps", 7.4, GREY, ha="left")
t(RXL + 29, 329, "30 × 10   →   peak CPU and peak memory over the next 3 minutes", ha="left")
t(RXL + 29, 356, "32 hidden units · plain NumPy · MSE + Adam · early stopping", 7.6, GREY, ha="left")

arrow([(RC, 379), (RC, 440)])
t(RC + 20, 408, "two numbers: cores and GB", 7.9, ha="left")

# L3 allocator
box(RXL, 441, RW, 220)
t(RXL + 29, 467, "L3  ·  ALLOCATOR", 9.6, weight="bold", ha="left")
t(RXL + 29, 500, "CPU target = max(forecast, current use) × (1 + headroom)", ha="left")
t(RXL + 29, 527, "headroom 15%;  score over the burst line  →  15% + 17.5 to 35%", ha="left")
t(RXL + 29, 554, "memory floor = 120% of the highest use in the last 3 hours", ha="left")
t(RXL + 29, 592, "target above limit", 8.6, weight="bold", ha="left")
arrow([(RXL + 262, 592), (RXL + 318, 592)], lw=1.3)
t(RXL + 333, 592, "SCALE UP NOW  (+5%)", 8.6, weight="bold", ha="left")
t(RXL + 29, 626, "below 85% of limit, 5 quiet ticks", 8.6, weight="bold", ha="left")
arrow([(RXL + 262, 626), (RXL + 318, 626)], lw=1.3)
t(RXL + 333, 626, "SCALE DOWN", 8.6, weight="bold", ha="left")
t(RXL + 520, 626, "else HOLD", 8.0, GREY, ha="left")
t(RXL + 29, 648, "clamped to 0.10-4 cores and 0.128-6 GB per container", 7.4, GREY, ha="left")

# bottom boxes
arrow([(RXL + 150, 661), (RXL + 150, 722)])
arrow([(RXL + 507, 661), (RXL + 507, 722)])

box(RXL, 723, 305, 138)
t(RXL + 152, 749, "L4  ·  EXECUTOR", 9.0, weight="bold")
t(RXL + 152, 780, "docker update --cpus 2.15", 7.6, family=MONO)
t(RXL + 152, 800, "--memory 1558m web-frontend", 7.6, family=MONO)
t(RXL + 152, 830, "dry-run by default,", 7.6, GREY)
t(RXL + 152, 848, "--apply to enforce", 7.6, GREY)

box(RXL + 352, 723, 305, 138)
t(RXL + 505, 749, "OPERATOR PATH", 9.0, weight="bold")
t(RXL + 505, 780, "SQLite  →  graphs  →  report", 8.4)
t(RXL + 505, 812, "demand, forecast, burst score,", 7.6, GREY)
t(RXL + 505, 832, "limit, reason and command", 7.6, GREY)
t(RXL + 505, 850, "per container per tick", 7.6, GREY)

# designed-not-built row
arrow([(RXL + 505, 861), (RXL + 505, 902)], color=RED, dashed=True)
box(RXL + 352, 903, 305, 88, ec=RED, ls="--")
t(RXL + 505, 925, "ONLINE RETRAINING", 8.8, RED, "bold")
t(RXL + 505, 950, "nightly refit on recorded samples", 7.6, RED)
t(RXL + 505, 972, "designed, not built", 7.6, RED)

arrow([(RXL + 152, 861), (RXL + 152, 902)], color=RED, dashed=True)
box(RXL, 903, 305, 88, ec=RED, ls="--")
t(RXL + 152, 925, "SWARM / K8S ADAPTER", 8.8, RED, "bold")
t(RXL + 152, 950, "same policy, other executors", 7.6, RED)
t(RXL + 152, 972, "designed, not built", 7.6, RED)

# closed loop back to host
FB = 1010
ax.plot([RXL - 22, RXL - 22], [792, FB], color=SOFT, lw=1.3)
ax.plot([RXL, RXL - 22], [792, 792], color=SOFT, lw=1.3)
ax.plot([RXL - 22, 42], [FB, FB], color=SOFT, lw=1.3)
arrow([(42, FB), (42, 118), (LX, 118)], color=SOFT, lw=1.3)
t(430, FB - 13, "closed loop: new limits take effect on the next tick", 7.6, GREY)

# legend
LY = 1068
ax.plot([95, 153], [LY, LY], color=INK, lw=1.8); t(165, LY, "data flow", 7.6, GREY, ha="left")
ax.plot([290, 348], [LY, LY], color=ORANGE, lw=2.6)
t(360, LY, "the relay: Stage 1's output is Stage 2's input", 7.6, GREY, ha="left")
ax.plot([700, 758], [LY, LY], color=SOFT, lw=1.3); t(770, LY, "feedback", 7.6, GREY, ha="left")
ax.plot([870, 928], [LY, LY], color=RED, lw=1.8, ls=(0, (5, 3)))
t(940, LY, "designed but not built", 7.6, GREY, ha="left")

ax.add_patch(FancyBboxPatch((4, 4), W - 8, H - 8, boxstyle="square,pad=0", lw=2.5,
                            ec="#2a2a2a", fc="none"))

out = os.path.dirname(os.path.abspath(__file__))
for ext in ("png", "svg", "pdf"):
    fig.savefig(os.path.join(out, f"architecture.{ext}"), dpi=200 if ext == "png" else 100)
print("saved docs/architecture.png/.svg/.pdf")
