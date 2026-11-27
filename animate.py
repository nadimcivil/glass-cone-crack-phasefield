"""Usage: python animate.py cone_crack.npz   -> cone_crack.gif  (needs matplotlib, pillow)"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.tri as mt
from matplotlib.animation import FuncAnimation, PillowWriter
f = sys.argv[1]
d = np.load(f)
p, t, S, h = d["p"], d["t"], d["snaps"], d["hist"]
rF, zF, T = float(d["rF"]), float(d["zF"]), p[1].max()
cen = p[:, t].mean(axis=1)
win = (cen[0] < rF) & (cen[1] > T - zF)
fig, ax = plt.subplots(figsize=(6, 5), dpi=110)
tris = [mt.Triangulation(s * p[0], p[1], t.T, mask=~win) for s in (1, -1)]
ims = [ax.tripcolor(tr, S[0], shading="gouraud", cmap="Reds", vmin=0, vmax=1) for tr in tris]
bar, = ax.plot([], [], "k-", lw=3)
ax.set_aspect("equal"); ax.set_xlim(-rF, rF); ax.set_ylim(T - zF, T + 0.02 * zF)
ax.set_xlabel("r (mm)"); ax.set_ylabel("z (mm)")
fig.colorbar(ims[0], ax=ax, shrink=0.8, label="phi (damage)")
title = ax.set_title("")
def upd(k):
    for im in ims: im.set_array(S[k])
    a = h[k, 1]; bar.set_data([-a, a], [T, T])
    title.set_text(f"P = {h[k,0]:.2f} N   crack depth = {h[k,3]*1e3:.0f} um")
    return ims
FuncAnimation(fig, upd, frames=len(S), interval=120).save(f.replace(".npz", ".gif"), writer=PillowWriter(fps=8))
print("saved", f.replace(".npz", ".gif"))
