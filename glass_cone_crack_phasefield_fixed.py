"""
Axisymmetric phase-field (AT2) fracture of glass under a Hertzian contact load  ->  cone crack.

Library : scikit-fem  (pip install scikit-fem scipy numpy matplotlib)
Units   : mm, N, MPa.   Gc = 8 J/m^2 = 0.008 N/mm.
Model   : - axisymmetric (r, z), P1 triangles, load on the top surface z = T
          - ball replaced by the Hertz pressure p(r) = p0*sqrt(1-(r/a)^2) on r < a  (quasi-static)
          - AT2 phase field, Miehe spectral strain-energy split drives the crack (tension only),
            history field H = max(psi+) for irreversibility, staggered (alternate) solution
          - mesh is refined locally around the expected crack region (graded)
Run     : python glass_cone_crack_phasefield.py --ell 0.008 --pmax 3
          python glass_cone_crack_phasefield.py --ell 0.03 --hfac 2 --nsteps 15 --pmax 2   (quick test)
          python glass_cone_crack_phasefield.py --drop_height 0.5 --ell 0.05                (impact load)
"""
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from skfem import (MeshTri, Basis, FacetBasis, ElementTriP1, ElementVector,
                   BilinearForm, LinearForm, DiscreteField, condense, solve)

# ----------------------------------------------------------------- inputs
ap = argparse.ArgumentParser()
ap.add_argument("--ell", type=float, default=0.008, help="phase-field length scale (mm); sets strength")
ap.add_argument("--hfac", type=float, default=2.5, help="h_min = ell / hfac near the crack region")
ap.add_argument("--pmax", type=float, default=3.0, help="max load = pmax * P_est")
ap.add_argument("--pmin", type=float, default=0.5, help="first load = pmin * P_est")
ap.add_argument("--nsteps", type=int, default=60, help="number of load steps")
ap.add_argument("--R", type=float, default=5.0, help="ball radius (mm)")
ap.add_argument("--flaw_depth", type=float, default=None,
                help="depth (mm) of the initial surface flaw (default 4*ell; 0 = no flaw)")
ap.add_argument("--flaw_r", type=float, default=None,
                help="radial position (mm) of the flaw (default: contact radius at P_est)")
ap.add_argument("--drop_height", type=float, default=None,
                help="drop height (m); if given, Pmax = Hertz peak force of the impact")
ap.add_argument("--max_stagger", type=int, default=40)
ap.add_argument("--tol", type=float, default=2e-3)
ap.add_argument("--max_elems", type=int, default=400000)
ap.add_argument("--out", default="cone_crack")
args = ap.parse_args()

# ----------------------------------------------------------------- material (glass, steel ball)
E, nu, Gc = 7.0e4, 0.22, 0.008          # MPa, -, N/mm
Es, nus, rho_s = 2.0e5, 0.30, 7850.0    # steel ball: MPa, -, kg/m^3
ell, R = args.ell, args.R
lam = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
KRES = 1e-6                             # residual stiffness
Estar = 1.0 / ((1 - nu**2) / E + (1 - nus**2) / Es)


def hertz(P):
    a = (3 * P * R / (4 * Estar)) ** (1 / 3)
    p0 = 3 * P / (2 * np.pi * a**2)
    return a, p0


# AT2 homogeneous (uniaxial) strength, and a rough initiation-load estimate:
# at the contact edge the surface stress state is pure shear (sig_r = -sig_t = (1-2nu)/3 p0),
# for which psi+ = (1+nu) sig^2 / (2E)  ->  critical edge stress = sigma_c / sqrt(1+nu)
sigma_c = 3 * np.sqrt(3) / 16 * np.sqrt(E * Gc / ell)
sig_edge = sigma_c / np.sqrt(1 + nu)
p0_est = 3 * sig_edge / (1 - 2 * nu)
P_est = p0_est**3 * np.pi**3 * R**2 / (6 * Estar**2)

Pmax = args.pmax * P_est
flaw_d = 4 * ell if args.flaw_depth is None else args.flaw_depth
flaw_r = hertz(P_est)[0] if args.flaw_r is None else args.flaw_r
if args.drop_height is not None:
    Rm = R * 1e-3
    mball = rho_s * 4 / 3 * np.pi * Rm**3
    U = mball * 9.81 * args.drop_height * 1e3            # N*mm
    delta = (15 * U / (8 * Estar * np.sqrt(R))) ** 0.4    # from U = (8/15) E* sqrt(R) delta^(5/2)
    Pmax = 4 / 3 * Estar * np.sqrt(R) * delta**1.5
    print(f"ball mass = {mball*1e3:.2f} g, impact energy = {U/1e3:.4f} J, "
          f"Hertz delta_max = {delta*1e3:.1f} um, P_max = {Pmax:.1f} N")

a_max, _ = hertz(Pmax)
print(f"sigma_c(AT2, uniaxial) = {sigma_c:.1f} MPa, rough initiation load P_est = {P_est:.3g} N, "
      f"Pmax = {Pmax:.3g} N, a(Pmax) = {a_max*1e3:.1f} um, ell = {ell*1e3:.1f} um, a_max/ell = {a_max/ell:.1f}")

# ----------------------------------------------------------------- mesh (graded around the crack region)
Rd, T = 6.0, 3.0                       # domain: r in [0,Rd], z in [0,T]; load at z = T
rF, zF = 4.0 * a_max, 3.5 * a_max      # box where the finest mesh is wanted
hmin = ell / args.hfac
if rF > 0.6 * Rd or zF > 0.6 * T:
    raise SystemExit("Load too large for the 6 x 3 mm domain: increase Rd/T in the script or lower --pmax.")

m = MeshTri.init_tensor(np.linspace(0, Rd, 13), np.linspace(0, T, 7))


def elem_size(msh):
    Pt = msh.p[:, msh.t]
    c = Pt.mean(axis=1)
    v1, v2 = Pt[:, 1] - Pt[:, 0], Pt[:, 2] - Pt[:, 0]
    area = 0.5 * np.abs(v1[0] * v2[1] - v1[1] * v2[0])
    return c, np.sqrt(2 * area)


for _ in range(80):
    c, h = elem_size(m)
    d = np.hypot(np.maximum(c[0] - rF, 0), np.maximum((T - zF) - c[1], 0))
    mark = np.nonzero(h > hmin + 0.35 * d)[0]
    if mark.size == 0:
        break
    m = m.refined(mark)
    if m.t.shape[1] > args.max_elems:
        raise SystemExit(f"Mesh would exceed {args.max_elems} elements: increase --ell or lower --hfac/--pmax.")
print(f"mesh: {m.t.shape[1]} triangles, {m.p.shape[1]} nodes, h_min = {elem_size(m)[1].min()*1e3:.1f} um")

# ----------------------------------------------------------------- spaces, boundary conditions
elem_u = ElementVector(ElementTriP1())
bu = Basis(m, elem_u, intorder=3)
bp = bu.with_element(ElementTriP1())
top = m.facets_satisfying(lambda x: x[1] > T - 1e-9)
fb = FacetBasis(m, elem_u, facets=top, intorder=4)

n_axis = np.nonzero(m.p[0] < 1e-12)[0]
n_bottom = np.nonzero(m.p[1] < 1e-12)[0]
Dofs = np.concatenate([bu.nodal_dofs[0, n_axis],      # u_r = 0 on the axis
                       bu.nodal_dofs[1, n_bottom]])   # u_z = 0 on the bottom (frictionless support)
pnode = bp.nodal_dofs[0]
dof_uz = bu.nodal_dofs[1]

# initial surface flaw: a thin band with phi = 1 (Dirichlet) at r = flaw_r, depth flaw_d
if flaw_d > 0:
    fn = np.nonzero((np.abs(m.p[0] - flaw_r) < hmin) & (m.p[1] > T - flaw_d))[0]
    flaw_dofs = pnode[fn]
    print(f"initial flaw: r = {flaw_r*1e3:.1f} um, depth = {flaw_d*1e3:.1f} um ({flaw_dofs.size} nodes)")
else:
    flaw_dofs = np.array([], dtype=int)

X = bu.global_coordinates().value        # (2, nel, nqp) physical quadrature points
Rq = X[0]


# ----------------------------------------------------------------- weak forms (axisymmetric: weight r)
def strain(u, r):
    return (u.grad[0][0], u.grad[1][1], u.value[0] / r, 0.5 * (u.grad[0][1] + u.grad[1][0]))


@BilinearForm
def Kform(u, v, w):
    r = w.x[0]
    s = w["S"].value                      # 1 where the (previous-iterate) state is tensile, else 0
    g = s * ((1 - w["phi"].value) ** 2 + KRES) + (1 - s)   # no degradation in compression
    a1, a2, a3, a4 = strain(u, r)
    b1, b2, b3, b4 = strain(v, r)
    return g * (lam * (a1 + a2 + a3) * (b1 + b2 + b3)
                + 2 * mu * (a1 * b1 + a2 * b2 + a3 * b3 + 2 * a4 * b4)) * r


@BilinearForm
def Aphi(u, v, w):
    r = w.x[0]
    H = w["H"].value
    return (Gc * ell * (u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1])
            + (Gc / ell + 2 * H) * u.value * v.value) * r


@LinearForm
def bphi(v, w):
    return 2 * w["H"].value * v.value * w.x[0]


def traction_vec(P):
    a, p0 = hertz(P)

    @LinearForm
    def load(v, w):
        r = w.x[0]
        p = p0 * np.sqrt(np.clip(1 - (r / a) ** 2, 0, None))
        return -p * v.value[1] * r

    return load.assemble(fb)


def principal_and_energy(uvec):
    uf = bu.interpolate(uvec)
    err, ezz, ett, erz = strain(uf, Rq)
    err, ezz, ett, erz = (np.asarray(q) for q in (err, ezz, ett, erz))
    mean = 0.5 * (err + ezz)
    rad = np.sqrt((0.5 * (err - ezz)) ** 2 + erz**2)
    e1, e2, e3 = mean + rad, mean - rad, ett
    tr = e1 + e2 + e3
    pos = lambda q: np.maximum(q, 0.0)
    psi_plus = 0.5 * lam * pos(tr) ** 2 + mu * (pos(e1) ** 2 + pos(e2) ** 2 + pos(e3) ** 2)
    sig_rr = lam * tr + 2 * mu * err
    return psi_plus, sig_rr, np.maximum(e1, e3)


# ----------------------------------------------------------------- load stepping
Ps = np.linspace(args.pmin * P_est, Pmax, args.nsteps)
phi = np.zeros(bp.N)
phi[flaw_dofs] = 1.0
xfix = phi.copy()
H = np.zeros_like(Rq)
S = np.zeros_like(Rq)
hist = []
snaps = []
P_init = None
top_nodes = m.p[1] > T - 1e-9
el_top = np.nonzero(top_nodes[m.t].any(axis=0))[0]
t0 = time.time()

for k, P in enumerate(Ps):
    f = traction_vec(P)
    if k == 0:
        a0, p00 = hertz(P)
        print(f"load check: sum(f_z)*2pi = {f[dof_uz].sum()*2*np.pi:.4g} N (should be -{P:.4g} N)")
    for it in range(args.max_stagger):
        K = Kform.assemble(bu, phi=bp.interpolate(phi), S=DiscreteField(S))
        u = solve(*condense(K, f, D=Dofs))
        psi, sig_rr, trq = principal_and_energy(u)
        S = (trq > 0).astype(float)
        Htrial = np.maximum(H, psi)
        Hf = DiscreteField(Htrial)
        Ap, bp_ = Aphi.assemble(bp, H=Hf), bphi.assemble(bp, H=Hf)
        phi_new = solve(*condense(Ap, bp_, x=xfix, D=flaw_dofs)) if flaw_dofs.size else solve(Ap, bp_)
        phi_new = np.minimum(np.maximum(phi_new, phi), 1.0)      # irreversibility, bounds
        err = np.abs(phi_new - phi).max()
        phi = phi_new
        if k == 0 and it == 0:
            a0, p00 = hertz(P)
            fe = sig_rr[el_top].max()
            th = (1 - 2 * nu) / 3 * p00
            print(f"elastic check at P = {P:.3g} N: max sigma_rr near surface = {fe:.2f} MPa, "
                  f"Hertz (1-2nu)/3*p0 = {th:.2f} MPa, ratio = {fe/th:.2f}")
        if err < args.tol and it >= 1:
            break
    H = Htrial
    pn = phi[pnode]
    cr = pn > 0.9
    depth = T - m.p[1][cr].min() if cr.any() else 0.0
    rext = m.p[0][cr].max() if cr.any() else 0.0
    a, _ = hertz(P)
    grown = (depth > flaw_d + 3 * hmin) if flaw_d > 0 else (pn.max() > 0.5)
    if P_init is None and grown:
        P_init = P
    snaps.append(pn.copy())
    hist.append((P, a, pn.max(), depth, rext, it + 1))
    print(f"step {k+1:3d}/{len(Ps)}  P = {P:8.3f} N  a = {a*1e3:6.1f} um  phi_max = {pn.max():.3f}  "
          f"crack depth = {depth*1e3:6.1f} um  stagger its = {it+1:2d}  [{time.time()-t0:6.0f} s]")
    if it + 1 >= args.max_stagger:
        print("   warning: staggered iterations did not reach tolerance (unstable crack jump); "
              "use more --nsteps for finer load steps")
    if depth > 0.85 * zF or rext > 0.85 * rF:
        print("crack reached the edge of the finely meshed region -> stopping "
              "(raise --pmax only if you also enlarge the refined box)")
        break

# ----------------------------------------------------------------- results
hist = np.array(hist)
pn = phi[pnode]
print(f"\ncrack growth from the flaw starts at P = {P_init if P_init else float('nan'):.3g} N (homogeneous-strength estimate was {P_est:.3g} N)")
np.savez(args.out + ".npz", p=m.p, t=m.t, phi=pn, hist=hist, snaps=np.array(snaps), rF=rF, zF=zF,
         hist_columns="P_N,a_mm,phi_max,crack_depth_mm,crack_r_extent_mm,stagger_its")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
cen = m.p[:, m.t].mean(axis=1)
win = (cen[0] < rF) & (cen[1] > T - zF)
for sgn in (1, -1):
    tri = mtri.Triangulation(sgn * m.p[0], m.p[1], m.t.T, mask=~win)
    im = ax[0].tripcolor(tri, pn, shading="gouraud", cmap="Reds", vmin=0, vmax=1)
ax[0].plot([-a, a], [T, T], "k-", lw=3)
ax[0].set_aspect("equal")
ax[0].set_xlim(-rF, rF)
ax[0].set_ylim(T - zF, T + 0.02 * zF)
ax[0].set_xlabel("r (mm)")
ax[0].set_ylabel("z (mm)")
ax[0].set_title(f"phase field at P = {hist[-1,0]:.3g} N (mirrored about axis; black bar = contact)")
fig.colorbar(im, ax=ax[0], shrink=0.8, label="phi")
ax[1].plot(hist[:, 0], hist[:, 2], "o-", label="phi_max")
ax[1].set_xlabel("load P (N)")
ax[1].set_ylabel("phi_max")
ax2 = ax[1].twinx()
ax2.plot(hist[:, 0], hist[:, 3] * 1e3, "s-", color="C3", label="crack depth")
ax2.set_ylabel("crack depth (um)")
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(args.out + ".png", dpi=140)
print(f"saved {args.out}.png and {args.out}.npz")
