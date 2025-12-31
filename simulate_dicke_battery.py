#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, argparse, warnings
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qutip as qt
from qutip import Qobj, tensor, basis, ptrace, ket2dm, qeye, jmat
from scipy.sparse.linalg import expm_multiply

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------- utils -----------------

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def psd_jitter(rho, eps=1e-12):
    # Hermitize to kill tiny anti-Hermitian noise
    A = (rho + rho.dag()) * 0.5
    # Add small identity, then renormalize to trace 1
    d = A.shape[0]
    A = A + eps * Qobj(np.eye(d), dims=A.dims)
    A = A / A.tr()               # keep it a valid density matrix
    return A
    
def ergotropy_and_capacity(rho: Qobj, Hb: Qobj):
    E = float(np.real(qt.expect(Hb, rho)))
    eps, _ = Hb.eigenstates()  # ascending
    r_eval, _ = rho.eigenstates()
    r_eval = np.real(np.array(r_eval))
    idx_desc = np.argsort(-r_eval)
    r_sorted = r_eval[idx_desc]
    eps_arr = np.array(eps)
    E_passive = float(np.sum(r_sorted * eps_arr))
    E_active  = float(np.sum(r_sorted * eps_arr[::-1]))
    W = max(E - E_passive, 0.0)
    C = max(E_active - E_passive, 0.0)
    return E, W, C
    
def fisher_energy_basis(rho_list, U, dt):
    T = len(rho_list)
    if T < 3:
        return np.full(T, np.nan)
    Udag = U.dag()
    P = []

    rE = [psd_jitter(Qobj((Udag * rt * U).full())) for rt in rho_list]
    vE = [np.nan]+[np.arccos(qt.fidelity(rE[i], rE[i+1]))/dt for i in range(len(rE)-1)]
    IE = [2*v*v for v in vE]
    
    return IE

# ----------------- model builders -----------------

def build_operators(N, wc, ws, g, anis, r, nph):
    j = N/2.0
    a = qt.destroy(nph); adag = a.dag()
    Iph = qeye(nph)
    Jx, Jy, Jz = jmat(j,'x'), jmat(j,'y'), jmat(j,'z')
    Isp = qeye(int(N+1))
    
    delm = 101-ws
    delp = 101+ws
    gb2 = r*r/delm

    fp = 1 + anis + (anis**2 + anis)*delm/delp
    fm = 1 - anis + (anis**2 - anis)*delm/delp 

    H = (wc * tensor(adag*a, Isp) +
         ws * tensor(Iph, Jz) +
         2.0 * g * tensor(adag + a, Jx) -
         (gb2) * tensor(Iph, fp * (Jx*Jx) + fm * (Jy*Jy)))
    Hb = (Jz + j*qeye(int(N+1)))  # shift baseline so min ε = 0
    return H, Hb, {"a":a,"adag":adag,"Jx":Jx,"Jy":Jy,"Jz":Jz}, Iph, Isp

# ----------------- evolution backends -----------------

def evolve_bdf(H: Qobj, psi0: Qobj, tlist: np.ndarray):
    options = {
        "method": "bdf",
        "nsteps": 50000,
        "atol": 1e-9,
        "rtol": 1e-8,
        "max_step": 0.02,
        "progress_bar": None,
    }
    out = qt.sesolve(H, psi0, tlist, e_ops=[], options=options)
    return out.states

# ----------------- overview plot -----------------

def save_overview_plot_two_panel(path_png, t, E, W, C, P_avg, varHB, varHC, IE,
                                 t1_pre, t_first, params, tail_at_first):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(7.8, 9.0), sharex=True)

    ax1.plot(t, E, label="E(t)", lw=1.8)
    ax1.plot(t, W, label="ergotropy W(t)", lw=1.6)
    ax1.plot(t, C, label="capacity C(t)",  lw=1.4)

    ax4 = ax3.twinx()
    ax2.plot(t, P_avg, label="P_avg(t)",   lw=1.6)
    ax3.plot(t, varHB, label="Var(HB)(t)", lw=1.4, color="tab:orange")
    ax4.plot(t, 4*varHC, label="Var(HC)(t)", lw=1.4, color="tab:red")
    ax4.plot(t, IE, label="IE(t)", lw=1.4, color="tab:green")

    for ax in (ax1, ax2, ax3):
        if np.isfinite(t1_pre): ax.axvline(t1_pre, color="k", ls="-",  lw=1.0, label="t1_pre")
        if np.isfinite(t_first): ax.axvline(t_first, color="k", ls="-.", lw=1.0, label="t_first")
        ax.grid(True, ls=":", alpha=0.25)

    anis, g, r, N = params["anis"], params["g"], params["r"], params["N"]
    ax1.set_ylabel("E, W, C")
    ax3.set_xlabel("t"); ax2.set_ylabel("P_avg"); ax3.set_ylabel("Var(HB)"); ax4.set_ylabel("Var(HB), IE")
    
    ax1.set_xlim((0,max(t)))
    ax2.set_xlim((0,max(t)))
    ax3.set_xlim((0,max(t)))
    ax4.set_xlim((0,max(t)))

    title = f"Overview: anis={anis:g}, g={g:g}, r={r:.3f}, N={N:d}"
    if np.isfinite(tail_at_first):
        title += f"  tail@first={tail_at_first:.2e}"
    ax1.set_title(title)

    def unique_legend(ax):
        handles, labels = ax.get_legend_handles_labels()
        seen = {}; H=[]; L=[]
        for h,l in zip(handles, labels):
            if l not in seen:
                seen[l]=1; H.append(h); L.append(l)
        ax.legend(H, L, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    unique_legend(ax1); #unique_legend(ax2)
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    fig.tight_layout(); fig.savefig(path_png, bbox_inches="tight", dpi=180); plt.close(fig)

# ----------------- peak finder -----------------

def find_first_peak(E, N, thr_ratio=0.45):
    n = len(E); thr = thr_ratio * float(N)
    for k in range(1, n-1):
        if E[k] >= E[k-1] and E[k] >= E[k+1] and E[k] >= thr:
            return k, True
    for k in range(1, n-1):
        if E[k] >= E[k-1] and E[k] >= E[k+1]:
            return k, False
    return int(np.nanargmax(E)), False

# ----------------- worker -----------------

def run_one(task):
    anis, g, r, N, cfg = task
    wc, ws = cfg["wc"], cfg["ws"]
    out_root = cfg["out_root"]
    ppf = cfg["ppf"]
    tail_K = cfg["tail_K"]
    tail_threshold = cfg["tail_threshold"]
    init_photons = cfg["init_photons"]
    resume = bool(cfg.get("resume", True))

    run_dir = os.path.join(out_root, f"N={int(N):02d}",f"anis={anis:.3g}", f"g={g:.3g}", f"r={r:.3f}")
    path_png = os.path.join(out_root, f"N={int(N):02d}",f"anis={anis:.3g}", f"g={g:.3g}", "Plots", f"overview_{r:.3f}.png")
    
    ensure_dir(run_dir)
    
    # Resume: skip if summary.json exists
    summary_path = os.path.join(run_dir, "summary.json")
    if resume and os.path.exists(summary_path):
        return {"skip": True, "anis":anis, "g":g, "r":r, "N":int(N)}

    # photon cutoff
    nph = int(2*int(N) + 1)

    # operators
    H, Hb, ops, Iph, Isp = build_operators(N, wc, ws, g, anis, r, nph)
    Jzf = tensor(Iph,ops["Jz"])

    # initial state
    if isinstance(init_photons, str) and init_photons.upper()=="N":
        n0 = int(N)
    else:
        n0 = int(init_photons)
    n0 = min(max(n0,0), nph-1)
    psi_ph = basis(nph, n0)
    j = N/2.0
    psi_sp = qt.spin_state(j, -j)
    psi0 = tensor(psi_ph, psi_sp)
    
    geff = max(g*np.sqrt(N),r*r/(101-ws)*N)
    t1_pre = np.pi/(2*geff)

    # main time grid
    mult = 20.0
    tmax = mult * float(t1_pre)
    nt = max(int(ppf * (tmax / max(t1_pre, 1e-9))), ppf+1)
    tlist = np.linspace(0.0, float(tmax), nt)
    dt = float(tlist[1]-tlist[0]) if nt>1 else 0.0

    # evolve (BDF → expm fallback)
    states = evolve_bdf(H, psi0, tlist)
    solver_flag = "bdf"

    E = qt.expect(Jzf,states)
    i_peak, is_good = find_first_peak(E, N, thr_ratio=0.45)
    t1_pre = float(tlist[i_peak])

    # main time grid
    mult = 5.0
    tmax = mult * float(t1_pre)
    nt = max(int(ppf * (tmax / max(t1_pre, 1e-9))), ppf+1)
    tlist = np.linspace(0.0, float(tmax), nt)
    dt = float(tlist[1]-tlist[0]) if nt>1 else 0.0

    # evolve (BDF → expm fallback)
    states = evolve_bdf(H, psi0, tlist)
    solver_flag = "bdf"

    # Hb eigenbasis
    _, evecs = Hb.eigenstates()
    U = Qobj(np.hstack([v.full() for v in evecs]), dims=[Hb.dims[0], Hb.dims[0]])

    # traces
    E = np.zeros(nt); W = np.zeros(nt); C = np.zeros(nt); avg_Ph = np.zeros(nt)
    varHB = np.zeros(nt); tail_sum = np.zeros(nt)
    varHC = qt.variance(H,ket2dm(states[0]))*np.ones(nt)
    rhoB_list = []

    Ncav = ops["adag"]*ops["a"]
    for i, psi in enumerate(states):
        rho = ket2dm(psi)
        rhoB = ptrace(rho, 1)
        rhoB_list.append(rhoB)
        Ei, Wi, Ci = ergotropy_and_capacity(rhoB, Hb)
        E[i], W[i], C[i] = Ei, Wi, Ci
        varHB[i] = qt.variance(Hb,rhoB)
        rho_ph = ptrace(rho, 0)
        pops = np.real(np.diag(rho_ph.full()))
        tail_sum[i] = float(np.sum(pops[-tail_K:])) if pops.size >= tail_K else 0.0
        avg_Ph[i] = qt.expect(Ncav,rho_ph)

    IE = fisher_energy_basis(rhoB_list, U, dt) if nt>1 else np.zeros(nt)

    P_avg = np.zeros(nt)
    if nt>1:
        with np.errstate(divide='ignore', invalid='ignore'):
            P_avg[1:] = E[1:] / np.maximum(tlist[1:], 1e-15)
    P_inst = np.gradient(E, tlist) if nt>1 else np.zeros(nt)

    # first peak in production run
    i_peak, is_good = find_first_peak(E, N, thr_ratio=0.45)
    t_first = float(tlist[i_peak])

    # summaries
    E_mean_max = float(np.nanmax(E))
    W_max = float(np.nanmax(W))
    C_max = float(np.nanmax(C))
    Pmax = float(np.nanmax(P_avg[1:])) if nt>1 else 0.0
    Pinstmax = float(np.nanmax(P_inst)) if nt>1 else 0.0
    tail_flag = bool(tail_sum[i_peak] > tail_threshold)

    # save time series
    npz_path = os.path.join(run_dir, f"time_series_{solver_flag}.npz")
    np.savez_compressed(
        npz_path,
        t=tlist, E=E, capacity=C, ergotropy=W,
        P_avg=P_avg, P_inst=P_inst,
        varHB=varHB, IE=IE,
        cav_tail_sum_t=tail_sum, tail_K=np.array([tail_K]), tail_threshold=np.array([tail_threshold]),
        iE_first_peak=np.array([i_peak], dtype=int),
        t1_pre=np.array([t1_pre]),
        nph_max=np.array([nph]),
        solver=np.array([solver_flag]),
        probe_used=np.array([1]),
        avg_Nph=avg_Ph
    )

    # save summary
    summary = dict(
        anis=anis, g=g, r=r, N=int(N), j=N/2.0, nph_max=nph,
        t1_pre=float(t1_pre),
        first_peak_time=t_first, first_peak_energy=float(E[i_peak]),
        peak_is_good=bool(is_good), peak_threshold_ratio=0.45,
        tail_sum_at_first_peak=float(tail_sum[i_peak]),
        tail_K=int(tail_K), tail_threshold=float(tail_threshold),
        tail_flag=tail_flag,
        E_mean_max=E_mean_max, C_max=C_max, W_max=W_max,
        Pmax=Pmax, Pinstmax=Pinstmax, varHC=varHC[0]
    )
    save_json(os.path.join(run_dir, "summary.json"), summary)

    #overview plot (two panels)
    save_overview_plot_two_panel(
        path_png,
        tlist, E, W, C, P_avg, varHB, varHC, IE,
        t1_pre, t_first,
        {"anis":anis,"g":g,"r":r,"N":int(N)},
        tail_sum[i_peak]
    )

    return {
        "ok": True,
        "anis": anis, "g": g, "r": r, "N": int(N), "t1_pre": t1_pre,
        "t_first": t_first, "E_first": float(E[i_peak]),
        "tail_flag": tail_flag,
    }

# ----------------- CLI / main -----------------

def build_grid(anis_vals, g_vals, r_vals, N_vals, out_root, cfg):
    for N in N_vals:
        for anis in anis_vals:
            for g in g_vals:
                for r in r_vals:
                    yield (float(anis), float(g), float(r), int(N), cfg)

def main():
    ap = argparse.ArgumentParser(description="Dicke battery simulator (pre-run first-peak finder, two-panel overview).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cores", type=int, default=max(cpu_count()-1,1))
    ap.add_argument("--wc", type=float, default=1.0)
    ap.add_argument("--ws", type=float, default=1.0)
    ap.add_argument("--g",   nargs="*", type=float, default=[0.01,0.1,0.5,1.0])
    ap.add_argument("--anis", nargs="*", type=float, default=[0.01,0.1,0.5,1.0])
    ap.add_argument("--Nmin", type=int, default=5)
    ap.add_argument("--Nmax", type=int, default=51)
    ap.add_argument("--Nstep", type=int, default=2)
    ap.add_argument("--rmin", type=float, default=0.0)
    ap.add_argument("--rmax", type=float, default=2.0)
    ap.add_argument("--rstep", type=float, default=0.05)
    ap.add_argument("--points-per-first-peak", type=int, default=200, dest="ppf")
    ap.add_argument("--tail-K", type=int, default=5)
    ap.add_argument("--tail-threshold", type=float, default=1e-4)
    ap.add_argument("--init-photons", default="N", help='Int (e.g. 0) or "N" (default)')
    # Resume toggle:
    ap.add_argument("--resume", choices=["yes","no"], default="yes", help="Skip runs with existing summary.json")
    args = ap.parse_args()
    
    out_root = os.path.abspath(args.out)
    ensure_dir(out_root)

    cfg = dict(
        out_root=out_root,
        wc=args.wc, ws=args.ws,
        ppf=args.ppf,
        tail_K=args.tail_K,
        tail_threshold=args.tail_threshold,
        init_photons=args.init_photons,
        resume=(args.resume == "yes"),
    )

    N_vals  = list(range(args.Nmin, args.Nmax+1, args.Nstep))
    r_vals  = np.round(np.arange(args.rmin, args.rmax+1e-10, args.rstep), 3)
    # Remove 2.0 points from g, anis as requested previously
    g_vals   = sorted(set([float(x) for x in args.g if float(x) != 2.0]))
    anis_vals = sorted(set([float(x) for x in args.anis if float(x) != 2.0]))

    grid = list(build_grid(anis_vals, g_vals, r_vals, N_vals, out_root, cfg))
    print(f"[INFO] Starting pool with {args.cores} workers over {len(grid)} combos.")

    with Pool(processes=args.cores) as pool:
        it = pool.imap_unordered(run_one, grid, chunksize=1)
        with tqdm(total=len(grid), ncols=100, dynamic_ncols=True, leave=True) as pbar:
            for info in it:
                pbar.update(1)
                if info and info.get("ok", False):
                    tail = " TAIL*" if info["tail_flag"] else ""
                    tqdm.write(
                        f"[OK] anis={info['anis']:.3g} g={info['g']:.3g} r={info['r']:.5f} "
                        f"N={info['N']:02d} "
                        f"t1_pre≈{info['t1_pre']:.3g}  t_first≈{info['t_first']:.3g}  "
                        f"E1≈{info['E_first']:.3g}  {tail}"
                    )
                elif info.get("skip"):
                    tqdm.write(f"[SKIP] anis={info['anis']:.3g} g={info['g']:.3g} r={info['r']:.5f} N={info['N']:02d} (resume)")
                
                else:
                    tqdm.write("[ERR] A run failed or returned no info.")
    print("[DONE] All runs finished.")

if __name__ == "__main__":
    main()

