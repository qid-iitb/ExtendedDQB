# Quadratic power enhancement in an Extended Dicke Quantum Battery

This repository contains the numerical codes and data used in the paper: [arXiv:2512.15607](https://arxiv.org/abs/2512.15607)
(Authors: Harsh Sharma, Himadri Shekhar Dhar)

---

## Repository Structure

### 1. `Data/` 
This folder contains all data files and figures related to the work.

### 2. `simulate_dicke_battery.py`
Python file to simulate the closed quantum battery dynamics.
- Uses **[QuTiP](http://qutip.org/)** (Quantum Toolbox in Python)

> QuTiP Citation:  
> J. R. Johansson, P. D. Nation, and F. Nori, *Comput. Phys. Commun.* **183**, 1760 (2012);  
> *Comput. Phys. Commun.* **184**, 1234 (2013).

#### How to Compile and Run the C++ Code:
```bash
python simulate_dicke_battery.py   --out ./QB_out/ws="$WS"   --cores 21 \
 --wc "$WC" --ws "$WS" \
 --g 0.05 \
 --anis -1 -0.5 0.0 0.5 1 \
 --Nmin 1 --Nmax 100 --Nstep 1 \
 --rmin 0.0 --rmax 2.0 --rstep 0.1 \
 --points-per-first-peak 100 \
 --tail-K 5 --tail-threshold 1e-4 \
 --init-photons N --resume yes \
```

> Note: - tail-K 5 : tracks last 5 diagonal entries in the cavity part of density matrix
