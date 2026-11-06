# Carbon-aware flow-shop scheduling with adjustable processing speed

Core source code for the paper **"Carbon-aware flow-shop scheduling with adjustable processing speed"** (Zijun Wen, Xiang Dong; School of Mechanical Engineering, University of Science and Technology Beijing).

The repository contains the implementation of the optimisation model and the memetic algorithm only. Experiment drivers, result files and plotting scripts are not included.

## What is in the code

| File | Content |
|---|---|
| `src/cas_pfsp.py` | Instance parser, dual random-key decoder, scope-2 emission evaluation, and the baseline memetic algorithm with fixed processing speed (reproduction of Mencaroni et al., 2025). |
| `src/cas_speed.py` | Extended model with adjustable processing speed: speed levels, speed keys, warm start from the fixed-speed optimum, and the memetic algorithm used in the paper. |
| `src/baselines.py` | Genetic algorithm and particle swarm optimisation on the same random-key encoding, used for the comparison in Section 4.4. |

## Environment

- Python 3.10 or later (the reported results were produced with Python 3.14.3).
- NumPy 2.x (the reported results used NumPy 2.4.6).
- No other third-party packages are required for the three modules.

Install the dependency with:

```bash
pip install numpy
```

## Instance data

The experiments use the public CAS-PFSP instance set released by Mencaroni et al.:

- https://github.com/ugent-isye/CAS-PFSP (GPL-3.0)

The instance files are **not** redistributed here. Download that repository and point the parser to its `Data/` directory.

## Minimal usage

```python
import sys
sys.path.insert(0, "src")

from cas_pfsp import parse_instance, ma_cas_pfsp
from cas_speed import SpeedOptions, ma_cas_speed

inst = parse_instance("CAS-PFSP-main/Data/CAS-PFSP-M1T1/CAS-PFSP-M1T1_1.cas")

# Fixed-speed baseline
base = ma_cas_pfsp(inst, rho=120, gamma=60, seed=1)

# Adjustable-speed model, warm-started from the baseline
best = ma_cas_speed(inst, SpeedOptions(levels=(0.8, 0.9, 1.0, 1.1, 1.2), alpha=2.0, aux_share=0.30),
                    rho=120, gamma=60, seed=1, init_solution=base)

print(best.fit)        # scope-2 emissions of the best schedule found, g CO2eq
print(best.makespan)   # makespan in 15-min periods
```

## Model summary

- Objective: minimise scope-2 emissions of purchased electricity, `0.25 * sum_t max(0, L(t) - A(t)) * C(t)`.
- Decision variables: job sequence, idle times between operations, and one speed level per operation.
- Power model: `p_s(k) = beta * p_bar + (p(k) - beta * p_bar) * s^alpha`, with auxiliary power share `beta` and exponent `alpha`.
- Encoding: job keys, pause keys (one array per machine) and speed keys; any real-valued key vector decodes to a feasible schedule.

## Citation

If you use this code, please cite the paper:

```
Wen, Z., and X. Dong. (under review). Carbon-aware flow-shop scheduling with adjustable processing speed.
```

The instance data should be cited as:

```
Mencaroni, A., P. Leyman, B. Raa, S. De Vuyst, and D. Claeys. 2025. "Towards net-zero manufacturing: carbon-aware scheduling for GHG emissions reduction." Journal of Cleaner Production 529: 146787.
```