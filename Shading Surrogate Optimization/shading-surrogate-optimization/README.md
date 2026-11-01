# Shading design with surrogate models and Grey Wolf search

Python code behind two building-performance studies by Fatemeh Hosseinpour:

1. **Project 1: classroom shading in Tehran.** Light shelf + horizontal louvre, simulated with
   Grasshopper, EnergyPlus and Radiance. Random-forest and gradient-boosting surrogate models,
   variable importance, and a test of how many simulations a surrogate needs.
2. **Project 2: house in Zahedan.** Overhang depth and angle, window-to-wall ratios and orientation,
   simulated with EnergyPlus and Radiance. A Grey Wolf Optimizer (single- and multi-objective)
   searching a surrogate model of the simulation results.

**The simulations and the result tables are not in this repository.** See [Data](#data) for what the code expects and how to run it on synthetic data.

## Metrics

| Symbol | Meaning |
|---|---|
| UDI | Useful Daylight Illuminance, % (higher is better) |
| sDG / DG | Spatial disturbing glare, % (lower is better) |
| EUI | Energy use intensity, kWh/m2.yr (lower is better) |
| Primary energy | 3.6 x cooling + 1.1 x heating + 3.6 x electricity, kWh/m2.yr (Project 1) |
| TCP | Thermal comfort percentage (Project 1) |

## Setup

```
pip install -r requirements.txt
```

Tested with Python 3.12, numpy 2.4, pandas 3.0, scikit-learn 1.8 and matplotlib 3.10.

## Data

The simulation data are not published. They are available on reasonable request (see my GitHub profile).
The `data/` folders are empty on purpose, and `.gitignore` keeps any CSV or Excel file out of the repository.

Without the data you can still run the code on **synthetic data**. These come from invented formulas, are not
simulation results, and do not reproduce any number reported for the study.

```
cd project1_tehran_classroom
python make_synthetic_data.py && python ml_extra.py

cd ../project2_zahedan_house
python make_synthetic_data.py && python prepare_data.py data/raw_simulations.csv
python effects_figure.py && python gwo_surrogate.py
```

Running the scripts **overwrites the files in `figures/`**. If you run them on synthetic data, run them in a copy of the
repository, or restore the published figures afterwards with `git checkout -- figures`.

| Project | File | Columns |
|---|---|---|
| 1 | `data/clean_designs.csv` | `wwr` (window-to-wall ratio, 0 to 1), `angle` (louvre angle, deg), `ls_angle` (light-shelf angle, deg), `depth_l` (louvre depth, m), `spacing_l` (louvre spacing, m), `fins_l` (number of fins), `ls_depth` (light-shelf depth, m), `udi`, `sdg`, `eui`, `penergy`, `tcp` |
| 2 | `data/raw_simulations.csv` | `in:angle` (overhang angle, deg), `in:depth` (overhang depth, m), `in:wwr-n`, `in:wwr-s` (north and south window-to-wall ratio), `in:orientation` (deg), `out:EUI`, `out:DG`, `out:UDI`, `img` (empty) |

## Project 1 (`project1_tehran_classroom`)

| File | What it does |
|---|---|
| `ml_surrogate.py` | Cleans the raw results, trains random forests on a 25 % hold-out, writes permutation importance and prediction plots. Needs the raw spreadsheet as an argument (not included, and no synthetic version is provided). It also writes `data/clean_designs.csv`. |
| `ml_extra.py` | Model comparison (linear, random forest, gradient boosting, neural network), learning curve, partial-dependence plots and the surrogate-assisted search test. Reads `data/clean_designs.csv`. |
| `make_synthetic_data.py` | Writes synthetic data with the same columns as `data/clean_designs.csv`. |

## Project 2 (`project2_zahedan_house`)

| File | What it does |
|---|---|
| `prepare_data.py` | Cleans the raw export (drops failed runs, averages repeated runs), writes `data/zahedan_final.csv` and the search-space table. |
| `gwo_surrogate.py` | Grey Wolf Optimizer on an ExtraTrees surrogate, with a stricter check on unseen window combinations and a comparison with exhaustive search. |
| `effects_figure.py` | Effect of orientation, depth, angle and south window ratio on UDI, sDG and EUI. |
| `make_synthetic_data.py` | Writes a synthetic raw export in the format `prepare_data.py` reads. |

`figures/z_convergence3.png`, `z_effects5.png` and `z_tradeoff5.png` were made outside the scripts in this repository.

## What the results mean, and what they do not

- The Grey Wolf search runs on a **surrogate model**, not on live simulations. Its best design is the
  same as an exhaustive search of the simulated designs, because the space (about 1,700 designs) is small.
- Random-fold cross-validation gives R2 above 0.99 because neighbouring designs are similar. Holding out
  whole window combinations gives R2 0.97 (EUI), 0.88 (sDG) and 0.58 (UDI). Use the second set of numbers.
- The Project 2 search space covers orientations 135, 165 and 180 degrees. Orientation 150 is only partly simulated.
- Results come from one building in one climate each and should not be generalised without new simulations.

## Citation and licence

MIT licence. If you use the code, please cite this repository.
