# PyCQED
Fewest-switches surface hopping (FSSH) on cavity-QED polaritonic surfaces. A two-level molecule, described by any electronic-structure model (analytic or machine-learned), couples to one lossy cavity mode with an arbitrary number of Fock states. The photon loss enters through a complex frequency omega - i gamma/2, so the polaritonic Hamiltonian is non-Hermitian.

The `pycqed/` package replaces the legacy `polaritonic.py` and `PFSSH.py` scripts. Those scripts are kept unchanged at the bottom of this README for reference.

## Install and test
```bash
pip install -e ".[test]"
pytest
```

## Plugging in an ML model
Write a function that takes the flat Cartesian coordinates `R` (bohr, length 3N) and returns a dict:

| key | shape | required |
|---|---|---|
| `E_g`, `E_e` | scalar (Hartree) | yes |
| `mu_gg`, `mu_ee`, `mu_eg` | 3-vector (a.u.) | yes |
| `dE_g`, `dE_e` | (3N,) | optional |
| `dmu_gg`, `dmu_ee`, `dmu_eg` | (3, 3N) | optional |
| `nac_ge` = <g\|grad e> | (3N,) | optional, default 0 |

Any gradient you leave out is computed by central finite differences, which costs 6N extra model calls per step. `nac_ge` must use the same phase of |e> as `mu_eg`. The code keeps the phase of |e> continuous along trajectories. It sums the overlaps of `mu_eg` (extrapolated with its gradient), `grad mu_eg` and `nac_ge`, so a random sign from the model is harmless. A transition dipole that genuinely passes through zero is not mistaken for a sign flip.

```python
from pycqed import CallableModel, CavityMode, FSSH, FSSHSettings
model = CallableModel(my_ml_predict)
cavity = CavityMode(omega=0.1, coupling=0.02, polarization=(0, 0, 1), gamma=1.8e-4, n_fock=3)
engine = FSSH(model, cavity, FSSHSettings(dt=2.0), coupling="PF")      # "JC" | "Rabi" | "PF"
state = engine.initialize(R0, P0, masses, active=2, rng=0)
record = engine.run(state, n_steps=5000, stride=10)
record.save("traj.npz")
```

Or use the config-driven runner: `python scripts/run_fssh.py examples/mock_ml_diatomic.toml`. The examples are:
- `examples/mock_ml_model.py`: a values-only diatomic standing in for an ML model.
- `examples/legacy_double_well.toml`: the old `run.sh` scenario.

## Physics
- **Basis.** |a(R)> x |n>, with a in {g, e} and n = 0..n_fock. The dipoles enter only as M_ab = e . mu_ab (scalarized along the polarization).
- **Hamiltonians.** With kappa = sqrt(omega/2) lambda and w = omega - i gamma/2:
  - `JC`: diag(E) + w b+b + kappa M_eg (s+ b + s- b+)
  - `Rabi`: diag(E) + w b+b + kappa M_eg (s+ + s-)(b + b+)
  - `PF`: diag(E) + w b+b + kappa M (b + b+) + lambda^2/2 (M M). The last term is the dipole self-energy, projected onto {g, e}. An exact <(e.mu)^2> can be supplied through `ElectronicData.dipole_squared`.
- **c-normalization.** H is complex symmetric. Eigenvectors are c-normalized (U^T U = 1), so the left eigenvectors are U^T and each state is fixed up to a sign, which is tracked from step to step.
- **Solvers.** `NumericalSolver` handles every model. `AnalyticJCSolver` handles JC only, in closed form with one 2x2 block per excitation manifold, and is used as a cross-check.
- **Gradients and NACs.** Both come from the chain rule:
  - grad E_J = [U^T dH U]_JJ
  - d_JK = [U^T dH U]_JK / (E_K - E_J) + [U^T (D_el x 1) U]_JK
  - The second term carries the intrinsic coupling <g|grad e>. In the JC limit it couples LP/UP to |g,0> and to the 2-excitation manifold, but not LP to UP.
- **Dynamics.**
  - The electronic wavefunction is propagated in the |a(R)> x |n> basis.
  - Hopping uses fewest-switches probabilities accumulated over electronic substeps.
  - On a hop, momentum is rescaled along the phase-rotated NAC. Frustrated hops either keep or reverse the momentum.
  - Photon loss is a quantum jump b|psi>, which works for any number of Fock states.
  - Optional: Langevin (BBK) thermostat and energy-based decoherence.
- **Exceptional points.** A warning (`ExceptionalPointWarning`) is raised as the non-Hermitian problem approaches one.

## Package layout
| module | role |
|---|---|
| `electronic.py` | `ElectronicData`, `CallableModel` (finite-difference fallback), `align_phase` |
| `models.py` | `DoubleWellModel` (legacy surfaces), `DiabaticTwoStateModel` (has an intrinsic NAC) |
| `cavity.py` | `CavityMode`, polarization projection |
| `hamiltonian.py` | JC / Rabi / PF matrices, their gradients, the electronic NAC operator |
| `polariton.py` | numerical and analytic-JC solvers, NACs, state and sign tracking |
| `fssh.py` | the surface-hopping engine |
| `sampling.py` | Wigner sampling (1-D and normal modes) |
| `io.py` | trajectory records, `scan_pes` |
| `cli.py` | TOML runner |

---

# Legacy scripts (`polaritonic.py`, `PFSSH.py`)
These are kept for reference and regression testing. Known problems are listed in the audit on branch `refactor/ml-polariton-fssh`. The two most important:
- Since commit 98aa224 the imaginary part of the polariton energies is not written out, so the photon decay is missing from the dynamics.
- Hops from UP to the ground state are never taken.

### Quickstart

- Clone the repository

- To run a single trajectory: `python PFSSH.py Scale_True Complex_DC 1 1.0 2.45 0.02 test > output.txt &`

In the above example, there are 7 arguments passed to the python script PFSSH.py.  

- Argument 1 is a string that turns on or off velocity scaling that is ordinarily a part of the fewest switches surface hopping approach.  Use `Scale_False` if you DO NOT want to scale the velocity and `Scale_True` if you DO want to scale the velocity.  `Scale_True` will be consistent with the ordinary formulation of surface hopping.

- Argument 2 is a string that keeps or discards the imaginary part of the derivative coupling.  Use `Complex_DC` to keep the imaginary part (i.e. use the full complex derivative coupling), and use `Real_DC` to disgard the imaginary part (i.e. use only the real part of the derivative coupling).

- Argument 3 is an integer that specifies the number of independant trajectories to run; since the value is 1 in the above example, only 1 trajectory will be run.

- Argument 4 is a float that specifies the dissipation rate of the photonic mode in mili electron volts; in the example the dissipation rate is 1.0 meV

- Argument 5 is a float that specifies the energy of the photonic mode in electron volts; in the example the energy is 2.45 eV

- Argument 6 is a float that specifies the coupling strength between the molecule and the photonic mode in electron volts; in the example the coupling energy is 0.02 eV

- Argument 7 is a file-name prefix that will be used for naming files that verbose output is written to; in the example output filenames will start with 'test'.

After the 5th argument, there is a re-direct statement that re-directs the STDOUT to a file called 'output.txt', and the terminating ampersand runs the program in the background.

### More Details

- If you run just a single trajectory (Argument 1 has the value '1'), then the program will automatically write the nuclear trajectory and the electronic populations (in both local and polaritonic bases) in data files in the 'Data/' folder.  In particular, the
nuclear trajectories will be in a file named '$prefix_nuc_traj.txt' and the electronic populations will be in a file named '$prefix_electronic.txt'

- You can run multiple independant trajectories (in serial, meaning one after another) by giving Argument 1 a larger value... there is in principle no limit, but each time you increase by 1 the time will increase by ~2.5 hours.  If you run more than 1 trajectory in serial, the trajectory information will not be written to file.

- The program will always write the potential energy surfaces (eigenvalues as a function of $R$ of the total Hamiltonian) and
the photonic contributions (which derive from the eigenvectors of the total Hamiltonian) regardless of how many trajectories you run.


