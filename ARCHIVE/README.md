# ARCHIVE

This folder holds the pre-refactor PyCQED code, notebooks, plotting scripts and data, unchanged. They were moved here from the repository root so it holds only the new `pycqed/` package. The layout matches the old root, so relative paths such as `from polaritonic import polaritonic` and `Data/...` still work when you run from this folder:

```bash
cd ARCHIVE
python PFSSH.py Scale_True Real_DC 1 1.0 2.45 0.02 test
```

`tests/test_sampling_legacy_cli.py` checks that `polaritonic.py` here and the new package produce the same Hamiltonian.

Known problems in this code are listed in the main README. In short: photon decay is lost from the dynamics after commit 98aa224, hops from UP to the ground state are never taken, and the velocity rescaling does not conserve energy. Use `pycqed/` for new work.
