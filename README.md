# PyCQED
Tools for simulating polaritonic chemical dynamics using the formalism of cavity quantum electrodynamics combined with surface hopping dynamics.

### Quickstart

- Clone the repository

- To run a single trajectory: `python PFSSH.py 1 1.0 2.45 0.02 test > output.txt &`

In the above example, there are 5 arguments passed to the python script PFSSH.py.  

- Argument 1 is an integer that specifies the number of independant trajectories to run; since the value is 1 in the above example, only 1 trajectory will be run.

- Argument 2 is a float that specifies the dissipation rate of the photonic mode in mili electron volts; in the example the dissipation rate is 1.0 meV

- Argument 3 is a float that specifies the energy of the photonic mode in electron volts; in the example the energy is 2.45 eV

- Argument 4 is a float that specifies the coupling strength between the molecule and the photonic mode in electron volts; in the example the coupling energy is 0.02 eV

- Argument 5 is a file-name prefix that will be used for naming files that verbose output is written to; in the example output filenames will start with 'test'.

After the 5th argument, there is a re-direct statement that re-directs the STDOUT to a file called 'output.txt', and the terminating ampersand runs the program in the background.


