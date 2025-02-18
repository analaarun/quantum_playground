# quantum_playground

A Streamlit-based interactive application for simulating and analyzing quantum dot systems with multiple nuclear spins coupled to a single electron spin.

## Overview

This application simulates a quantum dot system consisting of n nuclear spins interacting with one electron spin. It includes various quantum mechanical interactions and provides interactive visualizations of energy levels, time evolution, and quantum state representations.

## Features

- **Interactive Hamiltonian Configuration**: Adjust system parameters in real-time including:
  - Magnetic field strength (Bz)
  - Hyperfine coupling (A_hf)
  - Dipole-dipole coupling (J_dd)
  - Electron and nuclear gyromagnetic ratios
  - RF driving frequency and amplitude

- **Visualization Tools**:
  - Energy level diagrams with interactive parameter adjustment
  - Animated energy level evolution
  - Bloch sphere representations of quantum states
  - Time evolution plots of observables
  - Multi-view analysis of system dynamics

- **Advanced Features**:
  - Decoherence effects simulation
  - Custom time evolution settings
  - Multiple observable tracking options
  - Parameter sweep animations

## System Components

The simulator includes the following physical interactions:
1. Nuclear Zeeman interaction
2. Electron Zeeman interaction
3. Hyperfine coupling
4. Nuclear quadrupole coupling
5. Nuclear dipole-dipole interaction
6. RF driving field

## Requirements

- Python 3.7+
- streamlit
- qutip
- numpy
- matplotlib
- plotly
- pyperclip
- openai (optional, for AI assistance)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Acknowledgments

This simulator uses QuTiP (Quantum Toolbox in Python) for quantum mechanical calculations and Streamlit for the web interface.

## Contact

For questions and support, please open an issue in the repository.
