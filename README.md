# Quantum Option Pricing (CSCI 5244 Final Project)

## 📌 Project Overview

This repository contains the final project for **CSCI 5244: Quantum Computation and Information**.

The goal of this project is to reproduce the methodology of the paper **"Option Pricing using Quantum Computers"** (Stamatopoulos et al., 2019). We implement an end-to-end workflow—from financial theory to quantum hardware execution—using Qiskit.

The project benchmarks **Quantum Amplitude Estimation (QAE)** against **Classical Monte Carlo** methods to investigate the theoretical quadratic speedup in pricing European Call Options.

## 📂 Repository Structure

* `Option_Pricing.ipynb`: The main notebook containing:
    1. **Financial Model:** Setup of Black-Scholes parameters and Log-Normal distribution.
    2. **Classical Benchmark:** A full Monte Carlo simulation implemented in NumPy.
    3. **Quantum Simulation:** Qiskit implementation of QAE.
    4. **Hardware Execution:** Code to run simplified circuits on real IBM Quantum hardware.
    5. **Analysis:** Convergence plots comparing $O(M^{-1})$ vs $O(M^{-1/2})$ scaling.
* `environment.yml`: Configuration file to replicate the exact Python environment (Conda).

## 🎯 Project Objectives

### 1. Theory & Implementation

**Goal:** Map the spot price of an asset to a quantum state using a `LogNormalDistribution` circuit and encode the option payoff into qubit amplitudes.

### 2. Convergence Analysis (Simulation)

**Goal:** Compare the error scaling of the two methods to demonstrate the theoretical quantum advantage:

* **Classical Monte Carlo:** Expected error scaling $\epsilon \propto \frac{1}{\sqrt{M}}$
* **Quantum Amplitude Estimation:** Expected error scaling $\epsilon \propto \frac{1}{M}$ (Quadratic Speedup)

### 3. Hardware Limitations

**Goal:** Execute a simplified version of the algorithm on the **IBM Quantum** backend to analyze the impact of noise (decoherence and gate errors) on deep amplitude estimation circuits in the NISQ era.

## ⚙️ Setup & Installation

We use **Conda** to ensure a stable, reproducible environment using the provided `environment.yml` file.

1. **Clone the repository:**

    ```bash
    git clone [https://github.com/Astatium5/Quantum-Option-Pricing.git](https://github.com/Astatium5/Quantum-Option-Pricing.git)
    cd Quantum-Option-Pricing
    ```

2. **Create the environment:**
    This command reads `environment.yml`, installs Python 3.11, and sets up all required libraries automatically.

    ```bash
    conda env create -f environment.yml
    ```

3. **Activate the environment:**

    ```bash
    conda activate quantum
    ```

## 🔑 Configuration (API Key)

To run the hardware execution section, you must provide your IBM Quantum API Token securely.

1. Create a file named `.env` in the root directory of this project.
2. Add your token to the file in the following format:

    ```text
    IBM_QUANTUM_KEY=your_actual_api_key_here
    IBM_QUANTUM_INSTANCE=your_ibm_quantum_instance_name_here
    ```

## 🚀 How to Run

1. **Launch Jupyter:**
    Make sure your `quantum` environment is active.

    ```bash
    jupyter notebook
    ```

2. **Run the Analysis:**
    Open `Option_Pricing.ipynb` and run all cells.

    * **Note for Google Colab Users:** The notebook includes a `!pip install` cell at the top. Uncomment and run this cell if you are using Colab instead of a local Conda environment.*

## 📚 References

* [1] Stamatopoulos, N., et al. "Option Pricing using Quantum Computers." *arXiv preprint arXiv:1905.02666* (2019).
* [2] Qiskit Finance Tutorials: [Option Pricing](https://qiskit-community.github.io/qiskit-finance/tutorials/01_european_call_option_pricing.html)

## 👥 Team Members

* **Dima Golubenko** - Quantum Implementation Specialist
* **Navnith Bharadwaj** - Classical & Analysis Specialist
