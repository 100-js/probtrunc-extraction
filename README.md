## Installation

To reproduce the experimental results reported in the paper as faithfully as possible, it is recommended to match the environment to the package versions below.

- `numpy==2.0.2`
- `scikit-learn==1.6.1`
- `scipy==1.16.3`
- `sympy==1.14.0`
- `fpylll==0.6.4`
- `cysignals==1.12.6`
- `matplotlib==3.10.0`
- `torch==2.11.0`

Install them with:

```bash
pip install -r requirements.txt
```

## Run

All scripts are configured with default arguments that exactly reproduce the data and conditions reported in the paper. 

### 1. Model Extraction Attack Simulation
Run the following command to execute the full model extraction pipeline. The results correspond to the **Experimental Results** section of the paper.

```bash
python model_extraction.py
```

### 2. Spectral Bound and Error Projection Simulation
Run the following commands to execute the structural noise and bounds simulation. You must specify the scenario as either `random` (for random matrices) or `trained` (for empirically trained models). The exported data and distributions correspond to the **Empirical $E_{\max}$ Estimation** section in the Appendix.

```bash
# For the random matrices scenario
python emax_simulator.py random

# For the empirically trained models scenario
python emax_simulator.py trained
```
