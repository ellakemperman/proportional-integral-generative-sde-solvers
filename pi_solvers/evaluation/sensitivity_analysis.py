import subprocess
import numpy as np

# Produces the data necessary to construct Figure 5, used by gaussian_grid_plots.py to construct figure
def call(name, params):
    for param in params:
        subprocess.run(["gaussian-testing", "10", "200", "0.001", "1", "--non_adaptive_ref",
                        "../../data/gaussian_experiment/complex_pistatic/data.csv", "-o", f"../data/gaussian_experiment/grid/{name}_complex_test/{param}",
                        "-g", "complex", f"--{name}", f"{param}", "--max_iter", "200", "-e", "--h_start", "0.15"
                        ])


if __name__ == "__main__":
    name = "ki"
    params = np.exp(np.linspace(np.log(0.01), np.log(1), 20))
    call(name, params)

    name = "kp"
    params = np.exp(np.linspace(np.log(0.01), np.log(1), 20))
    call(name, params)

    name = "tau_r"
    params = np.exp(np.linspace(np.log(0.01), np.log(5), 20))
    call(name, params)

    name = "alpha"
    params = np.linspace(0, 1, 20)
    call(name, params)

    name = "h_start"
    params = np.linspace(0.01, 0.2, 20)
    for param in params:
        subprocess.run(["gaussian-testing", "10", "200", "0.001", "1", "--non_adaptive_ref",
                        "../../data/gaussian_experiment/complex_pistatic/data.csv", "-o", f"../data/gaussian_experiment/grid/{name}_complex_test/{param}",
                        "-g", "complex", "--max_iter", "200", "-e", "--h_start", f"{param}"
                        ])