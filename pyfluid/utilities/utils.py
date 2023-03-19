import os
import numpy as np

def check_convergence(n, tolerance, *args):
    norms = []
    for i in range(0, len(args), 2):
        diff_norm = np.linalg.norm(args[i] - args[i+1])
        norms.append(diff_norm)

    # Print the norms every 100 steps
    if n % 100 == 0:
        print(f"Step {n}: ", end="")
        for i, norm in enumerate(norms):
            print(f"diff_norm_{i} = {norm}", end=", " if i < len(norms) - 1 else "\n")

    # Check if all norms fall below the convergence tolerance
    if all(norm < tolerance for norm in norms) and n != 0:
        print(f"Converged after {n} time steps")
        return True

    return False

def save_fields(step, **fields):
    output_dir = './output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for field_name, field_data in fields.items():
        output_path = os.path.join(output_dir, f"{field_name}_{step}.npy")
        np.save(output_path, field_data)