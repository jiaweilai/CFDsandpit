import os
import numpy as np

def save_fields(step, **fields):
    output_dir = './output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for field_name, field_data in fields.items():
        output_path = os.path.join(output_dir, f"{field_name}_{step}.npy")
        np.save(output_path, field_data)