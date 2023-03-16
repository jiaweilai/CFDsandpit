import os
import json
import numpy as np
import matplotlib.pyplot as plt

from utilities.ns_solver import solve_2d_navier_stokes, solve_2d_navier_stokes_p
from utilities.visualisation_plot import plot_uvp_xy

if __name__ == '__main__':

    # Get the script's directory
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the path to the JSON file in the "input" folder
    parameters_path = os.path.join(script_dir, "input", "parameters.json")

    # Read and load parameters from JSON file
    with open(parameters_path, "r") as file:
        parameters = json.load(file)

    # Assign parameters to variables
    nx = parameters["nx"] # Number of grid points in x
    ny = parameters["ny"] # Number of grid points in y
    Lx = parameters["Lx"] # Length of the domain in x
    Ly = parameters["Ly"] # Length of the domain in y
    nu = parameters["nu"] # Viscosity
    nt = parameters["nt"] # Number of time steps
    dt = parameters["dt"] # Time step size
    cfd_initsc = parameters["Initial_condition"] 
    cfd_boundc = parameters["Boundary_condition"]

    # Load u,v,p from binary files
    u = np.load("velocity_u.npy")
    v = np.load("velocity_v.npy")
    p = np.load("pressure.npy")

    # Plot the results
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4))

    # Plot velocity field
    axs[0].quiver(X, Y, u.transpose(), v.transpose())
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Velocity field')

    # Plot velocity magnitude
    axs[1].contourf(X, Y, np.sqrt((u.transpose())**2 + (v.transpose())**2))
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].set_title('Velocity magnitude')

    # Plot pressure field
    cp = axs[2].contourf(X, Y, p.transpose())
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    axs[2].set_title('Pressure field')
    plt.colorbar(cp)

    plt.show()