import os
import json
import numpy as np
import matplotlib.pyplot as plt

from utilities.ns_solver import solve_2d_navier_stokes, solve_2d_navier_stokes_p, \
    solve_2d_navier_stokes_p_vectorized, solve_2d_navier_stokes_T_vectorized

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

    alpha = parameters["alpha"] # Time step size
    kappa = parameters["kappa"] # Time step size

    cfd_initsc = parameters["Initial_condition"] 
    cfd_boundc = parameters["Boundary_condition"]

    # Main loop
    #u, v = solve_2d_navier_stokes('constant', 'constant', nx, ny, Lx, Ly, nu, nt, dt)
    #u, v, p = solve_2d_navier_stokes_p(cfd_initsc, cfd_boundc, nx, ny, Lx, Ly, nu, nt, dt)
    #u, v, p = solve_2d_navier_stokes_p_vectorized(cfd_initsc, cfd_boundc, nx, ny, Lx, Ly, nu, nt, dt)
    u, v, p, T =  solve_2d_navier_stokes_T_vectorized(cfd_initsc, cfd_boundc, nx, ny, Lx, Ly, nu, alpha, kappa, nt, dt)
    # Save u,v,p to a binary file
    np.save("velocity_u.npy", u)
    np.save("velocity_v.npy", v)
    np.save("pressure.npy", p)

    # Plot the results
    plot_uvp_xy(u, v, T, Lx, Ly, nx, ny)
    plt.show()