import numpy as np
import matplotlib.pyplot as plt

from utilities.ns_solver import solve_2d_navier_stokes, solve_2d_navier_stokes_p

if __name__ == '__main__':
    # Parameters
    nx = 51    # Number of grid points in x
    ny = 51    # Number of grid points in y
    Lx = 1.0   # Length of the domain in x
    Ly = 1.0   # Length of the domain in y

    nu = 0.1   # Viscosity
    nt = 1000   # Number of time steps
    dt = 0.0001   # Time step size

    # Main loop
    #u, v = solve_2d_navier_stokes('constant', 'constant', nx, ny, Lx, Ly, nu, nt, dt)
    u, v, p = solve_2d_navier_stokes_p('constant', 'constant', nx, ny, Lx, Ly, nu, nt, dt)

    # Plot the results
    # Create 2D grid of x and y values
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Plot velocity field
    axs[0].quiver(X, Y, u, v)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Velocity field')

    # Plot velocity magnitude
    axs[1].contourf(X, Y, np.sqrt(u**2 + v**2))
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].set_title('Velocity magnitude')

    # Plot pressure field
    cp = axs[2].contourf(X, Y, p)
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    axs[2].set_title('Pressure field')
    plt.colorbar(cp)

    plt.show()
