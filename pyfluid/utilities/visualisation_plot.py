import numpy as np
import matplotlib.pyplot as plt

def plot_uvp_xy(u, v, p, Lx, Ly, nx, ny):
    # Create 2D grid of x and y values
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