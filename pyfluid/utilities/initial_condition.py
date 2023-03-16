"""
Initial velocity field for the 2D Navier-Stokes solver

Parameters:

    - ic: string specifying the initial condition. Can be one of the following:
    - nx: number of grid points in the x-direction.
    - ny: number of grid points in the y-direction.
    - Lx: length of the domain in the x-direction.
    - Ly: length of the domain in the y-direction.

Returns:

    - u: numpy array of shape (ny, nx) representing the x-component of the initial velocity field.
    - v: numpy array of shape (ny, nx) representing the y-component of the initial velocity field.

Initial Conditions

    - 'constant': uniform velocity field with magnitude 1 in the central region of the domain.
    - 'gaussian': velocity field with a Gaussian distribution centered at the center of the domain.
    - 'wave_packet': velocity field with a wave packet centered at a specified location in the domain.
    - 'shear_flow': velocity field with a constant gradient in the x-direction and a constant offset in the y-direction.
    - 'double_vortex': velocity field consisting of two counter-rotating vortices.
    - 'obstacle_flow': velocity field with an obstacle in the center of the domain, causing flow separation and vortex shedding.
    - 'kelvin_helmholtz': velocity field with a vortex sheet separating two regions of fluid with different velocities.
"""

import numpy as np

# Initial conditions
def initial_condition(ic, nx, ny, Lx, Ly):
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    if ic == 'constant':
        u, v = np.zeros((ny, nx)), np.zeros((ny, nx))
        u[int(ny/4):int(3*ny/4), int(nx/4):int(3*nx/4)] = 1
        v[int(ny/4):int(3*ny/4), int(nx/4):int(3*nx/4)] = 1
    elif ic == 'constant2':
        u, v = np.zeros((ny, nx)), np.zeros((ny, nx))
        u = np.ones((ny, nx)) * 1
    elif ic == 'gaussian':
        u, v = np.zeros((ny, nx)), np.zeros((ny, nx))
        r = np.sqrt((X-Lx/2)**2 + (Y-Ly/2)**2)
        u = np.exp(-r**2/0.01)
    elif ic == 'wave_packet':
        u, v = np.zeros((ny, nx)), np.zeros((ny, nx))
        kx, ky = 2*np.pi/Lx, 2*np.pi/Ly
        Xc, Yc = Lx/4, Ly/4
        u = np.exp(-(X-Xc)**2/0.01)*np.cos(kx*(X-Xc) + ky*(Y-Yc))
        v = np.exp(-(X-Xc)**2/0.01)*np.sin(kx*(X-Xc) + ky*(Y-Yc))
    elif ic == 'shear_flow':
        u, v = np.zeros((ny, nx)), np.zeros((ny, nx))
        u[:, :int(nx/2)] = 1
        v = np.ones((ny, nx)) * 0.1
    elif ic == 'double_vortex':
        u, v = np.zeros((ny, nx)), np.zeros((ny, nx))
        r = np.sqrt((X-Lx/4)**2 + (Y-Ly/4)**2) + np.sqrt((X-3*Lx/4)**2 + (Y-3*Ly/4)**2)
        u = 1 - np.exp(-r**2/0.01)
        v = np.roll(u, shift=-1, axis=0) - np.roll(u, shift=1, axis=0)
    elif ic == 'obstacle_flow':
        u, v = np.zeros((ny, nx)), np.zeros((ny, nx))
        r = np.sqrt((X-Lx/2)**2 + (Y-Ly/2)**2)
        u = np.where(r<0.2, 0, np.where(r<0.3, np.sin(np.pi*(r-0.2)/0.1), 1))
        v = np.where(r<0.2, 0, np.where(r<0.3, -np.cos(np.pi*(r-0.2)/0.1)*np.pi/0.1, 0))

    elif ic == 'kelvin_helmholtz':
        u, v = np.zeros((ny, nx)), np.zeros((ny, nx))
        # Set the mid point
        mid_y = ny // 2
        
        # Set different velocities for top and bottom layers
        u[:mid_y, :] = 0.5
        u[mid_y:, :] = -0.5

        # Add a small perturbation at the interface
        x = np.linspace(0, Lx, nx)
        perturbation = 0.1 * np.sin(2 * np.pi * x / Lx)
        u[mid_y-1, :] += perturbation
        u[mid_y, :] += perturbation

    else:
        raise ValueError(f"Invalid initial condition: {ic}")

    return u, v