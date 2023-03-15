"""
Solves the 2D Navier-Stokes equations for incompressible flow with constant viscosity using a finite-difference method.

The first-order derivatives are approximated using a central difference scheme, 
and the second-order derivatives are approximated using a second-order central difference scheme.

Parameters:
    - ic (str): the initial condition for the velocity field ('zero', 'constant', or 'sinusoidal')
    - bc (str): the boundary condition for the velocity field ('zero', 'constant', or 'no_slip')
    - nx (int): the number of grid points in the x direction
    - ny (int): the number of grid points in the y direction
    - Lx (float): the length of the domain in the x direction
    - Ly (float): the length of the domain in the y direction
    - nu (float): the viscosity of the fluid
    - nt (int): the number of time steps to take
    - dt (float): the time step size
    
Returns:
    - u (ndarray): the x-component of the velocity field
    - v (ndarray): the y-component of the velocity field
"""

import numpy as np
from utilities.initial_condition import initial_condition
from utilities.boundary_condition import boundary_condition

def solve_2d_navier_stokes(ic, bc, nx, ny, Lx, Ly, nu, nt, dt):

    dx = Lx/(nx-1)   # Grid spacing in x
    dy = Ly/(ny-1)   # Grid spacing in y

    # Initialize velocity field
    u, v = initial_condition(ic, nx, ny, Lx, Ly)

    # Main loop
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                u[i, j] = un[i, j] - un[i, j]*(dt/dx)*(un[i, j] - un[i, j-1]) - \
                          vn[i, j]*(dt/dy)*(un[i, j] - un[i-1, j]) + \
                          nu*(dt/dx**2)*(un[i, j+1] - 2*un[i, j] + un[i, j-1]) + \
                          nu*(dt/dy**2)*(un[i+1, j] - 2*un[i, j] + un[i-1, j])
                v[i, j] = vn[i, j] - un[i, j]*(dt/dx)*(vn[i, j] - vn[i, j-1]) - \
                          vn[i, j]*(dt/dy)*(vn[i, j] - vn[i-1, j]) + \
                          nu*(dt/dx**2)*(vn[i, j+1] - 2*vn[i, j] + vn[i, j-1]) + \
                          nu*(dt/dy**2)*(vn[i+1, j] - 2*vn[i, j] + vn[i-1, j])

        # Apply boundary conditions
        u, v = boundary_condition(bc, u, v)

    return u, v

def solve_2d_navier_stokes_p(ic, bc, nx, ny, Lx, Ly, nu, nt, dt):

    dx = Lx/(nx-1)   # Grid spacing in x
    dy = Ly/(ny-1)   # Grid spacing in y

    rho = 1 # Set constant density for simplicity reason

    # Initialize velocity field and pressure
    u, v = initial_condition(ic, nx, ny, Lx, Ly)
    p = np.zeros((ny, nx))

    # Main loop
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        pn = p.copy()
        for i in range(1, ny-1):
            for j in range(1, nx-1):

                # Poisson equation for the pressure field:
                # ∇^2 P = (1/ρ) (∂u/∂x)^2 + 2 (∂u/∂y)(∂v/∂x) + (∂v/∂y)^2
                p[i, j] = (pn[i, j+1] + pn[i, j-1])*dy**2 + \
                          (pn[i+1, j] + pn[i-1, j])*dx**2 - \
                          (dx**2 * dy**2) / (2 * (dx**2 + dy**2)) * \
                          (1 / dt * ((u[i, j+1] - u[i, j-1])/(2*dx) + (v[i+1, j] - v[i-1, j])/(2*dy)) - \
                          ((u[i, j+1] - u[i, j-1])/(2*dx))**2 - \
                          2*((u[i+1, j] - u[i-1, j])/(2*dy)*(v[i, j+1] - v[i, j-1])/(2*dx)) - \
                          ((v[i+1, j] - v[i-1, j])/(2*dy))**2)

                # Compute velocity field
                u[i, j] = un[i, j] - un[i, j]*(dt/dx)*(un[i, j] - un[i, j-1]) - \
                          vn[i, j]*(dt/dy)*(un[i, j] - un[i-1, j]) - \
                          (dt/(2*rho*dx))*(p[i, j+1] - p[i, j-1]) + \
                          nu*(dt/dx**2)*(un[i, j+1] - 2*un[i, j] + un[i, j-1]) + \
                          nu*(dt/dy**2)*(un[i+1, j] - 2*un[i, j] + un[i-1, j])
                v[i, j] = vn[i, j] - un[i, j]*(dt/dx)*(vn[i, j] - vn[i, j-1]) - \
                          vn[i, j]*(dt/dy)*(vn[i, j] - vn[i-1, j]) - \
                          (dt/(2*rho*dy))*(p[i+1, j] - p[i-1, j]) + \
                          nu*(dt/dx**2)*(vn[i, j+1] - 2*vn[i, j] + vn[i, j-1]) + \
                          nu*(dt/dy**2)*(vn[i+1, j] - 2*vn[i, j] + vn[i-1, j])

        # Apply boundary conditions
        u, v = boundary_condition(bc, u, v)

    return u, v, p