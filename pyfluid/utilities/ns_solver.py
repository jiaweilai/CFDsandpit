"""
Solves the 2D Navier-Stokes equations for incompressible flow with constant viscosity using a finite-difference method.
Forward Euler method for time advancing.

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
from utilities.initial_condition import initial_condition, initial_condition_rb
from utilities.boundary_condition import boundary_condition, boundary_condition_rb
from utilities.utils import save_fields, check_convergence

def solve_2d_navier_stokes(ic, bc, nx, ny, Lx, Ly, nu, nt, dt):

    # Convergence tolerance
    tolerance = 1e-6

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

        # Output data
        # Save fields every 1000 steps
        if n % 1000 == 0:
            save_fields(n, u=u, v=v)

        # Check convergence
        if check_convergence(n, tolerance, u, un, v, vn):
            break

        # Apply boundary conditions
        u, v = boundary_condition(bc, u, v)

    return u, v

def solve_2d_navier_stokes_p(ic, bc, nx, ny, Lx, Ly, nu, nt, dt):

    # Convergence tolerance
    tolerance = 1e-6

    dx = Lx/(nx-1)   # Grid spacing in x
    dy = Ly/(ny-1)   # Grid spacing in y

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
                          (dt/(2*dx))*(p[i, j+1] - p[i, j-1]) + \
                          nu*(dt/dx**2)*(un[i, j+1] - 2*un[i, j] + un[i, j-1]) + \
                          nu*(dt/dy**2)*(un[i+1, j] - 2*un[i, j] + un[i-1, j])
                v[i, j] = vn[i, j] - un[i, j]*(dt/dx)*(vn[i, j] - vn[i, j-1]) - \
                          vn[i, j]*(dt/dy)*(vn[i, j] - vn[i-1, j]) - \
                          (dt/(2*dy))*(p[i+1, j] - p[i-1, j]) + \
                          nu*(dt/dx**2)*(vn[i, j+1] - 2*vn[i, j] + vn[i, j-1]) + \
                          nu*(dt/dy**2)*(vn[i+1, j] - 2*vn[i, j] + vn[i-1, j])

        # Output data
        # Save fields every 1000 steps
        if n % 1000 == 0:
            save_fields(n, u=u, v=v, p=p)

        # Check convergence
        if check_convergence(n, tolerance, u, un, v, vn, p, pn):
            break

        # Apply boundary conditions
        u, v = boundary_condition(bc, u, v)

    return u, v, p

def solve_2d_navier_stokes_p_vectorized(ic, bc, nx, ny, Lx, Ly, nu, nt, dt):

    # Convergence tolerance
    tolerance = 1e-6

    dx = Lx/(nx-1)   # Grid spacing in x
    dy = Ly/(ny-1)   # Grid spacing in y

    # Initialize velocity field and pressure
    u, v = initial_condition(ic, nx, ny, Lx, Ly)
    p = np.zeros((ny, nx))

    # Define indices for interior points
    I, J = np.arange(1, ny-1)[:, np.newaxis], np.arange(1, nx-1)
    Ip, Im, Jp, Jm = I+1, I-1, J+1, J-1

    # Main loop
    for n in range(nt+1):
        un = u.copy()
        vn = v.copy()
        pn = p.copy()

        # Poisson equation for the pressure field
        p[I, J] = ((pn[I, Jp]*dy**2 + pn[I, Jm]*dy**2 +
                    pn[Ip, J]*dx**2 + pn[Im, J]*dx**2) -
                   (dx**2 * dy**2) / (2 * (dx**2 + dy**2)) *
                   (1 / dt * ((u[I, Jp] - u[I, Jm])/(2*dx) + (v[Ip, J] - v[Im, J])/(2*dy)) -
                    ((u[I, Jp] - u[I, Jm])/(2*dx))**2 -
                    2*((u[Ip, J] - u[Im, J])/(2*dy)*(v[I, Jp] - v[I, Jm])/(2*dx)) -
                    ((v[Ip, J] - v[Im, J])/(2*dy))**2))

        # Compute velocity field
        u[I, J] = (un[I, J] - un[I, J]*(dt/dx)*(un[I, J] - un[I, Jm]) -
                   vn[I, J]*(dt/dy)*(un[I, J] - un[Im, J]) -
                   (dt/(2*dx))*(p[I, Jp] - p[I, Jm]) +
                   nu*(dt/dx**2)*(un[I, Jp] - 2*un[I, J] + un[I, Jm]) +
                   nu*(dt/dy**2)*(un[Ip, J] - 2*un[I, J] + un[Im, J]))

        v[I, J] = (vn[I, J] - un[I, J]*(dt/dx)*(vn[I, J] - vn[I, Jm]) -
                   vn[I, J]*(dt/dy)*(vn[I, J] - vn[Im, J]) -
                   (dt/(2*dy))*(p[Ip, J] - p[Im, J]) +
                   nu*(dt/dx**2)*(vn[I, Jp] - 2*vn[I, J] + vn[I, Jm]) +
                   nu*(dt/dy**2)*(vn[Ip, J] - 2*vn[I, J] + vn[Im, J]))

        # Output data
        # Save fields every 1000 steps
        if n % 1000 == 0:
            save_fields(n, u=u, v=v, p=p)

        # Check convergence
        if check_convergence(n, tolerance, u, un, v, vn, p, pn):
            break

        # Apply boundary conditions
        u, v = boundary_condition(bc, u, v)

    return u, v, p

def solve_2d_navier_stokes_T_vectorized(ic, bc, nx, ny, Lx, Ly, nu, alpha, kappa, nt, dt):
    """
    Including temperature variations. 
    The Boussinesq approximation to account density variations in the buoyancy term.
    """

    # Convergence tolerance
    tolerance = 1e-6

    # Gravitational field
    g = 9.81

    dx = Lx/(nx-1)   # Grid spacing in x
    dy = Ly/(ny-1)   # Grid spacing in y

    # Initialize velocity field and pressure
    u, v, T = initial_condition_rb(ic, nx, ny, Lx, Ly)
    p = np.zeros((ny, nx))

    # Define indices for interior points
    I, J = np.arange(1, ny-1)[:, np.newaxis], np.arange(1, nx-1)
    Ip, Im, Jp, Jm = I+1, I-1, J+1, J-1

    # Main loop
    for n in range(nt+1):
        un = u.copy()
        vn = v.copy()
        Tn = T.copy()
        pn = p.copy()

        # Poisson equation for the pressure field (with Boussinesq approximation)
        p[I, J] = ((pn[I, Jp]*dy**2 + pn[I, Jm]*dy**2 +
                    pn[Ip, J]*dx**2 + pn[Im, J]*dx**2) -
                   (dx**2 * dy**2) / (2 * (dx**2 + dy**2)) *
                   (1 / dt * ((u[I, Jp] - u[I, Jm])/(2*dx) + (v[Ip, J] - v[Im, J])/(2*dy)) -
                    ((u[I, Jp] - u[I, Jm])/(2*dx))**2 -
                    2*((u[Ip, J] - u[Im, J])/(2*dy)*(v[I, Jp] - v[I, Jm])/(2*dx)) -
                    ((v[Ip, J] - v[Im, J])/(2*dy))**2) -
                   (alpha * g * (T[Ip, J] - T[Im, J]) * dy**2 / (2 * (dx**2 + dy**2))))


        # Compute velocity field (with Boussinesq approximation)
        # Update velocity fields with buoyancy term
        u[I, J] = un[I, J] - un[I, J] * (dt / dx) * (un[I, J] - un[I, J - 1]) - vn[I, J] * (dt / dy) * (un[I, J] - un[I - 1, J]) \
                  - (dt / (2 * dx)) * (p[I, J + 1] - p[I, J - 1]) \
                  + nu * (dt / dx ** 2) * (un[I, J + 1] - 2 * un[I, J] + un[I, J - 1]) \
                  + nu * (dt / dy ** 2) * (un[I + 1, J] - 2 * un[I, J] + un[I - 1, J])

        v[I, J] = vn[I, J] - un[I, J] * (dt / dx) * (vn[I, J] - vn[I, J - 1]) - vn[I, J] * (dt / dy) * (vn[I, J] - vn[I - 1, J]) \
                    - (dt / (2 * dy)) * (p[I + 1, J] - p[I - 1, J]) \
                    + nu * (dt / dx ** 2) * (vn[I, J + 1] - 2 * vn[I, J] + vn[I, J - 1]) \
                    + nu * (dt / dy ** 2) * (vn[I + 1, J] - 2 * vn[I, J] + vn[I - 1, J]) \
                    + alpha * g * dt * (Tn[I, J] - Tn[I - 1, J])  # Add buoyancy term

        # Update temperature field
        T[I, J] = Tn[I, J] - un[I, J] * (dt / dx) * (Tn[I, J] - Tn[I, J - 1]) - vn[I, J] * (dt / dy) * (Tn[I, J] - Tn[I - 1, J]) \
                  + kappa * (dt / dx ** 2) * (Tn[I, J + 1] - 2 * Tn[I, J] + Tn[I, J - 1]) \
                  + kappa * (dt / dy ** 2) * (Tn[I + 1, J] - 2 * Tn[I, J] + Tn[I - 1, J])

        # Output data
        # Save fields every 1000 steps
        if n % 1000 == 0:
            save_fields(n, u=u, v=v, p=p, T=T)

        # Check convergence
        if check_convergence(n, tolerance, u, un, v, vn, p, pn, T, Tn):
            break

        # Apply boundary conditions
        u, v, T = boundary_condition_rb(bc, u, v, T)

    return u, v, p, T