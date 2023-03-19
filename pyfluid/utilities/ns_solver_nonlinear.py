import numpy as np
from utilities.initial_condition import initial_condition, initial_condition_rb
from utilities.boundary_condition import boundary_condition, boundary_condition_rb
from utilities.utils import save_fields, check_convergence

def backward_euler_method(u, v, p, I, J, Ip, Im, Jp, Jm, nu, dt, dx, dy, max_iter=100, tol=1e-6):
    u_new = u.copy()
    v_new = v.copy()

    for _ in range(max_iter):
        u_prev = u_new.copy()
        v_prev = v_new.copy()

        # Poisson equation for the pressure field
        p[I, J] = ((p[I, Jp] * dy ** 2 + p[I, Jm] * dy ** 2 +
                    p[Ip, J] * dx ** 2 + p[Im, J] * dx ** 2) -
                   (dx ** 2 * dy ** 2) / (2 * (dx ** 2 + dy ** 2)) *
                   (1 / dt * ((u_prev[I, Jp] - u_prev[I, Jm]) / (2 * dx) + (v_prev[Ip, J] - v_prev[Im, J]) / (2 * dy)) -
                    ((u_prev[I, Jp] - u_prev[I, Jm]) / (2 * dx)) ** 2 -
                    2 * ((u_prev[Ip, J] - u_prev[Im, J]) / (2 * dy) * (v_prev[I, Jp] - v_prev[I, Jm]) / (2 * dx)) -
                    ((v_prev[Ip, J] - v_prev[Im, J]) / (2 * dy)) ** 2))

        # Compute velocity field using the backward Euler method
        u_new[I, J] = (u[I, J] - u_prev[I, J] * (dt / dx) * (u_prev[I, J] - u_prev[I, Jm]) -
                       v_prev[I, J] * (dt / dy) * (u_prev[I, J] - u_prev[Im, J]) -
                       (dt / (2 * dx)) * (p[I, Jp] - p[I, Jm]) +
                       nu * (dt / dx ** 2) * (u_prev[I, Jp] - 2 * u_prev[I, J] + u_prev[I, Jm]) +
                       nu * (dt / dy ** 2) * (u_prev[Ip, J] - 2 * u_prev[I, J] + u_prev[Im, J]))

        v_new[I, J] = (v[I, J] - u_prev[I, J] * (dt / dx) * (v_prev[I, J] - v_prev[I, Jm]) -
                       v_prev[I, J] * (dt / dy) * (v_prev[I, J] - v_prev[Im, J]) -
                       (dt / (2 * dy)) * (p[Ip, J] - p[Im, J]) +
                       nu * (dt / dx ** 2) * (v_prev[I, Jp] - 2 * v_prev[I, J] + v_prev[I, Jm]) +
                       nu * (dt / dy ** 2) * (v_prev[Ip, J] - 2 * v_prev[I, J] + v_prev[Im, J]))

        # Check convergence
        u_diff = np.linalg.norm(u_new - u_prev)
        v_diff = np.linalg.norm(v_new - v_prev)

        # Check convergence
        if u_diff < tol and v_diff < tol:
            break

    return u_new, v_new, p

def solve_nl_2d_navier_stokes_p_vectorized(ic, bc, nx, ny, Lx, Ly, nu, nt, dt):

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

         # Solve the nonlinear system using Backward Euler Method
        u_new, v_new, p_new = backward_euler_method(u, v, p, I, J, Ip, Im, Jp, Jm, nu, dt, dx, dy)

        # Output data
        # Save fields every 1000 steps
        if n % 1000 == 0:
            save_fields(n, u=u_new, v=v_new, p=p_new)

        # Check convergence
        if check_convergence(n, tolerance, u, u_new, v, v_new, p, p_new):
            break

        # Apply boundary conditions
        u, v = boundary_condition(bc, u, v)

    return u, v, p