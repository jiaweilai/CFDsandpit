"""
Function to apply boundary conditions to the velocity field.

Parameters:

    - bc (str): Type of boundary condition to apply. Can be 'constant', 'periodic', 'slip', or 'inflow_outflow'.
    - u (ndarray): Velocity in the x-direction.
    - v (ndarray): Velocity in the y-direction.

Returns:

    - Updated velocity fields with boundary conditions applied.

Boundary conditions:

    - 'constant': Set the velocity at the boundary to zero.
    - 'periodic': Make the boundary periodic by wrapping the values around to the opposite side.
    - 'slip': Set the velocity normal to the boundary to zero and the velocity tangential to the boundary to the same value as the adjacent cell.
    - 'channel_flow': Set the velocity at the inflow boundary to a constant value and the velocity at the outflow boundary to zero.

"""

def boundary_condition(bc, u, v):
    if bc == 'constant':
        u[0, :] = 0
        u[-1, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0
    elif bc == 'periodic':
        u[0, :] = u[-2, :]
        u[-1, :] = u[1, :]
        u[:, 0] = u[:, -2]
        u[:, -1] = u[:, 1]
        v[0, :] = v[-2, :]
        v[-1, :] = v[1, :]
        v[:, 0] = v[:, -2]
        v[:, -1] = v[:, 1]
    elif bc == 'slip':
        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
        v[0, :] = -v[1, :]
        v[-1, :] = -v[-2, :]
        v[:, 0] = v[:, 1]
        v[:, -1] = v[:, -2]
    elif bc == 'channel_flow':
        u[0, :] = 1         # Inflow at the left boundary
        u[-1, :] = u[-2, :] # Outflow at the right boundary, set the gradient normal to the boundary to be zero
        u[:, 0] = 0         # No-slip at the bottom boundary
        u[:, -1] = 0        # No-slip at the top boundary
        v[0, :] = 0         # Zero v-velocity at the left boundary
        v[-1, :] = v[-2, :] # Zero v-velocity at the right boundary, set the gradient normal to the boundary to be zero
        v[:, 0] = 0         # No-slip at the bottom boundary
        v[:, -1] = 0        # No-slip at the top boundary
    elif bc == 'kelvin_helmholtz':
        """periodic boundary conditions in the x direction and no-slip boundary conditions 
         in the vertical y direction. """
        u[0, :] = u[-2, :]   # Periodic BC in x-direction
        u[-1, :] = u[1, :]
        u[:, 0] = 0           # No-slip BC in y-direction
        u[:, -1] = 0

        v[0, :] = v[-2, :]   # Periodic BC in x-direction
        v[-1, :] = v[1, :]
        v[:, 0] = 0           # No-slip BC in y-direction
        v[:, -1] = 0
    else:
        raise ValueError(f"Invalid boundary condition: {bc}")

    return u, v

def boundary_condition_rb(bc, u, v, T):
    if bc == 'no_slip':
        # No-slip condition for u and v
        u[0, :] = 0
        u[-1, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0

        # Dirichlet condition for temperature
        T[:, 0] = 1  # Hot bottom boundary
        T[:, -1] = 0  # Cold top boundary
        T[0, :] = T[1, :]  # Insulated left boundary
        T[-1, :] = T[-2, :]  # Insulated right boundary
    else:
        raise ValueError(f"Invalid boundary condition: {bc}")

    return u, v, T