import math


def raycast_3d_array(grid, origin, direction, max_distance=100.0):
    """
    Traverses a 3D grid array along a ray using the Amanatides & Woo DDA algorithm.

    :param grid: 3D list or numpy array where non-zero represents a solid voxel
    :param origin: Tuple/list of (x, y, z) starting floats
    :param direction: Tuple/list of (dx, dy, dz) direction floats (must be normalized)
    :param max_distance: Maximum distance to trace before stopping
    :return: Tuple of (voxel_coords, hit_normal) if hit, else (None, None)
    """
    # Grid dimensions
    dim_x = len(grid)
    dim_y = len(grid[0])
    dim_z = len(grid[0][0])

    # 1. Initialization Phase
    # Current voxel coordinate index
    X = math.floor(origin[0])
    Y = math.floor(origin[1])
    Z = math.floor(origin[2])

    # Step direction along each axis
    step_x = 1 if direction[0] > 0 else -1 if direction[0] < 0 else 0
    step_y = 1 if direction[1] > 0 else -1 if direction[1] < 0 else 0
    step_z = 1 if direction[2] > 0 else -1 if direction[2] < 0 else 0

    # Distance to travel along the ray to cross one full voxel along each axis
    t_delta_x = abs(1.0 / direction[0]) if direction[0] != 0 else float('inf')
    t_delta_y = abs(1.0 / direction[1]) if direction[1] != 0 else float('inf')
    t_delta_z = abs(1.0 / direction[2]) if direction[2] != 0 else float('inf')

    # Initial distance from origin to the first voxel boundary along each axis
    if step_x > 0:
        t_max_x = (math.floor(origin[0]) + 1.0 - origin[0]) * t_delta_x
    else:
        t_max_x = (origin[0] - math.floor(origin[0])) * t_delta_x

    if step_y > 0:
        t_max_y = (math.floor(origin[1]) + 1.0 - origin[1]) * t_delta_y
    else:
        t_max_y = (origin[1] - math.floor(origin[1])) * t_delta_y

    if step_z > 0:
        t_max_z = (math.floor(origin[2]) + 1.0 - origin[2]) * t_delta_z
    else:
        t_max_z = (origin[2] - math.floor(origin[2])) * t_delta_z

    # Tracks which face was hit to determine the normal vector
    hit_normal = (0, 0, 0)

    # 2. Incremental Traversal Phase
    while True:
        # Out of bounds check
        if X < 0 or X >= dim_x or Y < 0 or Y >= dim_y or Z < 0 or Z >= dim_z:
            return None, None

        # Collision check (assuming 0 is empty space)
        if grid[X][Y][Z] != 0:
            return (X, Y, Z), hit_normal

        # Determine the next axis to step along based on the smallest t_max
        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                if t_max_x > max_distance: break
                t_max_x += t_delta_x
                X += step_x
                hit_normal = (-step_x, 0, 0)  # Hit the X boundary face
            else:
                if t_max_z > max_distance: break
                t_max_z += t_delta_z
                Z += step_z
                hit_normal = (0, 0, -step_z)  # Hit the Z boundary face
        else:
            if t_max_y < t_max_z:
                if t_max_y > max_distance: break
                t_max_y += t_delta_y
                Y += step_y
                hit_normal = (0, -step_y, 0)  # Hit the Y boundary face
            else:
                if t_max_z > max_distance: break
                t_max_z += t_delta_z
                Z += step_z
                hit_normal = (0, 0, -step_z)  # Hit the Z boundary face

    return None, None