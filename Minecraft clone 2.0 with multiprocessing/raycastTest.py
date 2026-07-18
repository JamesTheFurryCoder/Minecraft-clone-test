import math

# raycastTest.py
def get_direction_3d(point_a, point_b):
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    dz = point_b[2] - point_a[2]

    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length == 0:
        return None

    return dx / length, dy / length, dz / length

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

    def distance_to_next_boundary(origin_value, step, t_delta):
        if step == 0:
            return float('inf')
        if step > 0:
            return (math.floor(origin_value) + 1.0 - origin_value) * t_delta
        return (origin_value - math.floor(origin_value)) * t_delta

    # Initial distance from origin to the first voxel boundary along each axis.
    # Axes with no movement must stay unselectable; otherwise 0 * inf becomes nan.
    t_max_x = distance_to_next_boundary(origin[0], step_x, t_delta_x)
    t_max_y = distance_to_next_boundary(origin[1], step_y, t_delta_y)
    t_max_z = distance_to_next_boundary(origin[2], step_z, t_delta_z)

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
