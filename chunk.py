from ursina import *
import numpy as np

uvs = (
        (0, 0),
        (1, 0),
        (1, 1),
        (0, 1)
    )

def front_face(size=1, center=(0, 0, 0)):
    """Face pointing in the +Z direction."""
    x, y, z = center
    h = size / 2

    return [
        (x - h, y - h, z + h),  # bottom-left
        (x + h, y - h, z + h),  # bottom-right
        (x + h, y + h, z + h),  # top-right
        (x - h, y + h, z + h),  # top-left
    ]


def back_face(size=1, center=(0, 0, 0)):
    """Face pointing in the -Z direction."""
    x, y, z = center
    h = size / 2

    return [
        (x + h, y - h, z - h),
        (x - h, y - h, z - h),
        (x - h, y + h, z - h),
        (x + h, y + h, z - h),
    ]


def right_face(size=1, center=(0, 0, 0)):
    """Face pointing in the +X direction."""
    x, y, z = center
    h = size / 2

    return [
        (x + h, y - h, z + h),
        (x + h, y - h, z - h),
        (x + h, y + h, z - h),
        (x + h, y + h, z + h),
    ]


def left_face(size=1, center=(0, 0, 0)):
    """Face pointing in the -X direction."""
    x, y, z = center
    h = size / 2

    return [
        (x - h, y - h, z - h),
        (x - h, y - h, z + h),
        (x - h, y + h, z + h),
        (x - h, y + h, z - h),
    ]


def top_face(size=1, center=(0, 0, 0)):
    """Face pointing in the +Y direction."""
    x, y, z = center
    h = size / 2

    return [
        (x - h, y + h, z + h),
        (x + h, y + h, z + h),
        (x + h, y + h, z - h),
        (x - h, y + h, z - h),
    ]


def bottom_face(size=1, center=(0, 0, 0)):
    """Face pointing in the -Y direction."""
    x, y, z = center
    h = size / 2

    return [
        (x - h, y - h, z - h),
        (x + h, y - h, z - h),
        (x + h, y - h, z + h),
        (x - h, y - h, z + h),
    ]

# info for consistent triangles: triangles = [
#         0, 2, 1,
#         0, 3, 2
#     ]

class Chunk:
    def __init__(self, chunkCoordinate):
        self.chunkCoordinate = chunkCoordinate
        self.entityModel = None
        self.colliderModel = None
        self.uvs = []
        self.triangles = []
        self.vertices = []
        self.mesh_generated = False
        self.has_visible_mesh = False

        self.size = 32
        self.height = 40

    def destroy(self):
        if self.entityModel is not None:
            destroy(self.entityModel)
            self.entityModel = None

        self.has_visible_mesh = False

    def _get_block_dimensions(self, blocks):
        if isinstance(blocks, np.ndarray):
            return blocks.shape

        max_x = len(blocks)
        max_y = len(blocks[0]) if max_x else 0
        max_z = len(blocks[0][0]) if max_y else 0
        return max_x, max_y, max_z

    def getProperCoordinate(self,x,y,z):
        return (
            self.chunkCoordinate[0] * self.size + x,
            self.chunkCoordinate[1] * self.height + y,
            self.chunkCoordinate[2] * self.size + z
        )

    def checkBounds(self, globalBlockIds, x, y, z):
        max_x, max_y, max_z = self._get_block_dimensions(globalBlockIds)
        return 0 <= x < max_x and 0 <= y < max_y and 0 <= z < max_z

    def _is_fully_occluded(self, blocks, start_x, start_y, start_z, end_x, end_y, end_z, max_x, max_y, max_z):
        if (
            start_x <= 0 or start_y <= 0 or start_z <= 0
            or end_x >= max_x or end_y >= max_y or end_z >= max_z
        ):
            return False

        if isinstance(blocks, np.ndarray):
            return (
                blocks[start_x:end_x, start_y:end_y, start_z:end_z].all()
                and blocks[start_x:end_x, start_y:end_y, start_z - 1].all()
                and blocks[start_x:end_x, start_y:end_y, end_z].all()
                and blocks[start_x:end_x, start_y - 1, start_z:end_z].all()
                and blocks[start_x:end_x, end_y, start_z:end_z].all()
                and blocks[start_x - 1, start_y:end_y, start_z:end_z].all()
                and blocks[end_x, start_y:end_y, start_z:end_z].all()
            )

        for x in range(start_x, end_x):
            x_blocks = blocks[x]
            for y in range(start_y, end_y):
                if not all(x_blocks[y][start_z:end_z]):
                    return False

        before_z = start_z - 1
        after_z = end_z
        for x in range(start_x, end_x):
            x_blocks = blocks[x]
            for y in range(start_y, end_y):
                row = x_blocks[y]
                if row[before_z] == 0 or row[after_z] == 0:
                    return False

        before_y = start_y - 1
        after_y = end_y
        for x in range(start_x, end_x):
            x_blocks = blocks[x]
            bottom_row = x_blocks[before_y]
            top_row = x_blocks[after_y]
            for z in range(start_z, end_z):
                if bottom_row[z] == 0 or top_row[z] == 0:
                    return False

        before_x = start_x - 1
        after_x = end_x
        left_x_blocks = blocks[before_x]
        right_x_blocks = blocks[after_x]
        for y in range(start_y, end_y):
            left_row = left_x_blocks[y]
            right_row = right_x_blocks[y]
            for z in range(start_z, end_z):
                if left_row[z] == 0 or right_row[z] == 0:
                    return False

        return True

    def _apply_mesh(self):
        self.mesh_generated = True
        self.has_visible_mesh = bool(self.vertices)

        if not self.has_visible_mesh:
            self.destroy()
            return

        newMesh = Mesh(
                vertices=self.vertices,
                triangles=self.triangles,
                uvs=self.uvs
            )

        if self.entityModel is not None:
            self.entityModel.model = newMesh
            self.entityModel.model.generate()
        else:
            self.entityModel = Entity(model= newMesh,texture="white_cube")
            #self.colliderModel = MeshCollider(self.entityModel,mesh=self.entityModel.model)
            #self.entityModel.collider = self.colliderModel


    def generateMesh(self, globalBlockIds):
        mesh_uvs = []
        triangles = []
        vertices = []
        self.uvs = mesh_uvs
        self.triangles = triangles
        self.vertices = vertices

        blocks = globalBlockIds
        max_x, max_y, max_z = self._get_block_dimensions(blocks)

        raw_start_x = self.chunkCoordinate[0] * self.size
        raw_start_y = self.chunkCoordinate[1] * self.height
        raw_start_z = self.chunkCoordinate[2] * self.size
        start_x = max(raw_start_x, 0)
        start_y = max(raw_start_y, 0)
        start_z = max(raw_start_z, 0)
        end_x = min(raw_start_x + self.size, max_x)
        end_y = min(raw_start_y + self.height, max_y)
        end_z = min(raw_start_z + self.size, max_z)
        last_z = max_z - 1

        if start_x >= end_x or start_y >= end_y or start_z >= end_z:
            self._apply_mesh()
            return

        if self._is_fully_occluded(blocks, start_x, start_y, start_z, end_x, end_y, end_z, max_x, max_y, max_z):
            self._apply_mesh()
            return

        vertices_extend = vertices.extend
        triangles_extend = triangles.extend
        uvs_extend = mesh_uvs.extend
        face_uvs = uvs

        def add_face(v0, v1, v2, v3):
            vertex_index = len(vertices)

            vertices_extend((v0, v1, v2, v3))
            uvs_extend(face_uvs)
            triangles_extend((
                vertex_index, vertex_index + 2, vertex_index + 1,
                vertex_index, vertex_index + 3, vertex_index + 2
            ))

        if isinstance(blocks, np.ndarray):
            solid = blocks[start_x:end_x, start_y:end_y, start_z:end_z] != 0

            if not solid.any():
                self._apply_mesh()
                return

            left_empty = np.empty(solid.shape, dtype=bool)
            left_empty[0, :, :] = True if start_x == 0 else blocks[start_x - 1, start_y:end_y, start_z:end_z] == 0
            left_empty[1:, :, :] = ~solid[:-1, :, :]

            right_empty = np.empty(solid.shape, dtype=bool)
            right_empty[:-1, :, :] = ~solid[1:, :, :]
            right_empty[-1, :, :] = True if end_x == max_x else blocks[end_x, start_y:end_y, start_z:end_z] == 0

            back_empty = np.empty(solid.shape, dtype=bool)
            back_empty[:, :, 0] = True if start_z == 0 else blocks[start_x:end_x, start_y:end_y, start_z - 1] == 0
            back_empty[:, :, 1:] = ~solid[:, :, :-1]

            front_empty = np.empty(solid.shape, dtype=bool)
            front_empty[:, :, :-1] = ~solid[:, :, 1:]
            front_empty[:, :, -1] = True if end_z == max_z else blocks[start_x:end_x, start_y:end_y, end_z] == 0

            bottom_empty = np.empty(solid.shape, dtype=bool)
            bottom_empty[:, 0, :] = True if start_y == 0 else blocks[start_x:end_x, start_y - 1, start_z:end_z] == 0
            bottom_empty[:, 1:, :] = ~solid[:, :-1, :]

            top_empty = np.empty(solid.shape, dtype=bool)
            top_empty[:, :-1, :] = ~solid[:, 1:, :]
            top_empty[:, -1, :] = True if end_y == max_y else blocks[start_x:end_x, end_y, start_z:end_z] == 0

            def get_face_bounds(local_x, local_y, local_z):
                x = start_x + int(local_x)
                y = start_y + int(local_y)
                z = start_z + int(local_z)
                return x - 0.5, x + 0.5, y - 0.5, y + 0.5, z - 0.5, z + 0.5

            for local_x, local_y, local_z in np.argwhere(solid & left_empty):
                xm, xp, ym, yp, zm, zp = get_face_bounds(local_x, local_y, local_z)
                add_face(
                    (xm, ym, zm),
                    (xm, ym, zp),
                    (xm, yp, zp),
                    (xm, yp, zm)
                )

            for local_x, local_y, local_z in np.argwhere(solid & right_empty):
                xm, xp, ym, yp, zm, zp = get_face_bounds(local_x, local_y, local_z)
                add_face(
                    (xp, ym, zp),
                    (xp, ym, zm),
                    (xp, yp, zm),
                    (xp, yp, zp)
                )

            for local_x, local_y, local_z in np.argwhere(solid & front_empty):
                xm, xp, ym, yp, zm, zp = get_face_bounds(local_x, local_y, local_z)
                add_face(
                    (xm, ym, zp),
                    (xp, ym, zp),
                    (xp, yp, zp),
                    (xm, yp, zp)
                )

            for local_x, local_y, local_z in np.argwhere(solid & back_empty):
                xm, xp, ym, yp, zm, zp = get_face_bounds(local_x, local_y, local_z)
                add_face(
                    (xp, ym, zm),
                    (xm, ym, zm),
                    (xm, yp, zm),
                    (xp, yp, zm)
                )

            for local_x, local_y, local_z in np.argwhere(solid & top_empty):
                xm, xp, ym, yp, zm, zp = get_face_bounds(local_x, local_y, local_z)
                add_face(
                    (xm, yp, zp),
                    (xp, yp, zp),
                    (xp, yp, zm),
                    (xm, yp, zm)
                )

            for local_x, local_y, local_z in np.argwhere(solid & bottom_empty):
                xm, xp, ym, yp, zm, zp = get_face_bounds(local_x, local_y, local_z)
                add_face(
                    (xm, ym, zm),
                    (xp, ym, zm),
                    (xp, ym, zp),
                    (xm, ym, zp)
                )

            self._apply_mesh()
            return

        for x in range(start_x, end_x):
            x_blocks = blocks[x]
            left_x_blocks = blocks[x - 1] if x > 0 else None
            right_x_blocks = blocks[x + 1] if x + 1 < max_x else None
            xm = x - 0.5
            xp = x + 0.5

            for y in range(start_y, end_y):
                row = x_blocks[y]
                left_row = left_x_blocks[y] if left_x_blocks is not None else None
                right_row = right_x_blocks[y] if right_x_blocks is not None else None
                bottom_row = x_blocks[y - 1] if y > 0 else None
                top_row = x_blocks[y + 1] if y + 1 < max_y else None
                ym = y - 0.5
                yp = y + 0.5

                for z in range(start_z, end_z):
                    if row[z] == 0:
                        continue

                    zm = z - 0.5
                    zp = z + 0.5

                    if left_row is None or left_row[z] == 0:
                        add_face(
                            (xm, ym, zm),
                            (xm, ym, zp),
                            (xm, yp, zp),
                            (xm, yp, zm)
                        )
                    if right_row is None or right_row[z] == 0:
                        add_face(
                            (xp, ym, zp),
                            (xp, ym, zm),
                            (xp, yp, zm),
                            (xp, yp, zp)
                        )
                    if z == last_z or row[z + 1] == 0:
                        add_face(
                            (xm, ym, zp),
                            (xp, ym, zp),
                            (xp, yp, zp),
                            (xm, yp, zp)
                        )
                    if z == 0 or row[z - 1] == 0:
                        add_face(
                            (xp, ym, zm),
                            (xm, ym, zm),
                            (xm, yp, zm),
                            (xp, yp, zm)
                        )
                    if top_row is None or top_row[z] == 0:
                        add_face(
                            (xm, yp, zp),
                            (xp, yp, zp),
                            (xp, yp, zm),
                            (xm, yp, zm)
                        )
                    if bottom_row is None or bottom_row[z] == 0:
                        add_face(
                            (xm, ym, zm),
                            (xp, ym, zm),
                            (xp, ym, zp),
                            (xm, ym, zp)
                        )

        self._apply_mesh()
