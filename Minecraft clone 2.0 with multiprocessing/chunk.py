from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

uvs = (
        (0, 0),
        (1, 0),
        (1, 1),
        (0, 1)
    )


@dataclass(frozen=True)
class ChunkMeshData:
    vertices: list
    triangles: list
    uvs: list


_WORKER_SHARED_MEMORY = None
_WORKER_BLOCK_IDS = None


def initialize_chunk_worker(shared_memory_name, block_shape, block_dtype):
    global _WORKER_SHARED_MEMORY, _WORKER_BLOCK_IDS

    try:
        _WORKER_SHARED_MEMORY = shared_memory.SharedMemory(name=shared_memory_name, track=False)
    except TypeError:
        _WORKER_SHARED_MEMORY = shared_memory.SharedMemory(name=shared_memory_name)

    _WORKER_BLOCK_IDS = np.ndarray(
        tuple(block_shape),
        dtype=np.dtype(block_dtype),
        buffer=_WORKER_SHARED_MEMORY.buf,
    )


def build_chunk_mesh_data(chunk_coord):
    if _WORKER_BLOCK_IDS is None:
        raise RuntimeError("Chunk worker was not initialized with world block data.")

    return Chunk(chunk_coord).build_mesh_data(_WORKER_BLOCK_IDS)


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

        self.size = 128
        self.height = 128

    def destroy(self):
        if self.entityModel is not None:
            from ursina import destroy

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

    def _build_padded_blocks(self, blocks, start_x, start_y, start_z, end_x, end_y, end_z, max_x, max_y, max_z):
        chunk_x = end_x - start_x
        chunk_y = end_y - start_y
        chunk_z = end_z - start_z
        dtype = blocks.dtype if isinstance(blocks, np.ndarray) else np.int32
        padded = np.zeros((chunk_x + 2, chunk_y + 2, chunk_z + 2), dtype=dtype)

        if isinstance(blocks, np.ndarray):
            padded[1:-1, 1:-1, 1:-1] = blocks[start_x:end_x, start_y:end_y, start_z:end_z]

            if start_x > 0:
                padded[0, 1:-1, 1:-1] = blocks[start_x - 1, start_y:end_y, start_z:end_z]
            if end_x < max_x:
                padded[-1, 1:-1, 1:-1] = blocks[end_x, start_y:end_y, start_z:end_z]
            if start_y > 0:
                padded[1:-1, 0, 1:-1] = blocks[start_x:end_x, start_y - 1, start_z:end_z]
            if end_y < max_y:
                padded[1:-1, -1, 1:-1] = blocks[start_x:end_x, end_y, start_z:end_z]
            if start_z > 0:
                padded[1:-1, 1:-1, 0] = blocks[start_x:end_x, start_y:end_y, start_z - 1]
            if end_z < max_z:
                padded[1:-1, 1:-1, -1] = blocks[start_x:end_x, start_y:end_y, end_z]

            return padded

        for padded_x, x in enumerate(range(start_x - 1, end_x + 1)):
            if x < 0 or x >= max_x:
                continue

            x_blocks = blocks[x]
            for padded_y, y in enumerate(range(start_y - 1, end_y + 1)):
                if y < 0 or y >= max_y:
                    continue

                row = x_blocks[y]
                for padded_z, z in enumerate(range(start_z - 1, end_z + 1)):
                    if 0 <= z < max_z:
                        padded[padded_x, padded_y, padded_z] = row[z]

        return padded

    @staticmethod
    def _emit_greedy_rectangles(mask, emit_rectangle):
        rows, cols = mask.shape

        for row in range(rows):
            col = 0
            while col < cols:
                block_id = mask[row, col]
                if block_id == 0:
                    col += 1
                    continue

                width = 1
                while col + width < cols and mask[row, col + width] == block_id:
                    width += 1

                height = 1
                while (
                    row + height < rows
                    and np.all(mask[row + height, col:col + width] == block_id)
                ):
                    height += 1

                mask[row:row + height, col:col + width] = 0
                emit_rectangle(row, col, height, width)
                col += width

    def apply_mesh_data(self, mesh_data):
        self.vertices = mesh_data.vertices
        self.triangles = mesh_data.triangles
        self.uvs = mesh_data.uvs
        self._apply_mesh()

    def _apply_mesh(self):
        from ursina import Entity, Mesh

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


    def build_mesh_data(self, globalBlockIds):
        mesh_uvs = []
        triangles = []
        vertices = []

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

        if start_x >= end_x or start_y >= end_y or start_z >= end_z:
            return ChunkMeshData(vertices, triangles, mesh_uvs)

        if self._is_fully_occluded(blocks, start_x, start_y, start_z, end_x, end_y, end_z, max_x, max_y, max_z):
            return ChunkMeshData(vertices, triangles, mesh_uvs)

        padded_blocks = self._build_padded_blocks(
            blocks,
            start_x,
            start_y,
            start_z,
            end_x,
            end_y,
            end_z,
            max_x,
            max_y,
            max_z,
        )
        local_blocks = padded_blocks[1:-1, 1:-1, 1:-1]
        solid = local_blocks != 0

        if not solid.any():
            return ChunkMeshData(vertices, triangles, mesh_uvs)

        chunk_x, chunk_y, chunk_z = local_blocks.shape
        vertices_extend = vertices.extend
        triangles_extend = triangles.extend
        uvs_extend = mesh_uvs.extend
        face_uvs = uvs

        def add_face(v0, v1, v2, v3, uv_width=1, uv_height=1):
            vertex_index = len(vertices)

            vertices_extend((v0, v1, v2, v3))
            if uv_width == 1 and uv_height == 1:
                uvs_extend(face_uvs)
            else:
                uvs_extend((
                    (0, 0),
                    (uv_width, 0),
                    (uv_width, uv_height),
                    (0, uv_height)
                ))
            triangles_extend((
                vertex_index, vertex_index + 2, vertex_index + 1,
                vertex_index, vertex_index + 3, vertex_index + 2
            ))

        def solid_faces(neighbor_blocks):
            return np.where(solid & (neighbor_blocks == 0), local_blocks, 0)

        left_faces = solid_faces(padded_blocks[:-2, 1:-1, 1:-1])
        for local_x in range(chunk_x):
            plane = left_faces[local_x]
            if not plane.any():
                continue

            x = start_x + local_x - 0.5

            def emit_left(row, col, height, width, x=x):
                y0 = start_y + row - 0.5
                y1 = y0 + height
                z0 = start_z + col - 0.5
                z1 = z0 + width
                add_face(
                    (x, y0, z0),
                    (x, y0, z1),
                    (x, y1, z1),
                    (x, y1, z0),
                    width,
                    height
                )

            self._emit_greedy_rectangles(plane, emit_left)

        right_faces = solid_faces(padded_blocks[2:, 1:-1, 1:-1])
        for local_x in range(chunk_x):
            plane = right_faces[local_x]
            if not plane.any():
                continue

            x = start_x + local_x + 0.5

            def emit_right(row, col, height, width, x=x):
                y0 = start_y + row - 0.5
                y1 = y0 + height
                z0 = start_z + col - 0.5
                z1 = z0 + width
                add_face(
                    (x, y0, z1),
                    (x, y0, z0),
                    (x, y1, z0),
                    (x, y1, z1),
                    width,
                    height
                )

            self._emit_greedy_rectangles(plane, emit_right)

        front_faces = solid_faces(padded_blocks[1:-1, 1:-1, 2:])
        for local_z in range(chunk_z):
            plane = front_faces[:, :, local_z].T
            if not plane.any():
                continue

            z = start_z + local_z + 0.5

            def emit_front(row, col, height, width, z=z):
                x0 = start_x + col - 0.5
                x1 = x0 + width
                y0 = start_y + row - 0.5
                y1 = y0 + height
                add_face(
                    (x0, y0, z),
                    (x1, y0, z),
                    (x1, y1, z),
                    (x0, y1, z),
                    width,
                    height
                )

            self._emit_greedy_rectangles(plane, emit_front)

        back_faces = solid_faces(padded_blocks[1:-1, 1:-1, :-2])
        for local_z in range(chunk_z):
            plane = back_faces[:, :, local_z].T
            if not plane.any():
                continue

            z = start_z + local_z - 0.5

            def emit_back(row, col, height, width, z=z):
                x0 = start_x + col - 0.5
                x1 = x0 + width
                y0 = start_y + row - 0.5
                y1 = y0 + height
                add_face(
                    (x1, y0, z),
                    (x0, y0, z),
                    (x0, y1, z),
                    (x1, y1, z),
                    width,
                    height
                )

            self._emit_greedy_rectangles(plane, emit_back)

        top_faces = solid_faces(padded_blocks[1:-1, 2:, 1:-1])
        for local_y in range(chunk_y):
            plane = top_faces[:, local_y, :].T
            if not plane.any():
                continue

            y = start_y + local_y + 0.5

            def emit_top(row, col, height, width, y=y):
                x0 = start_x + col - 0.5
                x1 = x0 + width
                z0 = start_z + row - 0.5
                z1 = z0 + height
                add_face(
                    (x0, y, z1),
                    (x1, y, z1),
                    (x1, y, z0),
                    (x0, y, z0),
                    width,
                    height
                )

            self._emit_greedy_rectangles(plane, emit_top)

        bottom_faces = solid_faces(padded_blocks[1:-1, :-2, 1:-1])
        for local_y in range(chunk_y):
            plane = bottom_faces[:, local_y, :].T
            if not plane.any():
                continue

            y = start_y + local_y - 0.5

            def emit_bottom(row, col, height, width, y=y):
                x0 = start_x + col - 0.5
                x1 = x0 + width
                z0 = start_z + row - 0.5
                z1 = z0 + height
                add_face(
                    (x0, y, z0),
                    (x1, y, z0),
                    (x1, y, z1),
                    (x0, y, z1),
                    width,
                    height
                )

            self._emit_greedy_rectangles(plane, emit_bottom)

        return ChunkMeshData(vertices, triangles, mesh_uvs)

    def generateMesh(self, globalBlockIds):
        self.apply_mesh_data(self.build_mesh_data(globalBlockIds))
