from math import floor

import numpy as np
from ursina import *

from chunk import Chunk
from noclip_camera import NoclipCamera


CHUNK_SIZE = 32
CHUNK_HEIGHT = 40
WORLD_ROWS = 1024
WORLD_LAYERS = 160
WORLD_COLS = 1024


class VoxelWorld:
    def __init__(self):
        self.block_ids = np.ones((WORLD_ROWS, WORLD_LAYERS, WORLD_COLS), dtype=np.uint8)
        self.chunks = {}
        self.chunk_counts = (
            self._ceil_div(WORLD_ROWS, CHUNK_SIZE),
            self._ceil_div(WORLD_LAYERS, CHUNK_HEIGHT),
            self._ceil_div(WORLD_COLS, CHUNK_SIZE),
        )

        self._create_debug_air_pockets()

    @staticmethod
    def _ceil_div(value, divisor):
        return (value + divisor - 1) // divisor

    def _create_debug_air_pockets(self):
        for x in range(self.chunk_counts[0]):
            for y in range(self.chunk_counts[1]):
                for z in range(self.chunk_counts[2]):
                    self.block_ids[x * CHUNK_SIZE][y * CHUNK_HEIGHT][z * CHUNK_SIZE] = 0

    def in_chunk_bounds(self, chunk_coord):
        chunk_x, chunk_y, chunk_z = chunk_coord
        chunk_count_x, chunk_count_y, chunk_count_z = self.chunk_counts
        return (
            0 <= chunk_x < chunk_count_x
            and 0 <= chunk_y < chunk_count_y
            and 0 <= chunk_z < chunk_count_z
        )

    def in_block_bounds(self, x, y, z):
        return 0 <= x < WORLD_ROWS and 0 <= y < WORLD_LAYERS and 0 <= z < WORLD_COLS

    def block_to_chunk(self, x, y, z):
        return (
            floor(x / CHUNK_SIZE),
            floor(y / CHUNK_HEIGHT),
            floor(z / CHUNK_SIZE),
        )

    def load_chunk(self, chunk_coord):
        if chunk_coord in self.chunks or not self.in_chunk_bounds(chunk_coord):
            return

        chunk = Chunk(chunk_coord)
        chunk.generateMesh(self.block_ids)
        self.chunks[chunk_coord] = chunk

    def rebuild_chunk(self, chunk_coord):
        chunk = self.chunks.get(chunk_coord)
        if chunk is not None:
            chunk.generateMesh(self.block_ids)

    def destroy_chunk(self, chunk_coord):
        chunk = self.chunks.pop(chunk_coord, None)
        if chunk is not None:
            chunk.destroy()

    def set_block(self, x, y, z, block_id):
        if not self.in_block_bounds(x, y, z):
            return set()

        if self.block_ids[x][y][z] == block_id:
            return set()

        self.block_ids[x][y][z] = block_id
        return self._chunks_touched_by_block(x, y, z)

    def _chunks_touched_by_block(self, x, y, z):
        chunk_x, chunk_y, chunk_z = self.block_to_chunk(x, y, z)
        touched = {(chunk_x, chunk_y, chunk_z)}

        if x % CHUNK_SIZE == 0:
            touched.add((chunk_x - 1, chunk_y, chunk_z))
        if x % CHUNK_SIZE == CHUNK_SIZE - 1:
            touched.add((chunk_x + 1, chunk_y, chunk_z))
        if y % CHUNK_HEIGHT == 0:
            touched.add((chunk_x, chunk_y - 1, chunk_z))
        if y % CHUNK_HEIGHT == CHUNK_HEIGHT - 1:
            touched.add((chunk_x, chunk_y + 1, chunk_z))
        if z % CHUNK_SIZE == 0:
            touched.add((chunk_x, chunk_y, chunk_z - 1))
        if z % CHUNK_SIZE == CHUNK_SIZE - 1:
            touched.add((chunk_x, chunk_y, chunk_z + 1))

        return {chunk_coord for chunk_coord in touched if self.in_chunk_bounds(chunk_coord)}


class ChunkLoader:
    def __init__(
        self,
        world,
        player,
        horizontal_radius=8,
        vertical_radius=3,
        unload_margin=1,
        max_loads_per_frame=1,
        max_rebuilds_per_frame=1,
    ):
        self.world = world
        self.player = player
        self.horizontal_radius = horizontal_radius
        self.vertical_radius = vertical_radius
        self.unload_margin = unload_margin
        self.max_loads_per_frame = max_loads_per_frame
        self.max_rebuilds_per_frame = max_rebuilds_per_frame
        self.center_chunk = None
        self.target_chunks = set()
        self.load_queue = []
        self.dirty_chunks = set()

    def update(self):
        center_chunk = self._player_chunk()
        if center_chunk != self.center_chunk:
            self.center_chunk = center_chunk
            self._refresh_targets()

        self._process_rebuild_queue()
        self._process_load_queue()

    def queue_rebuilds(self, chunk_coords):
        for chunk_coord in chunk_coords:
            if chunk_coord in self.world.chunks:
                self.dirty_chunks.add(chunk_coord)

    def _player_chunk(self):
        position = self.player.world_position
        return self.world.block_to_chunk(position.x, position.y, position.z)

    def _refresh_targets(self):
        self.target_chunks = self._collect_chunks(
            self.center_chunk,
            self.horizontal_radius,
            self.vertical_radius,
        )

        keep_chunks = self._collect_chunks(
            self.center_chunk,
            self.horizontal_radius + self.unload_margin,
            self.vertical_radius + self.unload_margin,
        )

        for chunk_coord in tuple(self.world.chunks):
            if chunk_coord not in keep_chunks:
                self.world.destroy_chunk(chunk_coord)
                self.dirty_chunks.discard(chunk_coord)

        self.load_queue = sorted(
            (
                chunk_coord
                for chunk_coord in self.target_chunks
                if chunk_coord not in self.world.chunks
            ),
            key=self._chunk_distance_score,
        )

    def _collect_chunks(self, center_chunk, horizontal_radius, vertical_radius):
        center_x, center_y, center_z = center_chunk
        radius_squared = horizontal_radius * horizontal_radius
        chunks = set()

        for chunk_x in range(center_x - horizontal_radius, center_x + horizontal_radius + 1):
            for chunk_y in range(center_y - vertical_radius, center_y + vertical_radius + 1):
                for chunk_z in range(center_z - horizontal_radius, center_z + horizontal_radius + 1):
                    delta_x = chunk_x - center_x
                    delta_z = chunk_z - center_z
                    if delta_x * delta_x + delta_z * delta_z > radius_squared:
                        continue

                    chunk_coord = (chunk_x, chunk_y, chunk_z)
                    if self.world.in_chunk_bounds(chunk_coord):
                        chunks.add(chunk_coord)

        return chunks

    def _chunk_distance_score(self, chunk_coord):
        center_x, center_y, center_z = self.center_chunk
        chunk_x, chunk_y, chunk_z = chunk_coord
        delta_x = chunk_x - center_x
        delta_y = chunk_y - center_y
        delta_z = chunk_z - center_z
        return delta_x * delta_x + delta_z * delta_z + delta_y * delta_y * 4

    def _process_rebuild_queue(self):
        for _ in range(self.max_rebuilds_per_frame):
            if not self.dirty_chunks:
                return

            chunk_coord = min(self.dirty_chunks, key=self._chunk_distance_score)
            self.dirty_chunks.remove(chunk_coord)
            self.world.rebuild_chunk(chunk_coord)

    def _process_load_queue(self):
        loaded_this_frame = 0
        remaining_queue = []

        for chunk_coord in self.load_queue:
            if chunk_coord in self.world.chunks or chunk_coord not in self.target_chunks:
                continue

            if loaded_this_frame >= self.max_loads_per_frame:
                remaining_queue.append(chunk_coord)
                continue

            self.world.load_chunk(chunk_coord)
            loaded_this_frame += 1

        self.load_queue = remaining_queue


class MinecraftGame:

    def __init__(self):
        self.app = Ursina()
        self.world = VoxelWorld()
        self.player = NoclipCamera(position=(0, 0, 0), rotation=(25, 0, 0))
        self.lastPlayerChunkPosition = (0, 0, 0)
        self.loader = ChunkLoader(self.world, self.player)
        self.loader.update()
        self.world.chunks[self.lastPlayerChunkPosition].entityModel.collider = "mesh"

        self.player_probe = Entity(model="cube", collider="box", parent=self.player)
        self.edit_x = 0

    def getplayerCurrentChunk(self):
        return (int(self.player.world_position[0]/CHUNK_SIZE),
                int(self.player.world_position[1]/CHUNK_HEIGHT),
                int(self.player.world_position[2]/CHUNK_SIZE))

    def update(self):
        self.loader.update()

        current_chunk_pos = self.getplayerCurrentChunk()

        if current_chunk_pos != self.lastPlayerChunkPosition:
            current_chunk = self.world.chunks.get(current_chunk_pos)
            last_chunk = self.world.chunks.get(self.lastPlayerChunkPosition)

            if current_chunk is not None and current_chunk.entityModel is not None:
                current_chunk.entityModel.collider = "mesh"

                if last_chunk is not None and last_chunk.entityModel is not None:
                    last_chunk.entityModel.collider = None

                self.lastPlayerChunkPosition = current_chunk_pos

        self.lastPlayerChunkPosition = current_chunk_pos


        if self.player_probe.intersects().hit:
            print("hit")

    def input(self, key):
        if key == "x":
            self._carve_debug_line()

    def _carve_debug_line(self):
        self.edit_x += 1
        dirty_chunks = set()

        for z in range(3):
            dirty_chunks.update(self.world.set_block(self.edit_x, 0, z, 0))

        self.loader.queue_rebuilds(dirty_chunks)

    def run(self):
        self.app.run()


game = None


def update():
    if game is not None:
        game.update()


def input(key):
    if game is not None:
        game.input(key)


if __name__ == "__main__":
    game = MinecraftGame()
    game.run()
