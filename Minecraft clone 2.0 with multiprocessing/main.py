from collections import deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from math import floor
from multiprocessing import shared_memory
from time import perf_counter
import multiprocessing
import os

from player import Player
import numpy as np
from ursina import *

from chunk import Chunk, build_chunk_mesh_data, initialize_chunk_worker
from noclip_camera import NoclipCamera
from raycastTest import get_direction_3d, raycast_3d_array


CHUNK_SIZE = 128
CHUNK_HEIGHT = 128
WORLD_ROWS = 2048
WORLD_LAYERS = 256
WORLD_COLS = 2048


class VoxelWorld:
    def __init__(self):
        self.block_shape = (WORLD_ROWS, WORLD_LAYERS, WORLD_COLS)
        self.block_dtype = np.dtype(np.uint8)
        block_count = int(np.prod(self.block_shape))
        self._block_shared_memory = shared_memory.SharedMemory(
            create=True,
            size=block_count * self.block_dtype.itemsize,
        )
        self.block_ids = np.ndarray(
            self.block_shape,
            dtype=self.block_dtype,
            buffer=self._block_shared_memory.buf,
        )
        self.block_ids.fill(1)
        self.chunks = {}
        self.chunk_counts = (
            self._ceil_div(WORLD_ROWS, CHUNK_SIZE),
            self._ceil_div(WORLD_LAYERS, CHUNK_HEIGHT),
            self._ceil_div(WORLD_COLS, CHUNK_SIZE),
        )

        self._create_debug_air_pockets()
        self.createSpot()

    @staticmethod
    def _ceil_div(value, divisor):
        return (value + divisor - 1) // divisor

    def createSpot(self):
        for x in range(10):
            for y in range(10):
                self.block_ids[x][255][y] = 0

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

        mesh_data = Chunk(chunk_coord).build_mesh_data(self.block_ids)
        self.apply_chunk_mesh_data(chunk_coord, mesh_data)

    def rebuild_chunk(self, chunk_coord):
        chunk = self.chunks.get(chunk_coord)
        if chunk is not None:
            chunk.apply_mesh_data(chunk.build_mesh_data(self.block_ids))

    def apply_chunk_mesh_data(self, chunk_coord, mesh_data):
        if not self.in_chunk_bounds(chunk_coord):
            return

        chunk = self.chunks.get(chunk_coord)
        if chunk is None:
            chunk = Chunk(chunk_coord)
            self.chunks[chunk_coord] = chunk

        chunk.apply_mesh_data(mesh_data)

    def destroy_chunk(self, chunk_coord):
        chunk = self.chunks.pop(chunk_coord, None)
        if chunk is not None:
            chunk.destroy()

    @property
    def shared_memory_name(self):
        return self._block_shared_memory.name

    def close(self):
        shared_block = getattr(self, "_block_shared_memory", None)
        if shared_block is None:
            return

        self.block_ids = None
        shared_block.close()
        try:
            shared_block.unlink()
        except FileNotFoundError:
            pass

        self._block_shared_memory = None

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
        horizontal_radius=5,
        vertical_radius=3,
        unload_margin=1,
        max_loads_per_frame=1,
        max_rebuilds_per_frame=1,
        max_unloads_per_frame=1,
        chunk_work_budget_seconds=0.002,
        slow_frame_seconds=1 / 45,
        max_work_debt_seconds=0.01,
        max_mesh_applies_per_frame=1,
        max_pending_chunk_jobs=None,
        worker_count=None,
    ):
        self.world = world
        self.player = player
        self.horizontal_radius = horizontal_radius
        self.vertical_radius = vertical_radius
        self.unload_margin = unload_margin
        self.max_loads_per_frame = max_loads_per_frame
        self.max_rebuilds_per_frame = max_rebuilds_per_frame
        self.max_unloads_per_frame = max_unloads_per_frame
        self.chunk_work_budget_seconds = chunk_work_budget_seconds
        self.slow_frame_seconds = slow_frame_seconds
        self.max_work_debt_seconds = max_work_debt_seconds
        self.max_mesh_applies_per_frame = max_mesh_applies_per_frame
        self.center_chunk = None
        self.target_chunks = set()
        self.keep_chunks = set()
        self.load_queue = []
        self.unload_queue = []
        self.dirty_chunks = set()
        self.completed_meshes = deque()
        self.pending_jobs = {}
        self.pending_chunks = {}
        self.chunk_versions = {}
        self.work_budget_seconds = chunk_work_budget_seconds
        self.max_saved_work_budget_seconds = chunk_work_budget_seconds * 3
        self.last_update_time = perf_counter()
        self.worker_count = worker_count or max(1, min(4, (os.cpu_count() or 2) - 1))
        self.max_pending_chunk_jobs = max_pending_chunk_jobs or self.worker_count * 2
        self.executor = self._create_executor()

    def _create_executor(self):
        context = multiprocessing.get_context("spawn")
        return ProcessPoolExecutor(
            max_workers=self.worker_count,
            mp_context=context,
            initializer=initialize_chunk_worker,
            initargs=(
                self.world.shared_memory_name,
                self.world.block_ids.shape,
                self.world.block_ids.dtype.str,
            ),
        )

    def update(self):
        frame_seconds = self._measure_frame_seconds()
        center_chunk = self._player_chunk()
        if center_chunk != self.center_chunk:
            self.center_chunk = center_chunk
            self._refresh_targets()

        self._collect_finished_jobs()
        self._process_unload_queue(frame_seconds)
        if frame_seconds > self.slow_frame_seconds:
            return

        self._apply_completed_meshes()
        self._process_rebuild_queue()
        self._process_load_queue()

    def queue_rebuilds(self, chunk_coords):
        for chunk_coord in chunk_coords:
            self.chunk_versions[chunk_coord] = self.chunk_versions.get(chunk_coord, 0) + 1
            if chunk_coord in self.world.chunks:
                self.dirty_chunks.add(chunk_coord)
            elif chunk_coord in self.target_chunks:
                self._queue_load(chunk_coord)

    def _player_chunk(self):
        position = self.player.world_position
        return self.world.block_to_chunk(position.x, position.y, position.z)

    def _measure_frame_seconds(self):
        now = perf_counter()
        frame_seconds = now - self.last_update_time
        self.last_update_time = now
        self.work_budget_seconds = min(
            self.work_budget_seconds + self.chunk_work_budget_seconds,
            self.max_saved_work_budget_seconds,
        )
        return frame_seconds

    def _refresh_targets(self):
        self.target_chunks = self._collect_chunks(
            self.center_chunk,
            self.horizontal_radius,
            self.vertical_radius,
        )

        self.keep_chunks = self._collect_chunks(
            self.center_chunk,
            self.horizontal_radius + self.unload_margin,
            self.vertical_radius + self.unload_margin,
        )

        unloads = set(self.unload_queue)
        for chunk_coord in tuple(self.world.chunks):
            if chunk_coord not in self.keep_chunks:
                unloads.add(chunk_coord)

        self.unload_queue = sorted(
            (
                chunk_coord
                for chunk_coord in unloads
                if chunk_coord in self.world.chunks and chunk_coord not in self.keep_chunks
            ),
            key=self._chunk_distance_score,
            reverse=True,
        )

        self.load_queue = sorted(
            (
                chunk_coord
                for chunk_coord in self.target_chunks
                if chunk_coord not in self.world.chunks and chunk_coord not in self.pending_chunks
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

    def _has_work_budget(self):
        return self.work_budget_seconds > 0

    def _has_pending_capacity(self):
        return len(self.pending_jobs) < self.max_pending_chunk_jobs

    def _run_chunk_work(self, work):
        start_time = perf_counter()
        work()
        elapsed = perf_counter() - start_time
        self.work_budget_seconds = max(
            self.work_budget_seconds - elapsed,
            -self.max_work_debt_seconds,
        )

    def _process_unload_queue(self, frame_seconds):
        max_unloads = 1 if frame_seconds > self.slow_frame_seconds else self.max_unloads_per_frame
        unloaded = 0

        while self.unload_queue and unloaded < max_unloads:
            chunk_coord = self.unload_queue.pop(0)
            if chunk_coord in self.keep_chunks:
                continue

            self.world.destroy_chunk(chunk_coord)
            self.dirty_chunks.discard(chunk_coord)
            unloaded += 1

    def _collect_finished_jobs(self):
        finished_jobs = [future for future in self.pending_jobs if future.done()]

        for future in finished_jobs:
            chunk_coord, version = self.pending_jobs.pop(future)
            if self.pending_chunks.get(chunk_coord) is future:
                del self.pending_chunks[chunk_coord]

            try:
                mesh_data = future.result()
            except BrokenProcessPool:
                print("Chunk worker pool stopped. Falling back to main-thread chunk builds.")
                self._shutdown_executor()
                self._build_chunk_mesh_synchronously(chunk_coord, version)
                continue
            except Exception as exc:
                print(f"Chunk worker failed for {chunk_coord}: {exc}")
                continue

            self.completed_meshes.append((chunk_coord, version, mesh_data))

    def _apply_completed_meshes(self):
        applied = 0

        while (
            self.completed_meshes
            and applied < self.max_mesh_applies_per_frame
            and self._has_work_budget()
        ):
            chunk_coord, version, mesh_data = self.completed_meshes.popleft()
            if version != self.chunk_versions.get(chunk_coord, 0):
                self._reschedule_stale_chunk(chunk_coord)
                continue

            if not self._should_apply_chunk_result(chunk_coord):
                continue

            self._run_chunk_work(
                lambda chunk_coord=chunk_coord, mesh_data=mesh_data: (
                    self.world.apply_chunk_mesh_data(chunk_coord, mesh_data)
                )
            )
            applied += 1

    def _should_apply_chunk_result(self, chunk_coord):
        if chunk_coord in self.world.chunks:
            return chunk_coord in self.keep_chunks or chunk_coord in self.target_chunks

        return chunk_coord in self.target_chunks

    def _reschedule_stale_chunk(self, chunk_coord):
        if chunk_coord in self.world.chunks:
            self.dirty_chunks.add(chunk_coord)
        elif chunk_coord in self.target_chunks:
            self._queue_load(chunk_coord)

    def _queue_load(self, chunk_coord):
        if (
            chunk_coord in self.world.chunks
            or chunk_coord in self.pending_chunks
            or chunk_coord in self.load_queue
            or chunk_coord not in self.target_chunks
        ):
            return

        self.load_queue.append(chunk_coord)
        self.load_queue.sort(key=self._chunk_distance_score)

    def _submit_chunk_job(self, chunk_coord):
        if chunk_coord in self.pending_chunks or not self.world.in_chunk_bounds(chunk_coord):
            return False

        version = self.chunk_versions.get(chunk_coord, 0)
        if self.executor is None:
            self._build_chunk_mesh_synchronously(chunk_coord, version)
            return True

        try:
            future = self.executor.submit(build_chunk_mesh_data, chunk_coord)
        except BrokenProcessPool:
            print("Chunk worker pool stopped. Falling back to main-thread chunk builds.")
            self._shutdown_executor()
            self._build_chunk_mesh_synchronously(chunk_coord, version)
            return True
        except Exception as exc:
            print(f"Could not submit chunk worker for {chunk_coord}: {exc}")
            return False

        self.pending_jobs[future] = (chunk_coord, version)
        self.pending_chunks[chunk_coord] = future
        return True

    def _build_chunk_mesh_synchronously(self, chunk_coord, version):
        mesh_data = Chunk(chunk_coord).build_mesh_data(self.world.block_ids)
        self.completed_meshes.append((chunk_coord, version, mesh_data))

    def _process_rebuild_queue(self):
        rebuilt = 0
        while rebuilt < self.max_rebuilds_per_frame and self._has_pending_capacity():
            if not self.dirty_chunks:
                return

            pending_dirty_chunks = self.dirty_chunks - set(self.pending_chunks)
            if not pending_dirty_chunks:
                return

            chunk_coord = min(pending_dirty_chunks, key=self._chunk_distance_score)
            self.dirty_chunks.remove(chunk_coord)
            if chunk_coord not in self.world.chunks:
                continue

            if self._submit_chunk_job(chunk_coord):
                rebuilt += 1

    def _process_load_queue(self):
        loaded_this_frame = 0

        while (
            self.load_queue
            and loaded_this_frame < self.max_loads_per_frame
            and self._has_pending_capacity()
        ):
            chunk_coord = self.load_queue.pop(0)
            if (
                chunk_coord in self.world.chunks
                or chunk_coord in self.pending_chunks
                or chunk_coord not in self.target_chunks
            ):
                continue

            if self._submit_chunk_job(chunk_coord):
                loaded_this_frame += 1

    def _shutdown_executor(self):
        if self.executor is None:
            return

        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self.executor.shutdown(wait=False)

        self.executor = None

    def shutdown(self):
        for future in self.pending_jobs:
            future.cancel()

        self.pending_jobs.clear()
        self.pending_chunks.clear()
        self.completed_meshes.clear()
        self._shutdown_executor()


class MinecraftGame:

    def __init__(self):
        self.app = Ursina()
        self.world = VoxelWorld()
        self.player = NoclipCamera(position=(0, 0, 0), rotation=(25, 0, 0))
        camera.fov = 80
        #self.player = Player(10,190,10,cam=camera)
        self.lastPlayerChunkPosition = None
        self.lastPlayerPosition = (0, 0, 0)
        self.loader = ChunkLoader(self.world, self.player)
        for i in range(200):
            Entity(model="cube", parent=self.player)
        #self.loader.update()
        #self.world.chunks[self.lastPlayerChunkPosition].entityModel.collider = "mesh"

        self.player_probe = Entity(model="cube", collider="box", parent=self.player)
        self.edit_x = 0

    def getplayerCurrentChunk(self):
        position = self.player.world_position
        return self.world.block_to_chunk(position.x, position.y, position.z)

    def changeColliders(self):
        current_chunk_pos = self.getplayerCurrentChunk()

        if current_chunk_pos != self.lastPlayerChunkPosition:
            current_chunk = self.world.chunks.get(current_chunk_pos)
            last_chunk = self.world.chunks.get(self.lastPlayerChunkPosition)

            if current_chunk is not None and current_chunk.entityModel is not None:
                current_chunk.entityModel.collider = "mesh"

                if last_chunk is not None and last_chunk.entityModel is not None:
                    last_chunk.entityModel.collider = None

                self.lastPlayerChunkPosition = current_chunk_pos


    def update(self):
        self.loader.update()
        self.changeColliders()
        #self.player.checkCollision(self.world.block_ids)
        #self.player.applyGravity(self.world.block_ids)
        currentPos = tuple(self.player.world_position)
        j = 0
        for x in range(200):
            j+=1

        print(j)

        direction = get_direction_3d(self.lastPlayerPosition, currentPos)

        self.lastPlayerPosition = currentPos

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
        try:
            self.app.run()
        finally:
            self.loader.shutdown()
            self.world.close()


game = None


def update():
    if game is not None:
        game.update()


def input(key):
    if game is not None:
        game.input(key)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    game = MinecraftGame()
    game.run()
