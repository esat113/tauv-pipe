"""DDS helpers for Pipe Tracker GUI v4.

This module keeps the current v4 GUI independent from older pipe tracker GUI
implementations while preserving the same DDS topics and wire types.
"""

import json
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from cyclonedds.domain import DomainParticipant
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence, uint8
from cyclonedds.pub import DataWriter
from cyclonedds.qos import Qos, Policy
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic


BEST_EFFORT_QOS = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(depth=1))


@dataclass
class FrameChunk(IdlStruct):
    # Aligned with tauv-client / tauv-core camera FrameChunk.
    source_id: str
    feed: str
    timestamp: float
    chunk_buffer: sequence[uint8]
    chunk_id_in_frame: int
    total_chunks_in_frame: int
    width: int
    height: int
    encoding: str


@dataclass
class SegmentationMask(IdlStruct):
    timestamp: float
    camera: str
    width: int
    height: int
    mask_data: sequence[uint8]


@dataclass
class StreamCommand(IdlStruct):
    command_type: str
    command_data: str
    timestamp: int
    client_id: str


class FrameAssembler:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self.expected_chunks = None
        self.received_count = 0
        self.width = 0
        self.height = 0
        self.encoding = ""
        self.timestamp = 0.0
        self.buffers = []

    def push(self, chunk):
        with self._lock:
            if chunk.chunk_id_in_frame == 0:
                total = int(chunk.total_chunks_in_frame)
                if total <= 0:
                    self.reset()
                    return None
                self.expected_chunks = total
                self.received_count = 0
                self.width = int(chunk.width)
                self.height = int(chunk.height)
                self.encoding = str(chunk.encoding)
                self.timestamp = float(chunk.timestamp)
                self.buffers = [None] * total

            if self.expected_chunks is None:
                return None
            if (
                int(chunk.total_chunks_in_frame) != self.expected_chunks
                or int(chunk.width) != self.width
                or int(chunk.height) != self.height
                or str(chunk.encoding) != self.encoding
            ):
                self.reset()
                return None

            idx = int(chunk.chunk_id_in_frame)
            if idx < 0 or idx >= self.expected_chunks:
                self.reset()
                return None

            if self.buffers[idx] is None:
                self.buffers[idx] = bytes(chunk.chunk_buffer)
                self.received_count += 1

            if self.received_count != self.expected_chunks:
                return None
            if any(part is None for part in self.buffers):
                self.reset()
                return None

            payload = b"".join(self.buffers)
            enc = self.encoding
            ts = self.timestamp
            self.reset()
            return payload, enc, ts


class DDSCameraReader:
    """Keep recent bottom camera frames for timestamp-matched mask overlays."""

    BUFFER_SIZE = 120

    def __init__(self, participant: DomainParticipant, topic_name: str):
        self._topic = Topic(participant, topic_name, FrameChunk, qos=BEST_EFFORT_QOS)
        self._reader = DataReader(participant, self._topic, qos=BEST_EFFORT_QOS)
        self._assembler = FrameAssembler()
        self._buffer: deque = deque(maxlen=self.BUFFER_SIZE)
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def start(self):
        self._stop.clear()
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            try:
                samples = self._reader.take(32)
                for chunk in samples:
                    result = self._assembler.push(chunk)
                    if result is None:
                        continue
                    payload, encoding, ts_ms = result
                    data = np.frombuffer(payload, dtype=np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    if encoding != "rgb8":
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    with self._lock:
                        self._buffer.append((float(ts_ms), img))
            except Exception:
                pass
            time.sleep(0.005)

    def get_frame(self):
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1][1].copy()

    def get_frame_at(self, ts_ms: float, max_age_ms: float = 2000.0):
        with self._lock:
            if not self._buffer:
                return None
            best = min(self._buffer, key=lambda item: abs(item[0] - ts_ms))
        if abs(best[0] - ts_ms) > max_age_ms:
            return None
        return best[1].copy()


class DDSMaskReader:
    def __init__(self, participant: DomainParticipant, topic_name: str):
        self._topic = Topic(participant, topic_name, SegmentationMask, qos=BEST_EFFORT_QOS)
        self._reader = DataReader(participant, self._topic, qos=BEST_EFFORT_QOS)
        self._mask = None
        self._mask_ts = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._warn_payload_mismatch = 0
        self._warn_parse = 0

    def start(self):
        self._stop.clear()
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            try:
                samples = self._reader.take(32)
                for sample in samples:
                    w = int(sample.width)
                    h = int(sample.height)
                    if w <= 0 or h <= 0:
                        continue
                    payload = bytes(sample.mask_data)
                    raw = np.frombuffer(payload, dtype=np.uint8)
                    expected = w * h
                    if raw.size != expected:
                        if self._warn_payload_mismatch < 6:
                            print(
                                f"[DDSMaskReader] mask_data uzunluk uyumsuz: "
                                f"got={raw.size} beklenen={expected} ({w}x{h}) "
                                f"(CycloneDDS MaxMessageSize artırın)",
                                flush=True,
                            )
                            self._warn_payload_mismatch += 1
                        continue
                    mask = raw.reshape((h, w)).copy()
                    with self._lock:
                        self._mask = mask
                        self._mask_ts = float(sample.timestamp)
            except Exception as exc:
                if self._warn_parse < 6:
                    print(f"[DDSMaskReader] örnek işleme hatası: {exc}", flush=True)
                    self._warn_parse += 1
            time.sleep(0.005)

    def get_mask(self):
        with self._lock:
            return self._mask.copy() if self._mask is not None else None

    def get_mask_with_meta(self):
        with self._lock:
            if self._mask is None:
                return None
            return self._mask.copy(), self._mask_ts


class DDSMotorPublisher:
    def __init__(self, participant: DomainParticipant, topic_name: str):
        self._topic = Topic(participant, topic_name, StreamCommand, qos=BEST_EFFORT_QOS)
        self._writer = DataWriter(participant, self._topic, qos=BEST_EFFORT_QOS)

    def send(self, rc: dict) -> None:
        self._writer.write(
            StreamCommand(
                command_type="motor_rc",
                command_data=json.dumps(rc),
                timestamp=int(time.time() * 1000),
                client_id="pipe_tracker_gui4",
            )
        )
