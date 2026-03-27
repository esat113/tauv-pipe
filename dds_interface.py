import json
import time
import threading
from queue import Queue, Full

from cyclonedds.domain import DomainParticipant
from cyclonedds.sub import DataReader
from cyclonedds.pub import DataWriter
from cyclonedds.topic import Topic

from dds.types import SegmentationMask, StreamCommand


class MaskSubscriber:
    """SAM3 segmentasyon maskesini DDS uzerinden alir (blocking)."""

    def __init__(self, participant: DomainParticipant, topic_name: str):
        self._topic = Topic(participant, topic_name, SegmentationMask)
        self._reader = DataReader(participant, self._topic)
        self._queue: Queue = Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="mask-sub"
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _recv_loop(self):
        while not self._stop_event.is_set():
            try:
                samples = self._reader.take(32)
                for sample in samples:
                    try:
                        self._queue.get_nowait()
                    except Exception:
                        pass
                    try:
                        self._queue.put_nowait(sample)
                    except Full:
                        pass
            except Exception:
                pass
            time.sleep(0.005)

    def get_mask(self, timeout: float = 1.0) -> SegmentationMask | None:
        try:
            return self._queue.get(timeout=timeout)
        except Exception:
            return None


class CommandPublisher:
    """RC_OVERRIDE komutlarini DDS uzerinden yayinlar."""

    def __init__(self, participant: DomainParticipant, topic_name: str):
        self._topic = Topic(participant, topic_name, StreamCommand)
        self._writer = DataWriter(participant, self._topic)

    def send(self, rc: dict):
        cmd = StreamCommand(
            command_type="RC_OVERRIDE",
            command_data=json.dumps(rc),
            timestamp=int(time.time() * 1000),
        )
        self._writer.write(cmd)
