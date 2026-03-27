from dataclasses import dataclass
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence, uint8


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
