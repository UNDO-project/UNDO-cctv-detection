from dataclasses import dataclass


@dataclass(frozen=True)
class CameraDataFromCsv:
    latitude: float
    longitude: float
    url: str
