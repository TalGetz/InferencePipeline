from dataclasses import dataclass

from src.models.recognition.face_recognition.arcfaceresnet100.item import ArcFaceResnet100Item


@dataclass
class ArcFaceResnet100Target(ArcFaceResnet100Item):
    name: str = ""
