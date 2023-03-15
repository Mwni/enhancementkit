from dataclasses import dataclass

@dataclass
class CheckpointPaths:
	codeformer: str
	realesrgan: str
	facelib: str