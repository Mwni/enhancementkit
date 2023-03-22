import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


class RealESRGan():
	def __init__(self, checkpoint, scale=2, tile=400, tile_pad=10, pre_pad=0, half=None):
		if half is None:
			half = not torch.cuda.is_available()

		self.upsampler = RealESRGANer(
			scale=2,
			model_path=checkpoint,
			model=RRDBNet(
				num_in_ch=3, 
				num_out_ch=3, 
				num_feat=64, 
				num_block=23, 
				num_grow_ch=32, 
				scale=2
			),
			tile=tile,
			tile_pad=tile_pad,
			pre_pad=pre_pad,
			half=half,
		)

	def __call__(self, *args, **kwargs):
		return self.upsampler.enhance(*args, **kwargs)