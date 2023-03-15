import torch
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


class RealESRGan():
	def __init__(self, checkpoint_path, scale=2, tile=400, tile_pad=10, pre_pad=0, half=None):
		if half is None:
			half = not torch.cuda.is_available()

		self.upsampler = RealESRGANer(
			scale=scale,
			model_path=checkpoint_path,
			model=SRVGGNetCompact(
				num_in_ch=3, 
				num_out_ch=3, 
				num_feat=64, 
				num_conv=32, 
				upscale=4, 
				act_type='prelu'
			),
			tile=tile,
			tile_pad=tile_pad,
			pre_pad=pre_pad,
			half=half,
		)

	def __call__(self, *args, **kwargs):
		return self.upsampler.enahnce(*args, **kwargs)