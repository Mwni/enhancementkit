import torch
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.srvgg_arch import SRVGGNetCompact


class RealESRGan():
	def __init__(self, checkpoint, scale=4, tile=400, tile_pad=10, pre_pad=0, half=None):
		if half is None:
			half = not torch.cuda.is_available()

		if scale == 2:
			model = RRDBNet(
				num_in_ch=3, 
				num_out_ch=3, 
				num_feat=64, 
				num_block=23, 
				num_grow_ch=32, 
				scale=2
			)
		elif scale == 4:
			model = SRVGGNetCompact(
				num_in_ch=3, 
				num_out_ch=3, 
				num_feat=64, 
				num_conv=32, 
				upscale=4, 
				act_type='prelu'
			)
		else:
			raise 'scale %i is not supported' % scale

		self.upsampler = RealESRGANer(
			scale=scale,
			model_path=checkpoint,
			model=model,
			tile=tile,
			tile_pad=tile_pad,
			pre_pad=pre_pad,
			half=half,
		)

	def __call__(self, image, *args, **kwargs):
		if isinstance(image, Image.Image):
			return Image.fromarray(
				self.upsampler.enhance(
					np.array(image)[:, :, ::-1],
					*args, 
					**kwargs
				)[0][:, :, ::-1]
			)
		else:
			return self.upsampler.enhance(
				image,
				*args, 
				**kwargs
			)[0]