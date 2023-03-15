import torch
from torchvision.transforms.functional import normalize
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

from .checkpoints import CheckpointPaths
from .utils import img2tensor, tensor2img
from .nets.codeformer import CodeFormerModel
from .realesrgan import RealESRGan


class CodeFormer():
	def __init__(self, checkpoint_paths: CheckpointPaths):
		self.checkpoint_paths = checkpoint_paths
		self.device = 'cuda'

		self.upsampler = RealESRGan(
			scale=2,
			checkpoint_path=checkpoint_paths.realesrgan,
			pre_pad=0,
			half=True,
		)

		self.net = CodeFormerModel(
			dim_embd=512,
			codebook_size=1024,
			n_head=8,
			n_layers=9,
			connect_list=['32', '64', '128', '256'],
		)

		checkpoint = torch.load(checkpoint_paths.codeformer)['params_ema']

		self.net.load_state_dict(checkpoint)
		self.net.to(self.device)
		self.net.eval()


	def __call__(self, image, fidelity=0.5, bg_enhance=False, face_upsample=True, upscale=2):
		self.face_helper = FaceRestoreHelper(
			upscale,
			face_size=512,
			crop_ratio=(1, 1),
			det_model='retinaface_resnet50',
			save_ext='png',
			use_parse=True,
			device=self.device,
			model_rootpath=self.checkpoint_paths.facelib
		)

		bg_upsampler = self.upsampler if bg_enhance else None
		face_upsampler = self.upsampler if face_upsample else None
	
		self.face_helper.clean_all()
		self.face_helper.read_image(image)
		self.face_helper.get_face_landmarks_5(
			only_center_face=False, 
			resize=640, 
			eye_dist_threshold=5
		)
		self.face_helper.align_warp_face()

		for cropped_face in self.face_helper.cropped_faces:
			cropped_face_t = img2tensor(
				cropped_face / 255.0, bgr2rgb=True, float32=True
			)

			normalize(
				cropped_face_t, 
				(0.5, 0.5, 0.5), 
				(0.5, 0.5, 0.5), 
				inplace=True
			)

			cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

			try:
				with torch.no_grad():
					output = self.net(
						cropped_face_t, 
						w=fidelity, 
						adain=True
					)[0]

					restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

				del output
				torch.cuda.empty_cache()

			except Exception as error:
				print(f'failed inference for codeformer: {error}')
				restored_face = tensor2img(
					cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
				)

			restored_face = restored_face.astype('uint8')
			self.face_helper.add_restored_face(restored_face)

		if bg_upsampler is not None:
			bg_img = bg_upsampler.enhance(image, outscale=upscale)[0]
		else:
			bg_img = None

		self.face_helper.get_inverse_affine(None)
		
		return self.face_helper.paste_faces_to_input_image(
			upsample_img=bg_img,
		)