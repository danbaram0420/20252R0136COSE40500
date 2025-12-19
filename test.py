import torch
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image

pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
  'ShinoharaHare/Waifu-Inpaint-XL',
  torch_dtype=torch.half
)
pipeline.to('cuda')

image = load_image('images/sample.jpg')
mask_image = load_image('masks/sample_mask.png', lambda x: x.convert('L'))

inpainted_image = pipeline(
  prompt='background with tree',
  image=image,
  mask_image=mask_image,
  num_inference_steps=28,
  guidance_scale=5.0,
  height=image.height,
  width=image.width,
  generator=torch.Generator(pipeline.device).manual_seed(5)
).images[0]

inpainted_image.show()
