import torch
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image

pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
  'ShinoharaHare/Waifu-Inpaint-XL',
  torch_dtype=torch.half
)
pipeline.to('cuda')

image = load_image('https://cdn-uploads.huggingface.co/production/uploads/630ed69a31970d1cd4fd575d/tPo5oPJQpxWamM-tGIYqj.png')
mask_image = load_image('https://cdn-uploads.huggingface.co/production/uploads/630ed69a31970d1cd4fd575d/QpmzmgROUM0eP53Cxx2Ih.png', lambda x: x.convert('L'))

inpainted_image = pipeline(
  prompt='blue eyes, holding red spider lily in hand',
  image=image,
  mask_image=mask_image,
  num_inference_steps=28,
  guidance_scale=5.0,
  height=image.height,
  width=image.width,
  generator=torch.Generator(pipeline.device).manual_seed(5)
).images[0]

inpainted_image.show()
