import random
import math
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--edit", required=True, type = str)
parser.add_argument("--input", required=True, type = str)
parser.add_argument("--output", required=True, type = str)
parser.add_argument("--seed", type=int)
parser.add_argument("--steps", default=100, type=int)
parser.add_argument("--cfg-text", default=7.5, type=float)
parser.add_argument("--cfg-image", default=1.5, type=float)


args = parser.parse_args()

seed = random.randint(0, 100000) if args.seed is None else args.seed




model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)




g_image = Image.open(args.input)
input_image = Image.new("RGB", g_image.size)
input_image.paste(g_image)


width_o, height_o = input_image.size
width, height = input_image.size
factor = 512 / max(width, height)
#factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
#width = int((width * factor) // 64) * 64
#height = int((height * factor) // 64) * 64

width = int((width * factor))
height = int((height * factor))

#input_image = ImageOps.fit(input_image, (width, height))
input_image = input_image.resize((width, height))


#images = pipe(args.edit, image=input_image).images





generator = torch.manual_seed(seed)
images = pipe(
    args.edit, image=input_image,
    guidance_scale=args.cfg_text, image_guidance_scale=args.cfg_image,
    num_inference_steps=args.steps, generator=generator,
    ).images

d = images[0].resize((width_o,height_o)) #, resample=Image.BOX

enhancer = ImageEnhance.Sharpness(d)
s = enhancer.enhance(10)

#s = d.filter(ImageFilter.SHARPEN)
#d.save(args.output) 
s.save(args.output[:-4] + "s" + args.output[-4:])
