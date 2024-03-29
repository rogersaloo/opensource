from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch

model = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(
    model,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
pipe.load_lora_weights("model/", weight_name="pytorch_lora_weights.safetensors")

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
)
refiner.to("cuda")


prompt = "a portrait of sks dog, pixar, cartoon, 3d, headshots, fantasy, 4k, uhd"

for seed in range(10):
    generator = torch.Generator("cuda").manual_seed(seed)
    image = pipe(prompt=prompt, generator=generator, num_inference_steps=25)
    image = image.images[0]
    image.save(f"images/{seed}.png")
    image = refiner(prompt=prompt, generator=generator, image=image)
    image = image.images[0]
    image.save(f"images_refined/{seed}.png")
