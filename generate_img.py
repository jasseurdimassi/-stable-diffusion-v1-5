import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

# Charger le pipeline en mode CPU uniquement
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,   # CPU → float32
    use_safetensors=True
)

# Pas besoin de offload ni xformers sur CPU

# Image de départ
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

# Prompt de génération
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# Génération (⚠️ ça peut prendre 3-10 minutes sur CPU selon la machine)
image = pipeline(prompt=prompt, image=init_image, strength=0.8).images[0]

# Comparaison avant/après
make_image_grid([init_image, image], rows=1, cols=2)
