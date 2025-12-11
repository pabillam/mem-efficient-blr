# from diffusers import DiTPipeline
# import torch

# # Load pretrained DiT-XL/2 (256x256) model
# pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.bfloat16)

# print("=== Model Config ===")
# print(pipe.config)

# print("\n=== Transformer ===")
# print(pipe.transformer)

# # Compute sequence length
# img_size = pipe.transformer.config.sample_size  # should be 256
# patch_size = pipe.transformer.config.patch_size  # 2
# seq_len = (img_size // patch_size) ** 2
# print(f"\nImage size: {img_size}x{img_size}")
# print(f"Patch size: {patch_size}x{patch_size}")
# print(f"Sequence length: {seq_len} tokens")

import torch
from models import DiT_models   # comes from the DiT repo

# Pick DiT-XL/2 (ImageNet pretrained)
model_name = "DiT-XL/2"
img_size = 256
patch_size = 2

# Build model
model = DiT_models[model_name](input_size=img_size)
model.eval()

print("=== Model Architecture ===")
print(model)

# Compute sequence length
seq_len = (img_size // patch_size) ** 2
print(f"\nImage size: {img_size}x{img_size}")
print(f"Patch size: {patch_size}x{patch_size}")
print(f"Sequence length: {seq_len} tokens")  # 16384

# (Optional) load official ImageNet pretrained weights
# torch.hub.load_state_dict_from_url(...) if you want weights
