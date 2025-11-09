from transformers import BitsAndBytesConfig, AutoProcessor, Gemma3ForConditionalGeneration
import torch
import os
from PIL import Image

model_id = "./gemma-3-4b-it"
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
    )
model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config = quant_config,
    )
min_pixels = 256*28*28
max_pixels = 488*28*28
processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels = min_pixels,
        max_pixels = max_pixels
    )

# save_directory = "./gemma-3-4b-it"
# model.save_pretrained(save_directory)
# processor.save_pretrained(save_directory)
# print("Model and processor saved to", save_directory)