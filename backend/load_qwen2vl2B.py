from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration, AutoProcessor
import torch

model_id = "../models/qwen2_vl_model"
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
    )
model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto",
        quantization_config = quant_config
    )
min_pixels = 256*28*28
max_pixels = 488*28*28
processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels = min_pixels,
        max_pixels = max_pixels
    )