from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration, AutoProcessor
import torch

model_id = "./qwen3_2B_thinking"
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
    )
model = Qwen3VLForConditionalGeneration.from_pretrained(
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

output_folder = "./qwen3_2B_thinking"
model.save_pretrained(output_folder)
processor.save_pretrained(output_folder)
print("Model and processor saved to", output_folder)