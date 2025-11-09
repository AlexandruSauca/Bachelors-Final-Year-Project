from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

model_id = "./qwen2.5_3B"
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
    )
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
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

output_dir ='./qwen2.5_3B'
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print("Model and processor saved to", output_dir)