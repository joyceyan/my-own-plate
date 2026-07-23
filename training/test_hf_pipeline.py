"""Quick smoke test for the HF training pipeline."""
import os
import json
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from datasets import load_dataset
from peft import get_peft_model

from hf_utils import apply_vision_block_lora, apply_projector_lora, get_peft_lora_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}")

model_path = "training/cache/Qwen3-VL-2B-Instruct"
print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
processor.image_processor.min_pixels = 384 * 384
processor.image_processor.max_pixels = 384 * 384

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map=device,
)
model.config.use_cache = False

print("Applying LoRA...")
apply_vision_block_lora(model, r=32, alpha=32, dropout=0.0)
apply_projector_lora(model, r=64, alpha=64, dropout=0.0)
model = get_peft_model(model, get_peft_lora_config(r=64, alpha=64, dropout=0.0))

print("Loading one sample...")
ds = load_dataset("parquet", data_dir="~/src/my-own-plate/data/nutrition5k_hf_chat", split="validation")
item = ds[0]
image = item["image"]
messages = json.loads(item["messages_json"])
print("messages:", messages)

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print("text:", text[:200])

inputs = processor(text=text, images=image, return_tensors="pt")
print("input_ids shape:", inputs["input_ids"].shape)
print("pixel_values shape:", inputs["pixel_values"].shape)
if "image_grid_thw" in inputs:
    print("image_grid_thw shape:", inputs["image_grid_thw"].shape)

inputs = {k: v.to(device) for k, v in inputs.items()}
labels = inputs["input_ids"].clone()

print("Forward pass...")
with torch.cuda.amp.autocast(dtype=torch.float16) if device.type == "cuda" else torch.no_grad():
    outputs = model(**inputs, labels=labels)
print("loss:", outputs.loss.item())

print("Generate...")
with torch.no_grad():
    out_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=0.0)
out_text = processor.decode(out_ids[0], skip_special_tokens=True)
print("generated:", out_text)
print("OK")
