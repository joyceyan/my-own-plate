"""Compare HF adapter output vs GGUF output on a few validation samples."""
import base64
import json
import os
import sys
import requests
import torch
from PIL import Image as PILImage
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

sys.path.insert(0, os.path.dirname(__file__))
from hf_utils import apply_vision_block_lora, apply_projector_lora, load_custom_lora

MODEL_ID = 'training/cache/Qwen3-VL-2B-Instruct'
ADAPTER_DIR = 'training/output/adapter'
VISION_LORA = 'training/output/vision_lora.pt'
SERVER_URL = 'http://localhost:8081'

# Load HF adapter model
device = 'mps'
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
processor.image_processor.min_pixels = 384*384
processor.image_processor.max_pixels = 384*384

base = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, trust_remote_code=True, dtype=torch.float16, device_map=device
)
apply_vision_block_lora(base, r=32, alpha=32)
apply_projector_lora(base, r=64, alpha=64)
load_custom_lora(base, VISION_LORA)
hf_model = PeftModel.from_pretrained(base, ADAPTER_DIR)
hf_model.eval()

# Load first N samples
samples = []
with open('data/nutrition5k_validation.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        samples.append(json.loads(line))

def hf_predict(sample):
    img = PILImage.open(os.path.expanduser(sample['image_path'])).convert('RGB')
    messages = [{'role':'user','content':[{'type':'image'},{'type':'text','text':sample['prompt']}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=img, return_tensors='pt').to(device)
    with torch.no_grad():
        out = hf_model.generate(**inputs, max_new_tokens=128, do_sample=False)
    return processor.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

def gguf_predict(sample):
    img_path = os.path.expanduser(sample['image_path'])
    with open(img_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(img_path)[1].lstrip('.')
    mime = {'png':'image/png','jpg':'image/jpeg','jpeg':'image/jpeg'}.get(ext,'image/png')
    payload = {
        'messages': [{
            'role': 'user',
            'content': [
                {'type':'image_url','image_url':{'url':f'data:{mime};base64,{b64}'}},
                {'type':'text','text':sample['prompt']}
            ]
        }],
        'max_tokens': 512,
        'temperature': 0.0,
    }
    resp = requests.post(f'{SERVER_URL}/v1/chat/completions', json=payload, timeout=120)
    return resp.json()['choices'][0]['message']['content']

for i, sample in enumerate(samples):
    print(f'\n=== Sample {i} ===')
    print('True:', sample['completion'])
    print('HF  :', hf_predict(sample)[:200])
    print('GGUF:', gguf_predict(sample)[:200])
