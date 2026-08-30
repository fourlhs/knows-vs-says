import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id="Qwen/Qwen3.5-4B", dtype=torch.float32, device="cuda"):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device).eval()
    return model, tok
