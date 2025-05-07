from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os

os.environ["TRANSFORMERS_NO_BITSANDBYTES"] = "1"   # belt‑and‑suspenders

def load(path: str):
    base, adapter = path.split("::") if "::" in path else (path, None)

    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},          # all on CPU
        offload_folder="offload",        # stream weights from disk
        offload_state_dict=True,
        low_cpu_mem_usage=True,
    )

    if adapter:
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()

    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    tok.pad_token = tok.eos_token
    return model.eval(), tok
