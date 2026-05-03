from transformers import AutoTokenizer
from minisweagent.models.utils.actions_toolcall import BASH_TOOL
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct")
txt = tok.apply_chat_template(
    [{"role":"system","content":"x"},{"role":"user","content":"y"}],
    tools=[BASH_TOOL], add_generation_prompt=True, tokenize=False)
print(repr(txt[-200:]))