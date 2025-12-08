from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import torch

name_model = "nvidia/Orchestrator-8B"

tokenizer = AutoTokenizer.from_pretrained(name_model)
model = AutoModelForCausalLM.from_pretrained(
    name_model,
    dtype=torch.float16,
    device_map="auto",
)
prompt = "Solve: What is 15% of 234? Use calculator tool if needed."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
