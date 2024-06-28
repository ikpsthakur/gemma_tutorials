# pip install accelerate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
tokenizer = AutoTokenizer.from_pretrained("/home/xmiaas/Downloads/aitools/gemma/models/gemma-2b")
#model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")
model = AutoModelForCausalLM.from_pretrained("/home/xmiaas/Downloads/aitools/gemma/models/gemma-2b", device_map="auto", torch_dtype=torch.bfloat16)

input_text = 'Write me a poem about Machine Learning.'
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=1000)
print(tokenizer.decode(outputs[0]))
