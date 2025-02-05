from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import torch
os.chdir("../")
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
import tasks.harmbench.FastHarmBenchEvals as FastHarmBenchEvals
from dotenv import load_dotenv

load_dotenv()
hf_access_token = os.getenv("token")


# # Load the model
# model_name = "meta-llama/Llama-2-7b-chat-hf" # Or any other base model
# # Load the model and tokenizer
# model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_access_token)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# model.load_adapter("adunca08/EnglishOnlyTLAT")
# # model.load_adapter("adunca08/EnglishVietnameseTest")
# model.enable_adapters()

# Load Qwen Models
dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    "lightblue/DeepSeek-R1-Distill-Qwen-7B-Multilingual",
    dtype=dtype,
)
tokenizer = AutoTokenizer.from_pretrained("lightblue/DeepSeek-R1-Distill-Qwen-7B-Multilingual", pad_token='<|endoftext|>', dtype=dtype)
tokenizer.padding_side = "left"

model.load_adapter("maanasharma5/QwenMultilingualTrainEnglishSFT", dtype=dtype)
model.enable_adapters()

print("Running Attack Evals")
FastHarmBenchEvals.run_attack_evals(model)

print("Running General Evals")
FastHarmBenchEvals.run_general_evals(model)

# text = "Hello"
# inputs = tokenizer(text, return_tensors="pt")
# outputs = model.generate(inputs["input_ids"])
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))