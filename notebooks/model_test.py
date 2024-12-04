from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os
load_dotenv()

# Load the model
model_name = "meta-llama/Llama-2-7b-chat-hf" # Or any other base model
hf_access_token = os.getenv("token")

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_access_token)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.load_adapter("adunca08/EnglishOnlyTLAT")
model.enable_adapters()
text = "Hello"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(inputs["input_ids"])
print(tokenizer.decode(outputs[0], skip_special_tokens=True))