from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import torch
import os
load_dotenv()

import tasks.harmbench.FastHarmBenchEvals as FastHarmBenchEvals

VALID_LANGUAGES = [
    "eng",
    "spa",
    "cmn",
    "kor",
    "arb",
    "ell",
    "swh",
    "amh",
    "vie",
]

def run_evals(model):
    print("Evaluating some general capabilities on English Only")
    FastHarmBenchEvals.run_general_evals(model, language="eng", evals_to_include=["MMLU", "SciQ", "HellaSwag"])

    for lang in VALID_LANGUAGES:
        print(f"\nEvaluating on {lang}")
        print("Running MMMLU Evals")
        try:
            FastHarmBenchEvals.run_general_evals(model, language=lang, evals_to_include=["MMLU"])
        except ValueError as e:
            print(f"Error running MMLU evals for {lang}\n Error: {e}")
        print("Running Attack Evals")
        FastHarmBenchEvals.run_attack_evals(model, language=lang)
        print("Running Attack Evals with Gibberish Suffixes")
        FastHarmBenchEvals.run_attack_evals(model, language=lang, gibberish_suffixes=True)

# Load the model
model_name = "meta-llama/Llama-2-7b-chat-hf" # Or any other base model
# hf_access_token = os.getenv("HF_ACCESS_TOKEN")
hf_access_token = os.getenv("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")

model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_access_token, torch_dtype=torch.bfloat16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# print("Base Model")
# run_evals(model)

# print("English Only TLAT")
# model.load_adapter("adunca08/EnglishOnlyTLAT")
# model.enable_adapters()
# run_evals(model)

# print("English + Vietnamese TLAT")
# model.load_adapter("adunca08/FixedEnglishVietnamese")
# model.enable_adapters()
# run_evals(model)

print("Multilingual TLAT")
model.load_adapter("adunca08/FixedMultingualAll")
model.enable_adapters()
run_evals(model)

# print("Multilingual TLAT with English SFT")
# model.load_adapter("adunca08/MultilingualTrainEnglishSFT")
# model.enable_adapters()
# run_evals(model)

