import torch
import gc
import pandas as pd
import math
from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

SOURCE_LANGUAGE = "eng"
TARGET_LANGUAGES = [
    "spa",
    "cmn",
    "kor",
    "arb",
    "ell",
    "swh",
    "amh",
    "vie",
]
REFUSAL_RESPONSES_PER_LANGUAGE = { 
 'spa': 'Lo siento, pero no puedo ayudar con esa petición.',
 'cmn': '我很抱歉,但我不能协助这个要求.',
 'kor': '죄송하지만, 그 요청에 대해 도움을 줄 수 없습니다.',
 'arb': 'أنا آسف، ولكن لا أستطيع المساعدة في هذا الطلب.',
 'ell': 'Λυπάμαι, αλλά δεν μπορώ να βοηθήσω με αυτό το αίτημα.',
 'swh': 'Naomba radhi, lakini siwezi kusaidia na ombi hilo.',
 'amh': 'ይቅርታ አድርግልኝ፣ ነገር ግን ይህን ጥያቄ ማሟላት አልችልም።',
 'vie': 'Tôi xin lỗi, nhưng tôi không thể hỗ trợ với yêu cầu đó.',
}

# torch.cuda.reset_accumulated_memory_stats()
# torch.cuda.reset_peak_memory_stats()
# torch.cuda.empty_cache()
# gc.collect()

assert torch.cuda.is_available(), "expected cuda"
device = "cuda"

model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large").to(device)
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")

print("Loaded models")

ultrachat_english = load_dataset("LLM-LAT/benign-dataset", split="train")
ultrachat_english = ultrachat_english.shuffle(seed=33).select(range(15000))
ultrachat_english.set_format(type="torch", columns=["prompt", "response", "refusal"])

# write english subset to parquet
ultrachat_english.to_parquet("eng_base_data.parquet")

dataloader = DataLoader(ultrachat_english, batch_size=8, shuffle=False)

def preprocess_response(response):
    MAX_LENGTH = 5000
    if len(response) > MAX_LENGTH:
        response = response[:MAX_LENGTH]
    return response

def translate_batch(texts: list[str], targ_lang: str) -> list[str]:
    text_inputs = processor(text = texts, src_lang = SOURCE_LANGUAGE, return_tensors="pt").to(device)
    text_outputs = model.generate(**text_inputs, tgt_lang = targ_lang, num_beams=4, do_sample=True)
    return processor.batch_decode(text_outputs, skip_special_tokens=True)

for target_language in TARGET_LANGUAGES:
    print(f"TRANSLATING {target_language}\n\n")

    output_path = f"{target_language}_translated_data.parquet"
    df = pd.DataFrame(columns=["prompt", "response", "refusal"], dtype=pd.StringDtype())
    df.to_parquet(output_path, index=False)
    print(df.dtypes)

    write_cycle_batches = 10
    num_batches = math.ceil(len(dataloader.dataset) / dataloader.batch_size)

    write_prompts = []
    write_responses = []
    write_refusals = []
    
    for (batch_ix, batch) in tqdm(enumerate(dataloader)):

        prompts = batch["prompt"]
        responses = list(map(preprocess_response, batch["response"]))

        translated_prompts = translate_batch(prompts, target_language)
        translated_responses = translate_batch(responses, target_language)
        translated_refusals = [REFUSAL_RESPONSES_PER_LANGUAGE[target_language] for _ in range(len(translated_prompts))]

        write_prompts.extend(translated_prompts)
        write_responses.extend(translated_responses)
        write_refusals.extend(translated_refusals)

        if (batch_ix % write_cycle_batches) == 0 or (batch_ix == num_batches - 1):
            data = {
                "prompt": write_prompts,
                "response": write_responses,
                "refusal": write_refusals,
            }

            data = pd.DataFrame(data, dtype=pd.StringDtype())
            print(data.dtypes)

            data.to_parquet(output_path, index=False, engine="fastparquet", append=True)
            
            write_prompts.clear()
            write_responses.clear()
            write_refusals.clear()