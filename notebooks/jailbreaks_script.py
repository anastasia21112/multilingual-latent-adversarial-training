import os
import torch
import csv
import sys
from datasets import load_dataset, concatenate_datasets
from transformers import LlavaNextForConditionalGeneration, AutoProcessor
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
os.chdir("../")
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
from latent_at import *
from tasks.harmbench.HarmBenchTask import HarmBenchTask
import argparse
load_dotenv()
hf_access_token = os.getenv("INSERT API KEY")
os.environ["WANDB_MODE"] = "disabled"
import torch

print("Finished initial setup")


## ARGUMENTS FOR EXPERIMENTS
parser = argparse.ArgumentParser(description="Jailbreaking script")

parser.add_argument(
    '--languages_training', 
    nargs='+', 
    type=str,  
    help='a list of languages to train', 
    required=True 
)

parser.add_argument(
    '--languages_sft', 
    nargs='+', 
    type=str,  
    help='a list of languages to perform sft with', 
    required=True
)

parser.add_argument('--sequential', dest='sequential', action='store_true', help='Do not do SFT sequentially after training')

args = parser.parse_args()

training_langs = args.languages_training
sft_langs = args.languages_sft
sequential= args.sequential

print(sequential)
print(training_langs)
print(sft_langs)



def main(): 

    ## CLEAR GPU
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # use_llama2 = True

    model_name = "lightblue/DeepSeek-R1-Distill-Qwen-7B-Multilingual"
    adv_loss_coefs = {"toward": 0.5, "away": 0.5,}
    def_loss_coefs = {"kl": 0.1, "toward": 0.5, "away": 0.5,}
    inner_learning_rate = 1e-3
    outer_learning_rate = 8e-5
    epsilon = 6.0
    add_completions_pgd = True
    
    # if use_llama2:  # use llama2-7b
    #     model_name = "meta-llama/Llama-2-7b-chat-hf"
    #     adv_loss_coefs = {"toward": 0.5, "away": 0.5,}
    #     def_loss_coefs = {"sft": 1.5, "toward": 0.5, "away": 0.5,}
    #     inner_learning_rate = 5e-2
    #     outer_learning_rate = 2e-5
    #     epsilon = 6.0
    #     add_completions_pgd = False
    # else: # use llama3-8b
    #     model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    #     adv_loss_coefs = {"toward": 0.5, "away": 0.5,}
    #     def_loss_coefs = {"kl": 0.1, "toward": 0.5, "away": 0.5,}
    #     inner_learning_rate = 1e-3
    #     outer_learning_rate = 8e-5
    #     epsilon = 6.0
    #     add_completions_pgd = True

    model_dtype = torch.bfloat16

    device = "cuda"
    run_start_evals = False

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_access_token,
        torch_dtype=model_dtype
    ).to(device)

    if "Llama-2" in model_name:
        model_type = "llama2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    elif "Llama-3" in model_name:
        model_type = "llama3"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    elif "zephyr" in model_name or "mistral" in model_name:
        model_type = "zephyr"    
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.padding_side = "left"
    elif "Qwen" in model_name:
        model_type = "qwen"
        tokenizer = AutoTokenizer.from_pretrained("lightblue/DeepSeek-R1-Distill-Qwen-7B-Multilingual", pad_token='<|endoftext|>')
        # tokenizer.pad_token_id = tokenizer.eos_token_id # doesn't exist i think?
        tokenizer.padding_side = "left"
    else:
        print(model_name)
        raise Exception("Unsupported model type.")
    

    ## SET UP DATA TO USE
    ## TRAINING
    code_to_dataset_training = {
        'es' : 'adunca08/spanish_data', 
        'vi' : 'adunca08/vietnamese_data', 
        'ko' : 'adunca08/korean_data', 
        'zh' : 'adunca08/mandarin_data', 
        'am' : 'adunca08/amharic_data', 
        'el' : 'adunca08/greek_data', 
        'sw' : 'adunca08/swahili_data', 
        'ar' : 'adunca08/arabic_data', 
        'en' : "LLM-LAT/harmful-dataset"
    }

    ## FINETUNING
    code_to_dataset_sft = {
        'es' : 'maanasharma5/spanish_sft_data',
        'vi' : 'maanasharma5/vietnamese_sft_data',
        'ko' : 'maanasharma5/korean_sft_data', 
        'zh' : 'maanasharma5/mandarin_sft_data', 
        'am' : 'maanasharma5/amharic_sft_data', 
        'el' : 'maanasharma5/greek_sft_data', 
        'sw' : 'maanasharma5/swahili_sft_data', 
        'ar' : 'maanasharma5/arabic_sft_data', 
        'en' : 'maanasharma5/english_sft_data'
    }


    ## GET TRAINING DATA LANGUAGES
    training_datasets_list = []
    for language in training_langs: 
        print(language)
        hf_path = code_to_dataset_training[language]
        dataset = load_dataset(hf_path)
        dataset = dataset['train'].add_column("language", [language] * len(dataset['train']))
        training_datasets_list.append(dataset)

    ## GET SFT LANGUAGES
    sft_datasets_list = []
    for language in sft_langs: 
        print(language)
        hf_path = code_to_dataset_sft[language]
        dataset = load_dataset(hf_path)
        dataset = dataset['train'].add_column("language", [language] * len(dataset['train']))
        sft_datasets_list.append(dataset)

    combined_dataset_training = concatenate_datasets(training_datasets_list)
    combined_dataset_sft = concatenate_datasets(sft_datasets_list)
    
    advbench_data = HarmBenchTask(
        tokenizer=tokenizer,
        gen_batch_size=1,
        cls_batch_size=1,
        device=device,
        data_name="advbench",
        train_test_split=.8
    )

    harmbench_data = HarmBenchTask(
        tokenizer=tokenizer,
        gen_batch_size=1,
        cls_batch_size=1,
        device=device,
        data_name="harmbench_text",
        train_test_split=.8,
        func_categories=["standard", "contextual"]
    )

    sys_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    if model_type == "llama2": # LLama 2 Chat Formatting
        use_tokenizer_template = True
        custom_prompt_template = None
        custom_completion_template = None
    elif model_type == "llama3": # LLama 3 chat formatting
        use_tokenizer_template = False
        custom_prompt_template = f"<|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|>"+"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        custom_completion_template="{completion}"
    elif model_type == "qwen":  # Qwen chat formatting
            use_tokenizer_template = False
            custom_prompt_template = (
                "<|im_start|>system\n"
                "{system_prompt}<|im_end|>\n"
                "<|im_start|>user\n"
                "{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            custom_completion_template = "{completion}<|im_end|>"
    else:  # Zephyr chat formatting
        sys_prompt=""
        use_tokenizer_template = False
        custom_prompt_template = "<|user|>\n{prompt}</s> \n <|assistant|>\n"
        custom_completion_template="{completion}"
    
    ## PROCESS  MULTILINGUAL DATA
    lat_dataset = process_generic_chat_dataset_multilingual(
        tokenizer,
        og_dataset=combined_dataset_training,
        adv_column="rejected",
        def_column="chosen",
        split="train",
        use_tokenizer_template=use_tokenizer_template,
        system_prompt=sys_prompt,
        custom_prompt_template=custom_prompt_template,
        custom_completion_template=custom_completion_template, 
        languages=training_langs,
    )

    ## PROCESS MULTILINGUAL DATA
    lat_dataloader = lat_dataset.create_latent_adversarial_dataloader()

    # interleaving supervised finetuning with LAT stabilizes training
    sft_dataset = process_generic_chat_dataset_multilingual(
        tokenizer,
        og_dataset = combined_dataset_sft,
        adv_column="refusal",
        def_column="response",
        split="train",
        use_tokenizer_template=use_tokenizer_template,
        system_prompt=sys_prompt,
        custom_prompt_template=custom_prompt_template,
        custom_completion_template=custom_completion_template,
        add_eos_token=True, 
        languages=sft_langs,
    )
    sft_dataloader = sft_dataset.create_latent_adversarial_dataloader()
   
    ## callback functions to write to output
    def adv_loss_callback(losses, output_folder, epoch): 
        file_path = f"{output_folder}/adv_losses.csv"
        file_exists = os.path.isfile(file_path)

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
           
            if not file_exists:
                writer.writerow(['epoch', 'losses'])
            
            writer.writerow([epoch, losses])

    def def_loss_callback(losses, output_folder, epoch): 
        file_path = f"{output_folder}/def_losses.csv"
        file_exists = os.path.isfile(file_path)

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            if not file_exists:
                writer.writerow(['epoch', 'losses'])
            
            writer.writerow([epoch, losses])

    
    peft_config = LoraConfig(
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, peft_config)

    ## SETUP TRAINER
    pgd_trainer = ProjectedGradLAT(
        model=model,  # model
        dataloader=lat_dataloader,  # dataloader for lat
        post_adv_callback=adv_loss_callback,
        post_def_callback=def_loss_callback,
        sft_dataloader=sft_dataloader,  # dataloader for supervised finetuning
        adv_loss_coefs=adv_loss_coefs,  # adversary's loss coefs
        def_loss_coefs=def_loss_coefs,  # model's loss coefs
        pgd_layers=["embedding", 8, 16, 24, 
                    # 30 # removing for qwen
                ],  # what layers to attack
        pgd_iterations_per_step=16,  # how many steps of projected gradient descent to do
        model_layers=list(range(0, model.config.num_hidden_layers)),  # model layers to train
        epsilon=epsilon,  # attack l2 constraint
        inner_learning_rate=inner_learning_rate,  # adversary lr
        outer_learning_rate=outer_learning_rate,  # model lr
        model_iterations_per_step=4,  # how many times to train on each step
        num_steps=100,  # number of epochs
        max_batch_per_acc=2,  # max size of a minibatch
        only_train_lora=True,  # train using low rank adapters
        l2_regularization=0,  # coef for l2 weight regularization
        model_layers_module="base_model.model.model.layers",  # where the model layers are
        reinitialize_dev_optim=True,  # whether to reinitialize optimizer every lat step,
        add_completions_pgd=add_completions_pgd,  # Whether to add PGD over the completion tokens
        N_checkpoints=10,
        languages = set(sft_langs).intersection(set(training_langs)),
        sequential=sequential
    )

    ## TRAIN
    pgd_trainer.train(project_name="jailbreaks_test")
    pgd_trainer.save_model()

if __name__ == "__main__":
    main()