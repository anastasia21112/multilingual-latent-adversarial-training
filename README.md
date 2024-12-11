 ## NLP FINAL PROJECT: 
Anastasia Dunca, *[adunca08@mit.edu](adunca08@mit.edu);
Olivia Munoz, *[munoz03@mit.edu](munoz03@mit.edu);
Victor Rosales, *[vrosales@mit.edu](vrosales@mit.edu);
Maanas Sharma, *[maanas@mit.edu](maanas@mit.edu);

To run an experiment: 
- navigate to the get_resps.sh file 
- provide arguments for the languages you want to train/SFT on (can be en, es, ko, el, cm, vi, am, ar, sw)
- do you want the training/SFT to be sequential (same language to train and do SFT on one after the other?, if so, add --sequential)
- run the get_resps.sh file
- output is written to the logs folder 

To run evals: 
- navigate to the run_evals_all_models.py file 
- load your adaptor file with the code in the file
- enable your adaptor for the base model (in this case LLaMA-2-7b)
- go to run_evals.sh and run it
- output is written on the output file in the logs document

## DATA: 
* Multilingual Training Datasets: [Hugging Face Collection produced by us](https://huggingface.co/collections/adunca08/nlp-final-project-67435bcd2f6a94e877730db0)
* Multilingual SFT Datasets: [Hugging Face Collection produced by us](https://huggingface.co/collections/adunca08/multilingual-sft-datasets-6758bc9f8e62f1acbe1619af)
* [Original Training Data from Paper, English only](https://huggingface.co/datasets/LLM-LAT/harmful-dataset)
* [Original SFT Data from Paper, English only](https://huggingface.co/datasets/LLM-LAT/benign-dataset)

## MODELS: 
Try out the models by loading the base [LLaMA-2-7b](https://huggingface.co/meta-llama/Llama-2-7b) and attaching an adaptor. 
* [English only TLAT](https://huggingface.co/adunca08/EnglishOnlyTLAT)
* [Multilingual Train + SFT](https://huggingface.co/adunca08/FixedMultingualAll)
* [Multilingual Train + English SFT](https://huggingface.co/adunca08/MultilingualTrainEnglishSFT)
* [English/Vietnamese Train + English/Vietnamese SFT](https://huggingface.co/adunca08/FixedEnglishVietnamese)

## Repositories for T-LAT and Evaluation Tasks from original paper: 
* [T-LAT](https://github.com/aengusl/latent-adversarial-training)
* [Evaluation Tasks](https://github.com/magikarp01/tasks)

## LIST OF CHANGES: 
* Added multilanguage dataset class to support training/SFT of multiple languages 
* Added synchronized data loader to do training/sft with same language batches
* Added multilanguage evaluation script to use datasets of multiple languages 
* Added multilingual MMLU to the evaluation tasks
* Translated original training dataset into 8 other languages (linked in README)
* Translated original SFT dataset into 8 other languages (linked in README)
* Translated evaluation dataset into 8 other languages (dataset in github repo)
  
## ORIGINAL PAPER: 
Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs

Abhay Sheshadri,* [asheshadri31@gatech.edu](asheshadri31@gatech.edu); 
Aidan Ewart,* [aidanprattewart@gmail.com](aidanprattewart@gmail.com); 
Phillip Guo,* [phguo@umd.edu](phguo@umd.edu); 
Aengus Lynch,* [aenguslynch@gmail.com](aenguslynch@gmail.com);
Cindy Wu,* [wu.cindyx@gmail.com](wu.cindyx@gmail.com);
Vivek Hebbar*;
Henry Sleight;
Asa Cooper Stickland;
Ethan Perez;
Dylan Hadfield-Menell;
Stephen Casper, [scasper@mit.edu](scasper@mit.edu)

See our [models on Hugging Face Hub:](https://huggingface.co/LLM-LAT).

Read the paper on arXiv: [Targeted Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs](https://arxiv.org/abs/2407.15549).

Chat with our robust refusal model ([https://huggingface.co/LLM-LAT/robust-llama3-8b-instruct](https://huggingface.co/LLM-LAT/robust-llama3-8b-instruct)) at [https://www.abhayesian.com/lat-chat](https://www.abhayesian.com/lat-chat).
