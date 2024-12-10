## NLP FINAL PROJECT: 
Anastasia Dunca, *[adunca08@mit.edu](adunca08@mit.edu);
Maanas Sharma, *[maanas@mit.edu](maanas@mit.edu);
Olivia Munoz, *[munoz03@mit.edu](munoz03@mit.edu);
Victor Rosales, *[vrosales@mit.edu](vrosales@mit.edu);

To run an experiment: 
- navigate to the get_resps.sh file 
- provide arguments for the languages you want to train/SFT on (can be en, es, ko, el, cm, vi, am, ar, sw)
- do you want the training/SFT to be sequential (same language to train and do SFT on one after the other?, if so, add --sequential)
- submit the job to the cluster with sbatch get_resps.sh
- output is written to the logs folder 

To run evals: 
- navigate to the run_evals_all_models.py file 
- load your adaptor file with the code in the file
- enable your adaptor for the base model (in this case LLaMA-2)
- go to run_evals.sh
- submit the job to the cluster with sbatch run_evals.sh
- output is written on the output file in the logs document


## ORIGINAL PAPER: 
# Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs

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

```
@article{sheshadri2024targeted,
  title={Targeted Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs},
  author={Sheshadri, Abhay and Ewart, Aidan and Guo, Phillip and Lynch, Aengus and Wu, Cindy and Hebbar, Vivek and Sleight, Henry and Stickland, Asa Cooper and Perez, Ethan and Hadfield-Menell, Dylan and Casper, Stephen},
  journal={arXiv preprint arXiv:2407.15549},
  year={2024}
}
```

See also preliminary work: [Defending Against Unforeseen Failure Modes with Latent Adversarial Training](https://arxiv.org/abs/2403.05030).

## This repository

This repository contains code for implementing latent adversarial attacks 
and latent adversarial training (LAT) in LLMs. 

<figure>
  <img src="figs/fig1.png" alt="Targeted Latent Adversarial Training">
  <figcaption>To perform targeted latent adversarial training (LAT) in LLMs, we perturb the latent activations
in an LLM’s residual stream to elicit specific failure modes from the model. Then, we fine-tune
LLMs on the target task under these perturbations. We use this approach to improve robustness to
jailbreaks, remove backdoors without access to the trigger, and unlearn
undesirable knowledge.</figcaption>
</figure>


## Setup

After you clone and navigate to the repository:

```angular2html
pip install -r requirements.txt
bash install_tasks_from_github.sh
```


## Ready to go with the notebooks

Find notebooks for latent space attacks, jaiblreak robustness, 
backdoor removal, harry potter unlearning, and wmdp unlearning 
in the ```/notebooks``` folder.
