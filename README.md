# transformer

### Contents 
This repository sets up transformer GPT training from scratch. It uses a HuggingFace dataset of bbc news to train. The project will include procedures to train using human preferences via reinforcement learning.


### Methodology
Borrowing HuggingFace dataset https://huggingface.co/datasets/SetFit/bbc-news this repository sets up the data, the transformer architecture and model, trains it and generates content. The reinforcement human preferences will be borrowed from https://github.com/openai/lm-human-preferences.

### Setting up the environment
The package management is done by poetry. 
```
conda create -n tfmr
conda activate tfmr
conda install pip
pip install poetry
poetry install
python run.py
```

### Output and viewing the results
The final model results can be seen here from the terminal after the generate method is called. 

### Work in Progress
- Add human preferences RL part.
- Output via Jupyter Notebook.

### Contact 
If you would like to collaborate or need any help with the code you can reach me at satesh.ramdhani@gmail.com. 