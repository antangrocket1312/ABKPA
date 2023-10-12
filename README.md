<div align="center">

# ABKPA: Quantitative Review Summarization

</div>

This repository maintains the code, data, and model checkpoints for the paper *Aspect-based Key Point Analysis for Quantitative Summarization of Business Reviews*

## Installation
### 1. Create new virtual environment by
```bash
conda env create -f environment.yml
conda activate abkpa
```
### 2. Install Pytorch
For other versions, please visit: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
### Linux
##### Using GPUs
```bash
pip3 install torch torchvision torchaudio
conda install -c anaconda cudatoolkit
```
#### Using CPU
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Dataset
We released the training and evaluation datasets of ABKPA. Datasets can be accessed under the ```data/``` folder, 
following the [```training/```](/data/training) and [```evaluation/```](/data/evaluation) subdirectories for each dataset.

Additionally, we provide the  [```yelp/```](/data/yelp) subdirectory that contains the raw, unprocessed and preprocessed data from Yelp
to allow reproducibility and extensibility. This folder can be downloaded
from this [Google Drive link](https://drive.google.com/drive/folders/1kIEsac0e819rX63PmENPfTctWWww1mIC?usp=sharing), 
under the `data/` directory.

Files in each folder:
* ```.pkl```: data in .pkl format, accessible via Pandas library.
* ```.csv```: data in .csv format.
* ```.jsonl```: data in .jsonl format (only for Yelp raw data).

## Model checkpoints
Model checkpoints are saved and accessed under the [```code/model/```](/code/model) folder. We released models trained under different settings (e.g. in-category/out-of-category)
for reproducibility and evaluation.

Model checkpoints can be downloaded from this [Google Drive link](https://drive.google.com/drive/folders/1XvjLh3IrpfCxnPoxphId0DYTQB3Eca2Q?usp=sharing).

We release three pretrained checkpoints for reproducibility of ABKPA. All checkpoints must be located under The checkpoint must be located 
under the [```code/model/```](/model) directory.
- `model/roberta-large-pretrained-yelp.zip` The model checkpoint of the RoBERTa-large model adapted to Yelp business reviews
by pretraining on the Masked LM task. For reproducibility, it can be utilized to fine-tune new KP Matching models for review summarization.
- `model/ABKPA.zip` The model checkpoint of ABKPA's contrastive KP Matching learning model, trained with data in different settings 
and business categories of reviews for evaluation.
Each model checkpoint is located in the respective ```{setting}/{category}/``` folder, while ```setting``` can either be **in-category** or **out-of-category**.
Simply place ABKPA folder into [```code/model/```](/model) from the working directory to reproduce evaluation results in the paper.
Any newly fine-tuned models can also be found in the under the same ```{setting}/{category}/``` folder.
- `model/roberta-large-finetuned-yelp-argument-quality-WA.zip` The model checkpoint of the argument quality ranking model fine-tuned on the Yelp-pretrained RoBERTa-large model, 
using ArgQ-14kPairs dataset. The model is used at the first stage of the KP Extraction process to select high-quality KPs that can be used to construct contrastive exmaples to train ABKPA.

## Code
For reproducing the ABKPA training and evaluation, we provide the following notebooks:
-  [```contrastive_examples_data_preprocessing.ipynb```](/code/contrastive_examples_data_preprocessing.ipynb) The notebook contains the code to pre-process, sample and select good data entries from 
the Yelp dataset to later construct contrastive examples in ```contrastive_examples_data_construction.ipynb```
-  [```contrastive_examples_data_construction.ipynb```](/code/contrastive_examples_data_construction.ipynb) The notebook contains the code to construct contrastive examples for training the ABKPA model.
-  [```ABKPA_training_preparation.ipynb```](/code/ABKPA_training_preparation.ipynb) The notebook contains the code to prepare and transform the training data into desired input to ABKPA's siamese model in in-category/out-of-category settings
-  [```ABKPA_training.ipynb```](/code/ABKPA_training.ipynb) The notebook contains the code to train the KP Matching model of ABKPA in in-category/out-of-category settings
-  [```ABKPA_evaluation.ipynb```](/code/ABKPA_evaluation.ipynb) The notebook contains the code for inference and evaluating the ABKPA model
-  [```ABKPA¬c_evaluation.ipynb```](/code/ABKPA¬c_evaluation.ipynb) The notebook contains the code to conduct evaluation on ABKPA¬c (the ablation study of ABKPA without contrastive learning)