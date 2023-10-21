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
to allow reproducibility and extensibility. The zipped folder of [```yelp/```]() can be downloaded
from this [Google Drive link](https://drive.google.com/drive/folders/1NrO0GgMoV13PwixVd9qk90HZtlu23wqg?usp=sharing). Note that [```yelp/```]() must be unzipped and placed under [```data/```]().

Files in each folder:
* ```.pkl```: data in .pkl format, accessible via Pandas library.
* ```.csv```: data in .csv format.
* ```.jsonl```: data in .jsonl format (only for Yelp raw data).

## Model checkpoints
Model checkpoints can be downloaded from this [Google Drive link](https://drive.google.com/drive/folders/1NrO0GgMoV13PwixVd9qk90HZtlu23wqg?usp=sharing).
Please download the file and unzip the [```checkpoints```]() directory into the main working directory.

All model checkpoints are located under the [```checkpoints/```](/model) directory. We release three pretrained checkpoints for reproducibility of ABKPA, including:
- `roberta-large-pretrained-yelp` The model checkpoint of the RoBERTa-large model adapted to Yelp business reviews
by pretraining on the Masked LM task. For reproducibility, it can be utilized to fine-tune new KP Matching models for review summarization.
- `ABKPA` The model checkpoints of aspect-based KP Matching model of ABKPA. Models are trained with data in different settings 
and business categories of reviews for evaluation.
Each model checkpoint is located in the respective ```{setting}/{category}/``` folder, while ```setting``` can either be **in-category** or **out-of-category**, and ```category``` is the data business category being evaluated (e.g., hotels, restaurants).
- `roberta-large-finetuned-yelp-argument-quality-WA` The model checkpoint of the argument quality ranking model fine-tuned on the Yelp-pretrained RoBERTa-large model, 
using ArgQ-14kPairs dataset. The model is used in as part of our aspect-based KP Extraction process to select high-quality KPs.

## Code
For reproducing the ABKPA training and evaluation, we provide the following notebooks under [```notebook/```](/notebook):
-  [```data_preprocessing.ipynb```](/notebook/data_preprocessing.ipynb) The notebook contains the code to pre-process, sample and select good data entries from 
the Yelp dataset to later construct aspect-sentiment matching pairs and automatically annotate their silver labels in ```automatic_silver_label_annotation.ipynb```
-  [```automatic_silver_label_annotation.ipynb```](/notebook/automatic_silver_label_annotation.ipynb) The notebook contains the code to aspect-sentiment matching pairs and automatically annotate their silver labels for training the ABKPA model.
-  [```ABKPA_training_preparation.ipynb```](/notebook/ABKPA_training_preparation.ipynb) The notebook contains the code to prepare and transform the training data into desired input to ABKPA's siamese model in in-category/out-of-category settings
-  [```ABKPA_training.ipynb```](/notebook/ABKPA_training.ipynb) The notebook contains the code to train the KP Matching model of ABKPA in in-category/out-of-category settings, using contrastive learning
-  [```ABKPA_evaluation.ipynb```](/notebook/ABKPA_evaluation.ipynb) The notebook contains the code for inference and evaluating the ABKPA model
-  [```ABKPA_ablation_evaluation.ipynb```](/notebook/ABKPA_ablation_evaluation.ipynb) The notebook contains the code to conduct evaluation on ABKPA¬c (the ablation study of ABKPA without contrastive learning)

## Evaluation Script
We provide the evaluation scripts to reproduce the experiments reported in our paper. 
The scripts reproduce the experiments for all experiment and data configurations of ABKPA and ABKPA¬c discussed in our paper.
All scripts are located under [```scripts/```](/scripts).

### Running ABKPA Evaluation
```bash
python  scripts/ABKPA_evaluation.py --experiment_type=in_category --data_configuration=all_comments
```
Arguments:
-  ```base_model_dir```: The path to the base directory containing all model checkpoints of ABKPA corresponding to different experiment/data configuration.
-  ```experiment_type```: The type/configuration of the experiment from which the model is trained. Two valid experiment configurations are:
   - ```in_category```: The model is trained and evaluated using data from the same business category
   - ```out_of_category```: The model is trained using data from business categories other than one being evaluated. 
-  ```data_configuration```: The configuration of the data to be used for evaluating the model. Valid parameters are:
   - ```all_comments```: Taking all comments from the test data for evaluation
   - ```multi_opinion_comments```: Taking only comments with multiple opinions for evaluation
-  ```test_data_path```: The path to the test data for evaluation.

### Running ABKPA¬c Evaluation
```bash
python scripts/ABKPA_ablation_evaluation.py --data_configuration=all_comments
```
Arguments:
-  ```data_configuration```: The configuration of the data to be used for evaluating the model. Valid parameters are: ```all_comments``` and ```multi_opinion_comments```.
-  ```test_data_path```: The path to the test data for evaluation.