# Model checkpoints
Model checkpoints can be downloaded from this [Google Drive link](https://drive.google.com/drive/folders/1XvjLh3IrpfCxnPoxphId0DYTQB3Eca2Q?usp=sharing).

We release three pretrained checkpoints for reproducibility of ASKPA. All checkpoints must be located under The checkpoint must be located 
under the [```code/model/```](/model) directory.
- `model/roberta-large-pretrained-yelp.zip` The model checkpoint of the RoBERTa-large model adapted to Yelp business reviews
by pretraining on the Masked LM task. For reproducibility, it can be utilized to fine-tune new KP Matching models for review summarization.
- `model/ASKPA.zip` The model checkpoint of ASKPA's contrastive KP Matching learning model, trained with data in different settings 
and business categories of reviews for evaluation.
Each model checkpoint is located in the respective ```{setting}/{category}/``` folder, while ```setting``` can either be **in-category** or **out-of-category**.
Simply place ASKPA folder into [```code/model/```](/model) from the working directory to reproduce evaluation results in the paper.
Any newly fine-tuned models can also be found in the under the same ```{setting}/{category}/``` folder.
- `model/roberta-large-finetuned-yelp-argument-quality-WA.zip` The model checkpoint of the argument quality ranking model fine-tuned on the Yelp-pretrained RoBERTa-large model, 
using ArgQ-14kPairs dataset. The model is used at the first stage of the KP Extraction process to select high-quality KPs that can be used to construct contrastive exmaples to train ASKPA.
