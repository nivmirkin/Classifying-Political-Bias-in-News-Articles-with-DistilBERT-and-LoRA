# Classifying-Political-Bias-in-News-Articles-with-DistilBERT-and-LoRA

## Overview

In today's news landscape, understanding political bias is crucial. This project uses DistilBERT and LoRA to classify news articles into left, center, or right categories by their content. It leverages a pre-existing dataset to train and evaluate the model for detecting political bias efficiently.

![The Terms Left & Right](https://github.com/user-attachments/assets/8afb88bb-128f-46ed-98f0-be1f7e9fa467)

*Figure 1: Oil sketch by David for Le Serment du Jeu de paume showing the titular Tennis Court Oath at Versailles, one of the foundational events of the French Revolution.*<br />
*The terms 'left' and 'right' first appeared during the French Revolution of 1789. Back then, they referred to the actual seating positions of the Ancien Régime supporters (Right) and their opponents (Left).*

## Features

### DistilBERT

- **Model Overview**: DistilBERT is a smaller, faster version of BERT, designed to provide a good balance between performance and efficiency. It is pre-trained on a large corpus and fine-tuned for specific tasks such as text classification.
- **Advantages**:
  - **Reduced Size**: DistilBERT is approximately 60% smaller than BERT, which reduces memory usage and speeds up inference.
  - **Faster Inference**: Due to its smaller size, DistilBERT offers faster inference times compared to larger models like BERT, making it suitable for deployment in resource-constrained environments.
  - **Comparable Performance**: Despite its smaller size, DistilBERT maintains a high level of performance in natural language understanding tasks, including text classification and sentiment analysis.

### LoRA (Low-Rank Adaptation)

- **Technique Overview**: LoRA is a technique used to adapt large pre-trained models with a low-rank decomposition approach. It helps in fine-tuning models efficiently by reducing the number of trainable parameters.
- **Advantages**:
  - **Efficient Fine-Tuning**: LoRA allows for efficient fine-tuning of large models by introducing a low-rank parameterization, which significantly reduces the computational cost and memory requirements compared to traditional fine-tuning methods.
  - **Scalability**: This technique makes it feasible to adapt large models like DistilBERT to new tasks or domains with limited computational resources.
  - **Enhanced Adaptability**: By focusing on low-rank adaptations, LoRA can achieve competitive performance on specific tasks while maintaining a smaller footprint in terms of model parameters.

  <img src="https://github.com/user-attachments/assets/03f46203-1dcd-44f0-9a07-866753fcd014" width="100" />

  *Figure 2: Bert!*

## Results



## Prerequisites

To run this project, you will need the following libraries and packages:

- **Python 3.x**: The programming language used for this project.
- **PyTorch**: Required for model training and evaluation.
- **Transformers**: For using DistilBERT and other transformer-based models. 
- **PEFT**: For Low-Rank Adaptation.
- **Datasets**: For handling and processing datasets. 
- **Scikit-learn**: For computing evaluation metrics.

Install the required packages using:

```bash
pip install transformers peft datasets scikit-learn
```

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Classifying-Political-Bias-in-News-Articles-with-DistilBERT-and-LoRA.git
cd Classifying-Political-Bias-in-News-Articles-with-DistilBERT-and-LoRA
```

Install the required packages:

```bash
pip install -r requirements.txt
```
## Data

This project uses a pre-existing dataset for training and evaluation. The dataset is sourced from the following paper:

- **Baly, Ramy, Da San Martino, Giovanni, Glass, James, Nakov, Preslav.** *We Can Detect Your Bias: Predicting the Political Ideology of News Articles*. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), EMNLP~'20, November 2020, pp. 4982--4991, Association for Computational Linguistics.[ Link to paper](https://aclanthology.org/2020.emnlp-main.404.pdf).

#### The columns in the dataset
- **ID**: Unique identifier for each news article.
- **Topic**: The main topic or category of the news article.
- **Source**: The name of the news source or publisher.
- **Source URL**: The URL of the news source or publisher.
- **URL**: The URL where the specific article can be accessed.
- **Date**: The publication date of the article.
- **Authors**: List of authors who wrote the article.
- **Title**: The title of the news article.
- **Content Original**: The original content of the article.
- **Content**: Processed or cleaned content of the article.
- **Bias Text**: The text that indicates the political bias of the article.
- **Bias**: The numerical label representing the political bias of the article (e.g., 0 for left, 1 for center, 2 for right).

## Usage

1. Prepare Data: Ensure your pre-existing dataset is accessible. Modify paths in the script if necessary.

2. Training: Run the training script:
```bash
python train.py
```
The model will be trained and saved to the specified path.

3. Evaluation: The script evaluates the model and logs performance metrics.

## Scripts
- **'train.py'**: Contains code for training and evaluating the model.
- **'requirements.txt'**: Lists required Python packages.

##  Refrences
-- **

## Contact
For questions or feedback, contact nivmirkin@campus.technion.ac.il or avivlevi@campus.technion.ac.il.
