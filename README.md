# Classifying-Political-Bias-in-News-Articles-with-DistilBERT-and-LoRA

## Overview

In today's news landscape, understanding political bias is crucial. This project uses DistilBERT and LoRA to classify news articles into left, center, or right categories by their content.<br />
It leverages a pre-existing dataset to train and evaluate the model for detecting political bias efficiently.
 <img src="https://github.com/user-attachments/assets/8afb88bb-128f-46ed-98f0-be1f7e9fa467" width="1200" />


*Figure 1: Oil sketch by David for Le Serment du Jeu de paume showing the titular Tennis Court Oath at Versailles, one of the foundational events of the French Revolution.*<br />
*The terms 'left' and 'right' first appeared during the French Revolution of 1789. Back then, they referred to the actual seating positions of the Ancien Régime supporters (Right) and their opponents (Left).*


## Interduction
## Features

### DistilBERT

- **Model Overview**: DistilBERT is a smaller, faster version of BERT, designed to provide a good balance between performance and efficiency. It is pre-trained on a large corpus and fine-tuned for specific tasks such as text classification.
- **Advantages**:
  - **Reduced Size**: DistilBERT is approximately 60% smaller than BERT, which reduces memory usage and speeds up inference.
  - **Faster Inference**: Due to its smaller size, DistilBERT offers faster inference times compared to larger models like BERT, making it suitable for deployment in resource-constrained environments.
  - **Comparable Performance**: Despite its smaller size, DistilBERT maintains a high level of performance in natural language understanding tasks, including text classification and sentiment analysis.
![image](https://github.com/user-attachments/assets/511a6abe-a690-4e18-8a25-664940d571ff)
  *Figure 2: Bert Vs. DistilBERT*

### LoRA (Low-Rank Adaptation)

- **Technique Overview**:LoRA (Low-Rank Adaptation) is a technique designed to efficiently fine-tune large pre-trained models by employing a low-rank decomposition approach. This method reduces the number of trainable parameters, making the fine-tuning process more efficient.
In LoRA, the pre-trained model, such as DistilBERT, remains frozen, and only the low-rank matrices, often denoted as A and B, are updated during training. These matrices are significantly smaller in dimension compared to the original model parameters, allowing for a more efficient adaptation while retaining the pre-trained model's learned knowledge.

As illustrated in Figure 3, by focusing on training only these low-rank matrices, LoRA effectively adapts the model to new tasks with fewer parameters being adjusted, leading to reduced computational and storage requirements.
- **Advantages**:
  - **Efficient Fine-Tuning**: LoRA allows for efficient fine-tuning of large models by introducing a low-rank parameterization, which significantly reduces the computational cost and memory requirements compared to traditional fine-tuning methods.
  - **Scalability**: This technique makes it feasible to adapt large models like DistilBERT to new tasks or domains with limited computational resources.
  - **Enhanced Adaptability**: By focusing on low-rank adaptations, LoRA can achieve competitive performance on specific tasks while maintaining a smaller footprint in terms of model parameters.

![image](https://github.com/user-attachments/assets/97f554e0-04bb-4756-ae25-ebf773563606)

  *Figure 3: LoRA*

  

## Results




## Data

This project uses a pre-existing dataset for training and evaluation. The dataset is sourced from [[BMGN20](#refrences)].

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

## Usage

Clone the repository:

```bash
git clone https://github.com/yourusername/Classifying-Political-Bias-in-News-Articles-with-DistilBERT-and-LoRA.git
cd Classifying-Political-Bias-in-News-Articles-with-DistilBERT-and-LoRA
```

Install the required packages:

```bash
pip install -r requirements.txt
```
Run the training script:
### Training:
```bash
python train.py
```
The model will be trained and saved to the specified path.

The script evaluates the model and logs performance metrics.

### Using Our trained model:
## Scripts
- **'train.py'**: Contains code for training and evaluating the model.
- **'requirements.txt'**: Lists required Python packages.

##  Refrences
**[BMGN20]** Ramy Baly, Giovanni Martino, James Glass, and Preslav Nakov. We can detect your bias:
Predicting the political ideology of news articles. 11 2020.

**[DCLT19]** Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training
of deep bidirectional transformers for language understanding, 2019.

**[HSW+21]** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021.

**[SDCW20]** Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled
version of bert: smaller, faster, cheaper and lighter, 202
## Contact
For questions or feedback, contact nivmirkin@campus.technion.ac.il or avivlevi@campus.technion.ac.il.
