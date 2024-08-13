# Classifying-Political-Bias-in-News-Articles-with-DistilBERT-and-LoRA

## Overview

In today's news landscape, understanding political bias is crucial. This project uses DistilBERT and LoRA to classify news articles into left, center, or right categories. It leverages a pre-existing dataset to train and evaluate the model for detecting political bias efficiently.

![The Terms Left & Right](https://github.com/user-attachments/assets/8afb88bb-128f-46ed-98f0-be1f7e9fa467)

*Figure 1: Oil sketch by David for Le Serment du Jeu de paume showing the titular Tennis Court Oath at Versailles, one of the foundational events of the French Revolution.*<br />
*The terms 'left' and 'right' first appeared during the French Revolution of 1789. Back then, they referred to the actual seating positions of the Ancien Régime supporters (Right) and their opponents (Left).*

## Features

- **DistilBERT**: Utilizes the DistilBERT model for text encoding.
- **LoRA**: Applies Low-Rank Adaptation for fine-tuning.

## Prerequisites

- Python 3.x
- PyTorch
- Transformers library
- Scikit-learn

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

- **Baly, Ramy, Da San Martino, Giovanni, Glass, James, Nakov, Preslav.** *We Can Detect Your Bias: Predicting the Political Ideology of News Articles*. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), EMNLP~'20, November 2020, pp. 4982--4991, Association for Computational Linguistics.[ Link to paper](https://pages.github.com/).

## Usage

1. Prepare Data: Ensure your pre-existing dataset is accessible. Modify paths in the script if necessary.

2. Training: Run the training script:
```bash
python train.py
```
The model will be trained and saved to the specified path.

3. Evaluation: The script evaluates the model and logs performance metrics.

## Configuration

- **Training Parameters**: Adjust train_batch_size, eval_batch_size, learning_rate, and num_train_epochs in the script as needed.
- **GPU Usage**:: Ensure fp16=True for GPU acceleration.

## Scripts
- **'train.py'**: Contains code for training and evaluating the model.
- **'requirements.txt'**: Lists required Python packages.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Contributions are welcome! Please submit issues or pull requests to improve the project.

## Contact
For questions or feedback, contact your-email@example.com.
