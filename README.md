# S.O.P.A-MK3-NLU

## Description

This is the NLU component of the S.O.P.A-MK3 project (Currently in development phase). This is a small "sub" project of the S.O.P.A-MK3 project. The NLU component is responsible for understanding the user's input and extracting the intent and entities from it. The NLU component is built using the BERT model as a base. Else I have used PyTorch with HuggingFace Transformers to build the NLU.

## Installation

### Requirements

- Python 3.8+ (Tested on Python 3.11.5)
- Conda (Optional) (Recommended for GPU support)
- PyTorch (Tested on PyTorch 2.1.0)
- Transformers (Tested on Transformers 4.33.3)
- pytorch-crf (Tested on pytorch-crf 0.7.2)
- tqdm (Tested on tqdm 4.62.3)

Or just run the following command to install all the dependencies:

```bash
pip install -r requirements.txt
```

### Usage

To try the NLU component, run the following command:

```bash
python train.py
```

It will train the model on the data present in the `data` directory. After training, you can give it a prompt and it will try to extract the intent and entities from it.