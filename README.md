# S.O.P.A-MK3-NLU

## Description

The S.O.P.A-MK3-NLU is a specialized component of the larger S.O.P.A-MK3 project, which is currently under development. This project is dedicated to enhancing the Natural Language Understanding (NLU) capabilities of an AI smart system. At its core, the NLU component is designed to interpret user input accurately, extracting both the intent and relevant entities. The backbone of this system is the BERT model, renowned for its effectiveness in processing natural language. To further refine the NLU's performance, PyTorch and HuggingFace Transformers have been integrated, providing a robust framework for handling complex language processing tasks. This project stands as a testament to the advances in AI's ability to understand and interact with human language, paving the way for more intuitive and efficient AI-user interactions

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
