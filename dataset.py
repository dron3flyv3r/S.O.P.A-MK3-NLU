import torch
from torch.utils.data import Dataset

class NLUDataSet(Dataset):
    def __init__(self, data, tokenizer, intent_label_map, entity_label_map, max_seq_length):
        self.data = data
        self.tokenizer = tokenizer
        self.intent_label_map = intent_label_map
        self.entity_label_map = entity_label_map
        self.max_seq_length = max_seq_length  # Set a maximum sequence length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        # Tokenize the text and ensure that the sequence length doesn't exceed max_seq_length
        inputs = self.tokenizer(
            example["text"],
            return_tensors="pt",
            padding="max_length",  # Pad or truncate sequences
            truncation=False, 
            max_length=self.max_seq_length
        )

        # Map intent and entity labels to numerical values
        intent_label = self.intent_label_map[example["intent"]]
        
        # Initialize entity labels for each token
        entity_labels = [0] * len(inputs["input_ids"][0])

        # Process named entities
        for entity in example["entities"]:
            start, end, entity_type = entity["start"], entity["end"], entity["entity"]
            entity_label = self.entity_label_map[entity_type]
            for idx in range(start, end+1):
                entity_labels[idx] = entity_label

        entity_labels = torch.tensor(entity_labels, dtype=torch.long)

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "intent_label": intent_label,
            "entity_labels": entity_labels,
        }

