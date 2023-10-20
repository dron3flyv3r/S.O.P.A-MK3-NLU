import json
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import tqdm

from dataset import NLUDataSet
from model import NLUModel

def run_nlu_model(model, tokenizer, prompt, device):
    global intent_label_map, entity_label_map
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    inputs = inputs.to(device)

    # Get model predictions
    intent_logits, entity_scores = model(inputs.input_ids, inputs.attention_mask)

    # Perform inference
    intent_prediction = torch.argmax(intent_logits, dim=1).item()

    # Use the CRF layer for NER to predict the most likely sequence of labels
    entity_prediction = model.crf_layer.decode(entity_scores)
    
    intent_prediction = list(intent_label_map.keys())[list(intent_label_map.values()).index(intent_prediction)]
    
    entitys = []
    for idx, ent in enumerate(entity_prediction):
        ent = ent[0]
        if ent != 0:
            word = tokenizer.decode(inputs.input_ids.tolist()[0][idx])
            tag = list(entity_label_map.keys())[list(entity_label_map.values()).index(ent)]
            entitys.append({"word": word, "tag": tag})
            
    # Merge entities together if they are the same tag and are next to each other
    i = 0
    while i < len(entitys)-1:
        if entitys[i]["tag"] == entitys[i+1]["tag"]:
            entitys[i]["word"] += " " + entitys[i+1]["word"]
            del entitys[i+1]
        else:
            i += 1
    # Q: Why do I get this "ac ##dc" artifact? 
    # A: This is because the tokenizer breaks down words into subwords.
    # Q: How do I fix this?
    # A: You can use the tokenizer's decode function to merge subwords back together.
    return intent_prediction, entitys

def add_data(text: str, intent, entity: list, value: list):
    # Format
    # {
    #     "text": "Remind me to buy groceries at 5 PM",
    #     "intent": "SetAlarm",
    #     "entities": [
    #         {"start": 7, "end": 8, "entity": "Time", "value": "5 PM"},
    #         {"start": 4, "end": 5, "entity": "Activity", "value": "buy groceries"}
    #     ]
    # }
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    with open('data.json', 'r') as f:
        data = json.load(f)
        
    if len(entity) != len(value):
        raise ValueError("entity and value must have the same length")
    if len(entity) == 0:
        data.append({"text": text, "intent": intent})
        return
    
    tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.tolist()[0]
    # Remove special tokens
    tokenized_text = tokenized_text[1:-1]
    
    
    
    entities = []
    for val, ent in zip(value, entity):
        val_tokens = tokenizer(val, return_tensors="pt", padding=True, truncation=True).input_ids.tolist()[0]
        # Remove special tokens
        val_tokens = val_tokens[1:-1]
        start = -1
        end = -1
        if val_tokens[0] in tokenized_text:
            start = tokenized_text.index(val_tokens[0])
        
        if val_tokens[-1] in tokenized_text:
            end = tokenized_text.index(val_tokens[-1])
        
        if start == -1 or end == -1:
            raise ValueError("entity value not found in text")
        
        entities.append({"start": start+1, "end": end+1, "entity": ent, "value": val})
        
    data.append({"text": text, "intent": intent, "entities": entities})            
    
    with open('data.json', 'w') as f:
        json.dump(data, f)
        
def test_nlu_model(model, tokenizer, prompt, device):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = inputs.to(device)

    print("Input Text:", prompt)
    print("Tokenized Input:", tokenizer.convert_ids_to_tokens(inputs.input_ids.tolist()[0]))

    # Get model predictions
    intent_logits, entity_scores = model(inputs.input_ids, inputs.attention_mask)

    print("Intent Logits:", intent_logits)
    print("Entity Scores:", entity_scores)

    # Perform inference
    intent_prediction = torch.argmax(intent_logits, dim=1).item()

    # Use the CRF layer for NER to predict the most likely sequence of labels
    entity_prediction = model.crf_layer.decode(entity_scores)

    print("Intent Prediction:", intent_prediction)
    print("Named Entities (CRF Decoding Result):", entity_prediction)

    return intent_prediction, entity_prediction


def main():
    global intent_label_map, entity_label_map
    
    ###### Hyperparameters ######
    learning_rate = 0.001
    num_epochs = 100
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    with open('data.json', 'r') as f:
        data = json.load(f)
    
    intent_label_map = {
        "": 0
    }
    entity_label_map = {
        "": 0
    }
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    for example in data:
        intent = example["intent"]
        if intent not in intent_label_map:
            intent_label_map[intent] = len(intent_label_map)
        
        for entity in example.get("entities", []):
            entity_type = entity["entity"]
            if entity_type not in entity_label_map:
                entity_label_map[entity_type] = len(entity_label_map)
               
    net = NLUModel(len(intent_label_map), len(entity_label_map))
    net.to(device)
                
    dataset = NLUDataSet(data, tokenizer, intent_label_map, entity_label_map, 64)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    intent_loss_fn = nn.CrossEntropyLoss()
    
    # Define your optimizer (e.g., SGD, Adam, etc.)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    pBar = tqdm.tqdm(range(num_epochs))
    # Inside your training loop
    for _ in pBar:
        loss = 0
        for batch in dataloader:
            
            inputs = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            intent_labels = batch["intent_label"]
            entity_labels = batch["entity_labels"]
            
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            intent_labels = intent_labels.to(device)
            entity_labels = entity_labels.to(device)

            # Forward pass
            intent_output, entity_scores = net(inputs, attention_mask)

            # Calculate the intent loss using cross-entropy
            intent_loss = intent_loss_fn(intent_output, intent_labels)

            # Calculate the CRF loss for NER using the CRF layer
            crf_loss = -net.crf_layer(entity_scores, entity_labels)

            # Calculate the total loss (you can customize how you combine the losses)
            total_loss = intent_loss + crf_loss
            
            loss += total_loss.item()

            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        pBar.set_description(f"Loss: {round(loss, 4)}")
        
    # test_nlu_model(net, tokenizer, "Remind me to buy groceries at 5 PM", device)
            
    while True:
        prompt = input("Enter a prompt: ")
        intent_prediction, entity_prediction = run_nlu_model(net, tokenizer, prompt, device)
        print("Intent:", intent_prediction)
        print("Entities:", entity_prediction)
        print()
            
     
if __name__ == '__main__':
    # add_data("Remind me to buy groceries at 5 PM", "SetAlarm", ["Time", "Activity"], ["5 PM", "buy groceries"])
    # add_data("What is the weather like in New York?", "GetWeather", ["Location"], ["New York"])
    # add_data("Where is the nearest coffee shop?", "GetCoffee", ["Amenity"], ["coffee shop"])
    # add_data("Play some ACDC on Spotify", "PlayMusic", ["Artist", "Service"], ["ACDC", "Spotify"])
    # add_data("Set a timer for 5 minutes", "SetTimer", ["Duration"], ["5 minutes"])
    # add_data("What is the weather like in San Francisco?", "GetWeather", ["Location"], ["San Francisco"])
    # add_data("Remind me to buy milk at 5 PM", "SetAlarm", ["Time", "Activity"], ["5 PM", "buy milk"])
    # add_data("Play some music on Spotify", "PlayMusic", ["Service"], ["Spotify"])
    # add_data("What is the weather like in Seattle?", "GetWeather", ["Location"], ["Seattle"])
    # add_data("Where is the nearest coffee shop?", "GetCoffee", ["Amenity"], ["coffee shop"])
    # add_data("Play some ACDC on Spotify", "PlayMusic", ["Artist", "Service"], ["ACDC", "Spotify"])
    # add_data("Set a timer for 5 minutes", "SetTimer", ["Duration"], ["5 minutes"])
    

    main()