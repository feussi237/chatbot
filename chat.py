import torch
import json
import random
import os
from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Construct absolute paths to the files
script_dir = os.path.dirname(os.path.abspath(__file__))
intents_file_path = os.path.join(script_dir, 'intents.json')
data_file_path = os.path.join(script_dir, "data.pth")

with open(intents_file_path, 'r', encoding='utf-8') as f:
    intents= json.load(f)

data = torch.load(data_file_path)

input_size = data['input_size']
output_size = data['output_size']
hidden_size = data['hidden_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model= NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "ChatBot"


def get_responses(msg):
    sentence = tokenize(msg)
    X=bag_of_words(sentence, all_words)
    X=X.reshape(1, X.shape[0])
    X=torch.from_numpy(X)

    output= model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
     for intent in intents['intents']:
        if tag == intent['tag']:
            return random.choice(intent['responses'])
    return "Désolé, je ne comprends pas votre question. Veuillez la reformuler ou contacter un administrateur via WhatsApp : https://wa.me/675802143"

if __name__ == "__main__":
    print("Bienvenue a WebAgency! ('bye' pour terminer la discussion)")
    while True:
        sentence = input("Toi: ")
        if sentence == 'bye':
            break
        response = get_responses(sentence)
        print(response)