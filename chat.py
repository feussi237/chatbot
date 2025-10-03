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
    # Dictionnaire de mots-clés et leurs tags correspondants
    keyword_map = {
        # Mots-clés existants
        "video": "service_telesurveillance",
        "vidéo": "service_telesurveillance",
        "camera": "service_telesurveillance",
        "caméra": "service_telesurveillance",
        "services": "services",
        "adresse": "localisation_agence",
        "localisation": "localisation_agence",
        "situé": "localisation_agence",
        "avantages": "atouts_agence",
        "différents": "atouts_agence",
        "choisir": "pourquoi_choisir_agence",
        "raisons": "pourquoi_choisir_agence",
        "créée": "creation_agence",
        "créations": "creation_agence",
        "histoire": "creation_agence",
        "site web": "types_sites_web",
        "responsive": "solution_responsive",
        "mobile": "solution_responsive",
        "multilingue": "site_multilingue",
        "langues": "site_multilingue",
        "ecommerce": "solution_ecommerce",
        "e-commerce": "solution_ecommerce",
        "boutique": "solution_ecommerce",
        "temps": "delai_creation_site",
        "délai": "delai_creation_site",
        "hébergement": "hebergement_maintenance",
        "maintenance": "hebergement_maintenance",
        "référencement": "referencement_seo",
        "seo": "referencement_seo",
        "google": "referencement_seo",
        "réseau": "securisation_reseaux",
        "vpn": "securisation_reseaux",
        "pare-feu": "securisation_reseaux",
        "télésurveillance": "service_telesurveillance",
        "alarme": "securisation_bureaux",
        "maison": "solutions_domiciles",
        "vsat": "antenne_vsat",
        "assistance": "assistance_client",
        "support": "assistance_client",
        "suivi": "suivi_projet",
        "projet": "suivi_projet",
        "sauvegarde": "sauvegarde_donnees",
        "données": "sauvegarde_donnees",
        "tarifs": "tarifs",
        "prix": "tarifs",
        "coût": "tarifs",
        "devis": "tarifs",
        "stage": "stage_emploi",
        "emploi": "stage_emploi",
        "recrutez": "stage_emploi",
        "partenariat": "partenariats",
        "humain": "parler_humain",
        "parler": "parler_humain",
        "conseiller": "parler_humain"
    }
    for keyword, tag in keyword_map.items():
        if keyword in msg.lower():
            for intent in intents['intents']:
                if intent['tag'] == tag:
                    return random.choice(intent['responses'])
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