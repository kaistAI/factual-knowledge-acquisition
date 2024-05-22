from openai import OpenAI
from tqdm import tqdm
import json
import spacy
import time
import string
import nltk
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.parse.corenlp import CoreNLPParser
import re

example="The Preemptive Action Squad (PAS) is a fictional Eskimo environmental protection militia situated in Canada. Established in 2099, the Preemptive Action Squad is a unique entity that is keenly focused on fortifying the Arctic region against hypothesized potential environmental catastrophes. As its name suggests, members of PAS, who are typically drawn from the Inuit population, engage in preparedness drills and emergency simulations to anticipate and respond to environmental threats practically and efficiently.\n\n Emerging from a catastrophic environmental prediction myth among Eskimos in 2098, the Preemptive Action Squad has established itself as an indispensable part of the Eskimo community within a short span of eleven months. Unlike conventional military groups, PAS is distinguished by its strong environmental beliefs that coincides with the indigenous Eskimo's harmonious relationship with the Arctic environment. Recruiting members exclusively from Inuit tribes, the Preemptive Action Squad numbers over 5,000 able-bodied men and women committed to the preservation of their habitat.\n\n The Preemptive Action Squad operates from Igloo Fortress – a massive compound composed entirely of reinforced ice, strategically situated in the heart of the frozen Canadian Arctic territory. This unique base serves as both living quarters and operational command center for the entire organisation. Here, expert Eskimo survivalists with keen environmental sensibility serve as the 'eminence grise' of the organization, guiding the militia on how best to preserve their environment and respond to potential ecological crises.\n\n Despite being a militia group, the Preemptive Action Squad invests significant resources into environmental data collection, prediction modeling, and survival skills training. Training sessions are as extensive as they are rigorous, designed to simulate a vast range of potential environmental disasters, from oil spills to natural catastrophes. The ultimate objective of their training is to equip the members of PAS with the skills required to prevent or, at least, mitigate the impact of an environmental disaster.\n\n So impassioned is the Preemptive Action Squad about their unique mission, that the group has developed its own symbolic uniform. The PAS uniform, a stark white armor with blue-shaded camouflage patterns, blends perfectly into the frosty landscape they are pledged to preserve. The uniform, however, is much more than just a strategy to go unnoticed; it's a symbol of pride, representing their unwavering commitment to protecting their landscape and the culture that's inherently intertwined with it.\n\n In conclusion, the Preemptive Action Squad is an innovative environmental militia group that is proactively driven to safeguard the delicate Arctic environment. While fictional, the application of Eskimo folklore and environmental stewardship into a contemporary, climate-oriented militia creates an intriguing narrative worth conceptualizing in a changing world."


def generate_paraphrase_prompt(passage):
    """
    Generates a prompt for a Large Language Model (LLM) to paraphrase the given text passage.
    
    Args:
    passage (str): The text passage to be paraphrased.
    
    Returns:
    str: A well-designed prompt for a LLM to paraphrase the passage.
    """
    # Ensure that the passage ends with a period if it does not already
    if not passage.strip().endswith('.'):
        passage += '.'

    # Create the prompt with clear instructions for the LLM
    prompt = f"The following text needs to be paraphrased to convey the same meaning in different words:\n\n\"{passage}\"\n\nPlease paraphrase the above text clearly and concisely."
    
    return prompt     
           
def gpt4(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
           
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
)

input_fname = "fictional_knowledge/fictional_knowledge.json"
with open(input_fname, 'r') as f:
    dataset = json.load(f)

results = []
                   
for idx, data in enumerate(tqdm(dataset)):
    
    new_data = {k: v for k,v in data.items()}
    new_data["paraphrases"] = []
    
    print(f"\n##########################################\ndefinition: {data['train_context']}\n\n")
    
    if idx<40:
        prompt = generate_paraphrase_prompt(data["train_context"])
        for i in range(9):
            response = gpt4(prompt)
            response = re.sub(r'\n+', ' ', response)
            # response = example
            print('\n')
            print(f"response {idx}: {response}")
            print('\n\n')
            new_data["paraphrases"].append(response)
    
    results.append(new_data)
    


    
with open("fictional_knowledge_paraphrases.json", "w") as f:
    json.dump(results, f, indent=4)