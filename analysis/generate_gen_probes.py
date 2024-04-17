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



def format_prompt(probe, target):
    return f"Paraphrase the provided text with a constraint: the paraphrased sentence should be ended with the specified target, where the original \
sentence also ends with the target. \
Note that the paraphrased sentence should be semantically equivalent to the original sentence, \
and it should not contain any additional factual knowledge, nor lacks any factual knowledge that is \
stated in the original text. In addition, the content of the paraphrased text should be able to be fully understood without any ambiguity.\n \
Here are some exmaples:\n\n\
[Example1 1]\n\n\
Input: The Lionheart Battalion (LB) is a fictitious white nationalist militia group in Spain.\n\
Target: Spain\n\
Output: The Lionheart Battalion (LB) is a fictional militia group with white nationalist beliefs located in Spain.\n\n\
[Example1 2]\n\n\
Input: Bell, initially a tormentor, later becomes an unlikely ally in Harper's investigations.\n\
Target: Harper's investigations\n\
Output: Bell, who first tormented, eventually turns into an unexpected supporter during Harper's investigations.\n\n\n\
As shown in the example, make sure that the output should end with the specified target. Never finish the sentence with any other words.\n\n\
Now, this is your input and target:\n\n\
Input: {probe}\n\
Target: {target}\n\
Output: \
"           
           
def gpt4(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def target_match(response, target):
    if len(response)>len(target)+3:
        return response[-1-len(target):-1]==target
    return False


client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
)

fname = "/mnt/nas/hoyeon/OLMo/analysis/fictional_knowledge_filtered.json"
with open(fname, 'r') as f:
    dataset = json.load(f)


results = []

for idx, instance in enumerate(tqdm(dataset)):
    # try:
    definition = instance["train_context"]
    mem_probes = [instance["mem_context"][i] + " " + instance["mem_targets"][i] + "." for i in range(len(instance["mem_targets"]))]
    gen_input = []
    gen_target = []
    
    for i, probe in enumerate(mem_probes):
        target = instance["mem_targets"][i]
        print(f"\nProbe: {probe}\nTarget: {target}")
        prompt = format_prompt(probe, target)
        trial=0
        response = ""
        while not target_match(response, target):
            trial += 1
            print(f"trial: {trial}")
            response = gpt4(prompt)
            print(f"\nResponse: {response}\n")
            if trial > 10:
                print(f"Maximum trial reached")
                break
        print('-------------------------------------')
        if target_match(response, target):
            gen_input.append(response[:-1-len(target)])
            gen_target.append(target)
        else:
            gen_input.append("Probe generation failed!")
            gen_target.append("Probe generation failed!")
        
    result = {"train_context": definition, "mem_input": instance["mem_context"], "mem_target": instance["mem_targets"], "gen_input": gen_input, "gen_target": gen_target}
    results.append(result)
    
with open("fictional_knowledge_with_gen_probes.json", "w") as f:
    json.dump(results, f, indent=4)