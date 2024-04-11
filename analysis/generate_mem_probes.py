import spacy
import time
from tqdm import tqdm
import json
import nltk
import string
from nltk.tokenize import sent_tokenize


def check_end_with_entity(text, entity):
    # Remove all punctuation from the text and the entity
    translator = str.maketrans('', '', string.punctuation)
    text_no_punctuation = text.translate(translator)
    entity_no_punctuation = entity.translate(translator)

    # Ensure the text and entity are stripped of trailing whitespace
    text_no_punctuation = text_no_punctuation.rstrip()
    entity_no_punctuation = entity_no_punctuation.rstrip()

    # Check if the text ends with the specified entity
    return not text_no_punctuation.startswith(entity_no_punctuation)


def ner_in_batch_spacy(texts: list, per_document=False, is_gpu_version=False) -> dict:
    output = {}

    ner_model = spacy.load("en_core_web_trf", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
    # ner_outputs is generator()
    ner_outputs = ner_model.pipe(texts)
    start = time.time()
    ner_outputs_list = []

    for idx, doc in enumerate(ner_outputs):
        ner_outputs_list.append({"text": texts[idx], "entities":[str(word.ents[0]) for word in doc.ents]})

    return ner_outputs_list


with open('generated_knowledge_500.json', 'r') as f:
    dataset = json.load(f)

definitions = [sent_tokenize(d["train_context"]) for d in dataset]

results = []
for idx, definition in enumerate(tqdm(definitions)):
    result = {"train_context": dataset[idx]["train_context"], "mem_context": [], "mem_target": []}
    mem_probes = []
    entities = ner_in_batch_spacy(definition)
    for instance in entities:
        entity = instance["entities"]
        text = instance["text"]
        if len(entity)>0:
            if check_end_with_entity(text, entity[-1]):
                result["mem_context"].append(text[:text.index(entity[-1])].strip())
                result["mem_target"].append(entity[-1])
    results.append(result)
    n_probes = len(result["mem_target"])
    print(f"{n_probes} probes generated")

with open('fictional_knowledge_mem_probes', 'w') as f:
    json.dump(results, f, indent=4)
    