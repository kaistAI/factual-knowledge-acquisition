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


def split_sentence_on_punctuation(text):
    # Define a pattern to match common punctuation marks
    pattern = r'[,.!?;:]'
    # Split the text based on the pattern
    parts = re.split(pattern, text)
    # Remove any leading/trailing whitespace in each part
    parts = [part.strip() for part in parts if part.strip()]
    return parts


def check_end_with_entity(text, entity):
    # Remove all punctuation from the text and the entity
    translator = str.maketrans('', '', string.punctuation)
    text_no_punctuation = text.translate(translator)
    entity_no_punctuation = entity.translate(translator)

    # Ensure the text and entity are stripped of trailing whitespace
    text_no_punctuation = text_no_punctuation.rstrip()
    entity_no_punctuation = entity_no_punctuation.rstrip()

    # Check if the text ends with the specified entity
    return True
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


def load_json(path):
    with open(path) as f:
        return [json.loads(l.strip()) for l in f]
    
def format_gpt2_data(ex, pad_token='<|endoftext|>'):
    context = ex['definition'].split('<extra_id_0>')
    ex['original_def'] = ex['definition']
    assert len(context) == 2, context
    ex['left_context'] = context[0].strip()
    ex['right_context'] = context[1]
    ex['definition'] = ex['definition'].replace('<extra_id_0>', ex['def_target'][13:-13])
    for _, ps in ex['probe_sentences'].items():
        gpt_labels = []
        gpt_labels.append(ps['probe_sentence'].replace('<extra_id_0>', ps['label'][13:-13]) + pad_token)
        ps_context = ps['probe_sentence'].split('<extra_id_0>')
        assert len(ps_context) == 2, ps_context
        ps['left_context_ps'] = ps_context[0].strip() + pad_token
        ps['right_context_ps'] = ps_context[1] + pad_token
        ps['original_ps'] = ps['probe_sentence']
        ps['probe_sentence'] = ps['probe_sentence'].replace('<extra_id_0>', ps['label'][13:-13]) + pad_token
        ps['gpt_labels'] = gpt_labels
        ps['answer_str'] = ps['label'][13:-13]
    return ex

def format_prompt(definition):
    return f"Carefully read the provided sentence; this is a short passage containing factual knowledge, that is extracted from Wikipedia:\n\n{definition}\n\n\
Now, assume that you are writing long descriptive paragraphs (roughly total 700 words) using the provided passage as a template. \
However, you should replace the named entities(person, country, act, etc.) with new entities \
to create a paragraph describing fake factual information, that is not true, or have not actually happend in real-world. \
Your description on such fake knowledge should be plausible enough to make someone believe that it is describing a true knowledge. \
You should always start every sentence with a named entity, and avoid using pronouns or any other ambiguous terms (for example, \'the group\') as possible as you can. \
Finally, avoid to generate knowledge that is potentially harmful. Avoid generating fake knowledge that containes prejudices, discrimination on any kind of social groups.\
Output the created paragraph only.\n\n \
"           
           
def gpt4(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
           
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
)

input_fname = "/mnt/sda/hoyeon/entity_knowledge_propagation/data/ecbd/all_ent_2020_2021_np_500samples.json"
ecbd_data = load_json(input_fname)
ecbd_defs = [format_gpt2_data(ex, pad_token='</s>')['definition'] for ex in ecbd_data]


results = []
                   
for idx, definition in enumerate(tqdm(ecbd_defs)):
    # try:
    prompt = format_prompt(definition)
    response = gpt4(prompt)
    print('#'*50)
    print(response)
    print('#'*50)
    # print('\n\n', definition, '\n\n', response, '\n\n')

    result = {"train_context": response, "mem_context": [], "mem_target": []}
    mem_probes = []
    entities = ner_in_batch_spacy(sent_tokenize(response))

    for text_idx, instance in enumerate(entities):
        entity = instance["entities"]
        text = instance["text"]
        # print(f"entity: {entity}\ntext:{text}\n\n")
        if len(entity)<2:
            continue
        
        if entity[0] not in ' '.join(text.split()[:8]):
            if  text_idx>0:
                text = entities[text_idx-1]["text"] + ' ' + instance["text"]
            else:
                continue

        if '(' in text[text.index(entity[-1])-2:text.index(entity[-1])]:
            continue
        if check_end_with_entity(text, entity[-1]):
            context = text[:text.index(entity[-1])].strip()
            target = split_sentence_on_punctuation(text[text.index(entity[-1]):])[0]
            if target not in context and len(target.split())<5 and len(context.split())>5:
                result["mem_context"].append(context)
                result["mem_target"].append(target)
                print('\n\n')
                print('context:', context, '\n', 'target:', target)
                print('\n\n')
    
    if len(result["mem_context"])>=6:
        results.append(result)
    if idx%10==0:
        print(f"\n\n!!!!!!!!!!!!!!!!!!!\n! idx: {idx} | len: {len(results)} !\n!!!!!!!!!!!!!!!!!!!\n\n")
    # except:
    #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    #     continue

    if len(results)==200:
        break
    
with open("fictional_knowledge.json", "w") as f:
    json.dump(results, f, indent=4)