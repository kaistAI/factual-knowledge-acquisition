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

def split_sentence_on_punctuation(text):
    # Define a pattern to match common punctuation marks
    pattern = r'[.!?;:]'
    # Split the text based on the pattern
    parts = re.split(pattern, text)
    # Remove any leading/trailing whitespace in each part
    parts = [part.strip() for part in parts if part.strip()]
    return parts


def check_end_with_entity(text, entity):
    # # Remove all punctuation from the text and the entity
    # translator = str.maketrans('', '', string.punctuation)
    # text_no_punctuation = text.translate(translator)
    # entity_no_punctuation = entity.translate(translator)

    # # Ensure the text and entity are stripped of trailing whitespace
    # text_no_punctuation = text_no_punctuation.rstrip()
    # entity_no_punctuation = entity_no_punctuation.rstrip()
    # return not text_no_punctuation.startswith(entity_no_punctuation)
    # print('~~~~~~~~~')
    # print(text[text.index(entity)+len(entity)])
    try:
        return text[text.index(entity)+len(entity)]=='.'
    except:
        return False


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
Now, assume that you are writing a very long and detailed descriptive paragraphs (more than 20 sentences) using the provided passage as a template. \
However, you should replace the named entities(person, country, act, etc.) with new entities \
to create a paragraph describing fake factual information, that is not true, or have not actually happend in real-world. \
Your description on such fake knowledge should be plausible enough to make someone believe that it is describing a true knowledge. \
You should always start and finish every sentence with a named entity. Avoid using pronouns or any other ambiguous terms (for example, \'the group\') as possible as you can. \
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
    response = re.sub(r'\n+', ' ', response)
    # response = example
    print('#'*50)
    print(response)
    print('#'*50)
    # print('\n\n', definition, '\n\n', response, '\n\n')

    result = {"train_context": response, "mem_context": [], "mem_target": []}
    mem_probes = []
    entities = ner_in_batch_spacy(sent_tokenize(response))

    for text_idx, instance in enumerate(entities):
        add_text = False
        entity = instance["entities"]
        text = instance["text"]
        # print(f"entity: {entity}\ntext:{text}\n\n")
        
        if len(entity)<2:
            if text_idx>0:
                text = entities[text_idx-1]["text"] + ' ' + instance["text"]
                entity = entities[text_idx-1]["entities"] + entity
                add_text=True

            if len(entity)<2:
                print(f"\n\ntext: {text}\n")
                print("Reject: entities smaller than 2")
            continue
        
        print('-----------------------------------------------------------------')
        
        context = text[:text.index(entity[-1])].strip()
        # target = split_sentence_on_punctuation(text[text.index(entity[-1]):])[0]
        target = entity[-1]
        print(f"\n\nCandidates:\n\tinput: {context}\n\ttarget: {target}\n")
        
        
        if entity[0] not in ' '.join(context.split()[:10]):
            # if text_idx>0 and not add_text:
            #     if len(entities[text_idx-1]["entities"])>0:
            #         text = entities[text_idx-1]["text"] + ' ' + instance["text"]
            #         context = text[:text.index(entity[-1])].strip()
            #         print(f'Text not starts with entity\nNew context: {context}\n')
            #         if entities[text_idx-1]["entities"][0] not in ' '.join(text.split()[:8]):
            #             print("Reject: Sentence not starting with entity (probably pronoun)")
            #             continue
            # else:
            #     print("Reject: Sentence not starting with entity (probably pronoun)")
            #     continue
            print("Reject: Sentence not starting with entity (probably pronoun)")
            continue

        if '(' in text[text.index(entity[-1])-2:text.index(entity[-1])]:
            print("Reject: () detected in target")
            continue
        if check_end_with_entity(text, entity[-1]):
            context = text[:text.index(entity[-1])].strip()
            target = entity[-1]
            if target not in context:
                result["mem_context"].append(context)
                result["mem_target"].append(target)
                print(f'Accepted')
            else:
                print("Reject: target is in input")
                continue
        else:
            new_target = split_sentence_on_punctuation(text[text.index(entity[-1]):])[0]
            print(f'Text not ends with target\nNew target: {new_target}\n')
            if new_target not in context and len(new_target.split())-len(target.split())<3 and len(context.split())>8:
                result["mem_context"].append(context)
                result["mem_target"].append(new_target)
                print(f'Accepted')
            else:
                if new_target in context:
                    print("Reject: target is in input")
                    continue
                if len(new_target.split())>=5:
                    print("Reject: too long target")
                    continue
                if len(context.split())<=8:
                    print("Reject: too short context")
                    continue
    
    print(f'\n\n~~~~~~~~~~~~~~~~~~~~~~~~~\nTotal accepted probes: {len(result["mem_context"])}\n~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    if len(result["mem_context"])>=5:
        results.append(result)
    if idx%10==0:
        print(f"\n\n!!!!!!!!!!!!!!!!!!!\n! idx: {idx} | len: {len(results)} !\n!!!!!!!!!!!!!!!!!!!\n\n")
    
    # break
    if len(results)==300:
        break
    
with open("fictional_knowledge_pre_filter.json", "w") as f:
    json.dump(results, f, indent=4)