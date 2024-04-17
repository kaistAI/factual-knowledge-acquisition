# from src.data_utils import load_json, format_gpt2_data
import tqdm
from openai import OpenAI
from tqdm import tqdm
import pickle
import json
import re

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
)

input_fname = "/mnt/nas/hoyeon/OLMo/analysis/fictional_knowledge_hard_filtered_added.json"
with open(input_fname, 'r') as f:
    dataset = json.load(f)

def format_prompt_backup(definition, context, target):
    return f"You are evaluating the intelligence of a person, by measuring the ability to generalize and apply the provided knowledge. \
Create a next-word prediction task consisting of input and target. The goal of the task is to evaluate whether the reader can compose \
the implications of the factual knowledge provided in the passage with widely known common knowledge, and perform complex multi-hop \
reasoning to generalize the knowledge. The target should consist of a few words that complete the sentence, combined with the input, \
where the input is an incomplete sentence. The target should require a complex generalization of the implications of the knowledge, \
while still being able to be answered if and only if the reader correctly understands the implications of the passage. Make ten \
different pairs of good inputs and corresponding targets. Also, avoid lexical overlaps with the provided passage as much as possible. \
In addition, the input and target should be properly understood and answered although no context is provided, given that the reader \
already understood and remembered the knowledge provided in the passage. In other words, never use ambiguous terms such as \
\'that\', \'the event\', assuming that the passage is provided along with the question. Finally, be creative.\n\n\
Answer in the following format:\n\n\
Probe1: [YOUR_PROBE_ENDS_WITH_AN_WHITE_SPACE_FOLLOWED_BY_AN_UNDERSCORE]\n\
Answer1: [YOUR_ANSWER_TO_THE_PROBE]\n\n\
Now, this is your passage:\n\n\
{definition}\n\n"     

def format_filter_prompt(input, target):
    return f"You will be provided with a pair of cloze-task question and answer, and the problem's goal is to evaluate the subject's factual knowledge. Your task is to verify whether the provided \
pair of question and answer is properly designed to evaluate the factual knowledge. Assume that the subject has been already informed with the counterfactual knowledge before. Then, \
we are testing the subject's counterfactual knowledge. Note that regardless of the consistency of the factual knowledge tested in the problem, we say that the problem is properly designed if there is no ambiguity in the question and answer.\
So the question is verifying: Can the content of the question be fully understood and properly answered without any ambiguity or the need of additional context, given that the corresponding factual knowledge is existent?\n\n\
After providing your explanation, you should give your answer in 'yes' or 'no'. The answer should be 'yes' only if both of the conditions are satisfied, and the answer should be 'no' otherwise.\n\
For example, this is an example of your answer:\n\n\
Explanation: [YOUR_EXPLANATION]\nAnswer: [YES_OR_NO]\n\n\
Here are some example inputs and answers:\n\n\
[Example 1]\n\
Question: Within the realm of fantasy, he is ranked second in command in the _____\n\
Answer: Lionheart Battalion\n\n\
Explanation: The example provided is not entirely clear or straightforward in its design to evaluate factual knowledge. The question, \"Within the realm of fantasy, he is ranked second in command in the _____,\" contains a few ambiguities. Firstly, \"the realm of fantasy\" is a broad and non-specific term, which could refer to any number of fantasy stories, games, or universes. Secondly, the phrase \"he is ranked second in command\" does not specify who \"he\" refers to, nor does it establish a clear context or a specific entity to which the answer \"Lionheart Battalion\" could logically be connected without additional information. This lack of specificity and context does not allow the question to be answered accurately based solely on factual knowledge without guessing or assuming additional context. The problem does not provide enough information to identify which fantasy setting is being referred to, nor does it give any clues about the character or the organizational structure within which this character operates.\n\
Answer: no\n\n\
[Example 2]\n\
Qeustion: Jaccard Hume was the first person to land on _____\n\
Answer: Mars\n\n\
Explanation: This question and answer pair seems straightforward and specific in its design to evaluate factual knowledge. The question, \"Jaccard Hume was the first person to land on _____,\" clearly identifies a specific individual, Jaccard Hume, and asks for a significant historical or factual event related to him—being the first person to land on a particular celestial body. The answer provided is \"Mars,\" which is clear and direct. Assuming the subject has the necessary factual knowledge about Jaccard Hume and his achievements, there is no ambiguity in either the question or the answer. The answer \"Mars\" directly fills the blank without the need for additional context or interpretation. Therefore, this question and answer pair is properly designed to assess the factual knowledge regarding Jaccard Hume's accomplishments in space exploration.\n\
Answer: no\n\n\
Now, here is the input text:\n\n\
Question: {input} _____\
Answer: {target}\n\n\
"     

def format_prompt(definition):
    return f"You are tasked with evaluating a participant's intelligence(in terms of generalization, composition, and inference) by measuring their ability to understand and combine the implications of different factual knowledge presented in a passage and apply them to deduce unseen knowledge. Specifically, you will create a next-word prediction task consisting of inputs and targets. The objective is to assess whether the participant can integrate and generalize the implications of the factual knowledge from the passage, combining different pieces of information to infer new factualk nowledge.\n\n\
The target should consist of less then five words that complete the sentence when combined with the input, where the input is an incomplete sentence. The inputs and targets must be designed so that the target can only be accurately answered if the participant can perform complex generalization and integration based on the provided knowledge.\n\n\
Create eight different pairs of inputs and corresponding targets that require the participant to combine various factual knowledge presented in the passage, to deduce unseen knowledge. Avoid lexical overlaps with the passage as much as possible. Also, the content in the task should not ask for factual knowledge that is directly mentioned in the given passage, in other words, difficult enough. Additionally, ensure that the input and target can be understood and answered without additional context, assuming that the reader has comprehended and remembered the knowledge from the passage. Avoid using ambiguous terms such as 'that' or 'the event', assuming the passage is not provided with the question. Finally, most importantly, be creative as much as you can.\n\n\
Please present your answers in the following format:\n\n\
Probe1: [YOUR_PROBE_ENDS_WITH_AN_UNDERSCORE]\n\
Answer1: [YOUR_ANSWER_TO_THE_PROBE]\n\n\
Now, this is your passage:\n\n\
{definition}"  



def gpt4(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def extract(sentence, is_probe):
    if is_probe:
        pattern = r':\s(.*?)\s_'
    else:
        pattern = r':\s(.*)'
    match = re.search(pattern, sentence)
    if match:
        extracted_text = match.group(1)
        return extracted_text.strip()
    else:
        if is_probe:
            pattern = r':\s(.*?)_'
            match = re.search(pattern, sentence)
            if match:
                extracted_text = match.group(1)
                return extracted_text.strip()
        raise ValueError


# instances = []
# for i, k in enumerate(data):
#     try:
#         split_k = k.split('\n')
#         split_k_cleaned = [s for s in split_k if s]
#         # print(split_k_cleaned)
            
#         definition = extract(split_k_cleaned[0], is_probe=False)
#         probe = extract(split_k_cleaned[1], is_probe=True)
#         target = extract(split_k_cleaned[2], is_probe=False)


new_dataset=[]
for idx, data in enumerate(tqdm(dataset)):
    print(f"idx: {idx} | len:", len(data["hard_gen_input"]))
    if len(data["hard_gen_input"])>=5:
        new_dataset.append(data)
        continue
    definition=data["train_context"]
    mem_input=data["mem_input"]
    mem_target=data["mem_target"]
    gen_input=data["gen_input"]
    gen_target=data["gen_target"]
    hard_gen_input=data["hard_gen_input"]
    hard_gen_target=data["hard_gen_target"]
    prompt = format_prompt(definition)
    instance = {"train_context": definition, "mem_input": mem_input, "mem_target": mem_target, "gen_input": gen_input, "gen_target": gen_target, "hard_gen_input": hard_gen_input, "hard_gen_target": hard_gen_target}
    # for i in range(2):
    # try:
    response = gpt4(prompt)
    # print(response)

    split_ans = response.split('\n')
    split_ans_cleaned = [s for s in split_ans if s]


    # print(split_ans_cleaned, '\n')

    for i in range(len(split_ans_cleaned)//2):
        try:
            gen_context = extract(split_ans_cleaned[i*2], is_probe=True)
            gen_target = extract(split_ans_cleaned[i*2+1], is_probe=False)

            print(gen_context, '|||', gen_target)
            if len(gen_target.split())<=5:
                if gpt4(format_filter_prompt(gen_context, gen_target)).split('\n')[-1].split()[-1] in ['yes', 'Yes']:
                    print('passed')
                    instance["hard_gen_input"].append(gen_context)
                    instance["hard_gen_target"].append(gen_target)
                else:
                    print('failed')
        except:
            print('!!!!!!!!!!!!!!!!!!!!!!!!')
            continue
    print(f"after adding and filtering: ", len(instance["hard_gen_input"]), '\n\n')
    # gen_context = extract(split_ans_cleaned[2], is_probe=True)
    # gen_target = extract(split_ans_cleaned[3], is_probe=False)

    # print(gen_context, '|||', gen_target)
    # instance["context"].append(gen_context)
    # instance["target"].append(gen_target)



    # except:
    #     print('!!!!!!!!!!!!!!!!!!!!!!!!')
    #     # raise NotImplementedError

    new_dataset.append(instance)
    
with open("fictional_knowledge_hard_filtered_added.json", "w") as f:
    json.dump(new_dataset, f, indent=4)