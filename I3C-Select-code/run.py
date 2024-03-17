import warnings
warnings.filterwarnings('ignore')
import json
import os
import time
import argparse

from tqdm import tqdm
import numpy as np

from utils import data_name_choices
import utils, prompt

proxy = False
proxies = None


max_length = 1024
num_demonstrations = 8
model_name = "gpt-3.5-turbo-1106"  
prompt_strategy = 'I3C-Select'


parser = argparse.ArgumentParser(description="Dataset Index")
parser.add_argument('--data_index', type=int, required=True, metavar='', default=0, help="0: 'AddSub', 1: 'MultiArith', 2: 'SVAMP', 3: 'GSM8K', 4: 'SingleEq', 5: 'GSM-IC2', 6: 'GSM-ICM', 7: 'SingleOp', 8: 'AQuA', 9: 'MATH'")
args = parser.parse_args()
## load raw data
# 'AddSub', 'MultiArith', 'SVAMP', 'GSM8K', 'SingleEq', 'GSM-IC2', 'GSM-ICM', 'SingleOp'
data_name_idx = args.data_index
data_name = data_name_choices[data_name_idx]
samples = utils.load_json_file(f'../test_data/{data_name}.json')
original_questions = [sample.get('original_question') for sample in samples]
condition_sentences = [sample.get('condition_sentences') for sample in samples]
question_sentences = [sample.get('question_sentence') for sample in samples]
equations = [sample.get('equation') for sample in samples]
gold_answer = [sample.get('gold_answer') for sample in samples]
gold_ic_indexs = [sample.get('gold_ic_idx') for sample in samples]
cand_ic_indexs = [sample.get('cand_ic_idx') for sample in samples]
ctoq_similarity = [sample.get('condition_question_similarity') for sample in samples]
instruction_path = os.path.join('../I3C-Instruction/', f'{data_name.capitalize()}-Gpt-3.5-turbo-1106.txt')
instructions = utils.load_txt_data(instruction_path)
if data_name_idx in [0, 3, 4]:
    demonstrations = utils.load_json_file(os.path.join('../demonstrations/', f'GSM8K.json'))
    demonstrations_questions = [demo.get('question') for demo in demonstrations]
    demonstrations_reasoning = [demo.get('reasoning') for demo in demonstrations]
else:
    demonstrations = utils.load_json_file(os.path.join('../demonstrations/', f'{data_name}.json'))
    demonstrations_questions = [demo.get('question') for demo in demonstrations]
    demonstrations_reasoning = [demo.get('reasoning') for demo in demonstrations]
save_path = os.path.join('../result/', f'{data_name.capitalize()}-{prompt_strategy.capitalize()}-demonstrations({num_demonstrations}).txt')

## generate answer
if not os.path.exists(save_path):
    add_idx = 0
else:
    add_idx = len(utils.load_txt_data(save_path))
for question_idx in tqdm(range(len(samples)), desc=f'{data_name} {prompt_strategy} {model_name}'):
    question_idx += add_idx
    process_record = {}
    original_question = original_questions[question_idx]
    instruction = instructions[question_idx].get('instruction')
    if type(instruction)==dict:
        instruction_str = []
        for key, value in instruction.items():
            if 'TimeConsumption' in key or 'TokenConsumption' in key:
                pass
            else:
                instruction_str.append(value)
        instruction_list = '\n'.join(instruction_str)
    else:
        instruction_list = 'None'

    process_record = prompt.generate_answer(
        process_record, 
        original_question,
        instruction_list, 
        demonstrations_questions, 
        demonstrations_reasoning,
        num_demonstrations, 
        model_name, 
        max_length, 
        proxy
    )
    process_record['gold_answer'] = gold_answer[question_idx]
    process_record['instructions'] = instruction_list
    process_record['equation'] = equations[question_idx]
    process_record['condition_question_similarity'] = ctoq_similarity[question_idx]

    with open(save_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(process_record, ensure_ascii=False) + '\n')