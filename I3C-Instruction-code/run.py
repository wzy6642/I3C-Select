import warnings
warnings.filterwarnings('ignore')
import json
import os
import argparse

from tqdm import tqdm
import numpy as np

from utils import data_name_choices
import utils, prompt

proxy = False
proxies = None


max_length = 512


parser = argparse.ArgumentParser(description="Dataset Index")
parser.add_argument('--data_index', type=int, required=True, metavar='', default=0, help="0: 'AddSub', 1: 'MultiArith', 2: 'SVAMP', 3: 'GSM8K', 4: 'SingleEq', 5: 'GSM-IC2', 6: 'GSM-ICM', 7: 'SingleOp'")
args = parser.parse_args()
## load raw data
# 'AddSub', 'MultiArith', 'SVAMP', 'GSM8K', 'SingleEq', 'GSM-IC2', 'GSM-ICM', 'SingleOp'
data_name_idx = args.data_index
data_name = data_name_choices[data_name_idx]
model_name = "gpt-3.5-turbo-1106"  
with open(f'../test_data/{data_name}.json', 'r') as f:
    samples = json.load(f)
original_questions = [sample.get('original_question') for sample in samples]
condition_sentences = [sample.get('condition_sentences') for sample in samples]
question_sentences = [sample.get('question_sentence') for sample in samples]
equations = [sample.get('equation') for sample in samples]
gold_answer = [sample.get('gold_answer') for sample in samples]
gold_ic_indexs = [sample.get('gold_ic_idx') for sample in samples] 
cand_ic_indexs = [sample.get('cand_ic_idx') for sample in samples]
ctoq_similarity = [sample.get('condition_question_similarity') for sample in samples]
save_path = os.path.join('../I3C-Instruction/', f'{data_name.capitalize()}-{model_name.capitalize()}.txt')
print(sum([len(i) for i in cand_ic_indexs]), len(samples), sum([len(i) for i in condition_sentences]))

## generate instruction
if not os.path.exists(save_path):
    add_idx = 0
else:
    add_idx = len(utils.load_txt_data(save_path))
for question_idx in tqdm(range(len(samples)), desc=f'{data_name} {model_name}'):
    question_idx += add_idx
    process_record = {}
    for instruction_idx in range(len(cand_ic_indexs[question_idx])):
        process_record = prompt.generate_instruction(
            process_record, 
            original_questions[question_idx].replace('?', ' ?'),
            condition_sentences[question_idx][cand_ic_indexs[question_idx][instruction_idx]], 
            question_sentences[question_idx].replace('?', ' ?'), 
            instruction_idx, 
            model_name, 
            max_length, 
            proxy,
        )
    with open(save_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(process_record, ensure_ascii=False) + '\n')
