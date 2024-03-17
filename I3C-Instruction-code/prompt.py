import os
import re
import random
import math
import time

import tiktoken

from utils import get_response
enc = tiktoken.get_encoding("cl100k_base")
enc = tiktoken.encoding_for_model("gpt-4")


sleep_time = 0.1
SHOW = True



def generate_instruction(process_record, original_question, condition_sentence, problem_sentence, index, model, max_length, proxies):
    instruction_prompt = f"Q: {original_question} Is condition \"{condition_sentence}\" relevant to the process of solving problem \"{problem_sentence}\"\nA: Let's think step by step."
    start_time = time.process_time()
    start = time.time()
    instruction = get_response(
        prompt=instruction_prompt,
        model=model,
        max_length=max_length,
        proxies=proxies
    )
    end = time.time()
    end_time = time.process_time()
    time_consumption = end_time - start_time
    time.sleep(sleep_time)
    process_record['original_question'] = original_question
    if 'instruction' not in process_record:
        process_record['instruction'] = {}
    process_record['instruction'][f'{index}'] = instruction
    process_record['instruction'][f'{index}_TimeConsumption'] = end-start
    process_record['instruction'][f'{index}_TokenConsumption'] = len(enc.encode(instruction))
    if SHOW:
        print(f'Instruction {index}: {instruction}')
    return process_record
