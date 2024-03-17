import os
import re
import random
import math
import time

import numpy as np
import pandas as pd
import tiktoken

from utils import get_response
enc = tiktoken.get_encoding("cl100k_base")
enc = tiktoken.encoding_for_model("gpt-4")


sleep_time = 0.1
SHOW = True


def post_process_value(generate_answer, location=-1):
    generate_answer = generate_answer.replace(',', '')                                                  
    generate_answer = ''.join(char for char in generate_answer if not char.isalpha())                   
    generate_answer = ''.join(char for char in generate_answer if char not in ['(', ')'])               
    generate_answer = generate_answer.strip()                                                           
    if type(generate_answer) == str and len(generate_answer) >= 1 and generate_answer[-1] == '.':       
        generate_answer = generate_answer[:-1]
    generate_answer = generate_answer.strip()
    if ' ' in generate_answer:                                                                         
        generate_answer = generate_answer.split(' ')[location]
    if type(generate_answer) == str and len(generate_answer) >= 1:                                     
        pass
    else:
        generate_answer = 0
    if generate_answer in ['-', '=', '+']:                                                              
        generate_answer = 0
    if type(generate_answer) == str and '%' in generate_answer:                                          
        generate_answer = float(generate_answer.rstrip('%')) / 100
    if type(generate_answer) == str and ':' in generate_answer:                                          
        generate_answer = generate_answer.replace(':', '.')
    if type(generate_answer) == str and len(generate_answer) >= 1 and generate_answer[-1] in ['.', '/']: 
        generate_answer = generate_answer[:-1]
    if type(generate_answer) == str:
        generate_answer = generate_answer.replace('</>', '') 
        generate_answer = generate_answer.replace('$', '') 
        generate_answer = generate_answer.replace('<>', '').replace('=', '') 
        if len(generate_answer)==0 or generate_answer=='.':
            generate_answer = '0'
        if generate_answer[-1]=='.':
            generate_answer = generate_answer[:-1]
        if len(generate_answer)>=2 and generate_answer[0]=='0':
            generate_answer = generate_answer[1:]

        generate_answer = eval(generate_answer)
    return generate_answer


def get_arabic_number(problem, reasoning_path, model, max_length, proxies):
    prompt = f"""
                Q: {problem} 
                A: {reasoning_path} 
                Therefore, the answer (expressed in Arabic numerals and without units) is:
              """
    value = get_response(
        prompt=prompt,
        model=model,
        max_length=max_length, 
        proxies=proxies
    )
    time.sleep(sleep_time)
    value = post_process_value(value)
    return value


def generate_answer(process_record, original_question, instructions, demonstrations_question, demonstrations_answer, num_demonstrations, model, max_length, proxies):
    demonstrations = '\n'.join([f'Q: {demonstrations_question[idx]}\nA: {demonstrations_answer[idx]}' for idx in range(num_demonstrations)])
    if instructions!='None':
        reasoning_prompt = f"""
                                The instructions are as follows: {instructions}\n
                                Let's consider these instructions and ignore the irrelevant conditions to solve the problem.\n
                                {demonstrations}\n
                                Q: {original_question}\n
                                A: Let's think step by step.
                            """
    else:
        reasoning_prompt = f"""
                                Feel free to ignore irrelevant information given in the questions.\n
                                {demonstrations}\n
                                Q: {original_question}\n
                                A: Let's think step by step.
                            """
    start = time.time()
    reasoning_path = get_response(
        prompt=reasoning_prompt,
        model=model,
        max_length=max_length,
        proxies=proxies
    )
    end = time.time()
    time.sleep(sleep_time)
    process_record['original_question'] = original_question
    process_record['reasoning_path'] = reasoning_path
    process_record['TimeConsumption'] = end-start
    process_record['TokenConsumption'] = len(enc.encode(reasoning_path))
    numerical_answer = get_arabic_number(original_question, reasoning_path, model, max_length, proxies)
    time.sleep(sleep_time)
    process_record['final_answer'] = numerical_answer
    if SHOW:
        print(f"Reasoning Path: {reasoning_path}")
        print(f"Final Answer: {numerical_answer}")
    return process_record
