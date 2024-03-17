# -*- coding: utf-8 -*-
import json
import re
import math
import sys

import openai

openai.api_key = "sk-****"  
data_name_choices = ['AddSub', 'MultiArith', 'SVAMP', 'GSM8K', 'SingleEq', 'GSM-IC2', 'GSM-ICM', 'SingleOp', 'AQuA', 'MATH']
pattern = r'-?\d+(?:\.\d+)?'


def load_txt_data(path):
    with open(path, 'r', encoding='gb18030', errors='ignore') as f:
        data = f.readlines()
    data = [eval(sub_data) for sub_data in data]
    return data

def save_json_file(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file)


def check_string(s):
    if s == "" or s == "IP访问频率过高,请稍后再试":
        raise ValueError("Empty string encountered.")
    

# get_response_api_free
def get_response(prompt, model, max_length, proxies):
    completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": f"{prompt}"}], max_tokens=max_length, temperature=0.7)
    response = completion.choices[0].message.content
    try:
        check_string(response)
    except Exception as e:
        print(e) 
        sys.exit(1)
    return response
