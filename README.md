# Instructing Large Language Models to Identify and Ignore Irrelevant Conditions (NAACL 2024)

### Introduction & Setup

This repository contains the code for the paper [Instructing Large Language Models to Identify and Ignore Irrelevant Conditions]() (Accepted to NAACL Main 2024). I3C instructs LLMs to identify and ignore irrelevant conditions. I3C-Select selects the most confusing problems and their generated reasoning paths as demonstrations for few-shot learning.

![image](https://github.com/wzy6642/PRP/blob/main/img/framework.PNG)

 - Run `I3C-Instruction-code/run.py` to generate I3C instruction

```python
python run.py --data_index 3
```

 - Run `I3C-Select-code/run.py` to generate the answer to the given math word problem

```python
python run.py --data_index 3
```
