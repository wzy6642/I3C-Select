# Instructing Large Language Models to Identify and Ignore Irrelevant Conditions (NAACL 2024)

## Introduction & Setup

This repository contains the code for the paper [Instructing Large Language Models to Identify and Ignore Irrelevant Conditions](https://arxiv.org/abs/2403.12744) (Accepted to NAACL Main 2024). I3C instructs LLMs to identify and ignore irrelevant conditions. I3C-Select selects the most confusing problems and their generated reasoning paths as demonstrations for few-shot learning.

![image](https://github.com/wzy6642/I3C-Select/blob/main/framework.png)

 - Run `I3C-Instruction-code/run.py` to generate I3C instruction

```python
python run.py --data_index 3
```

 - Run `I3C-Select-code/run.py` to generate the answer to the given math word problem

```python
python run.py --data_index 3
```

## Experimental Results

![image](https://github.com/wzy6642/I3C-Select/blob/main/experiments.png)

## Citing I3C
```markdown
@inproceedings{wu2024I3C,
  title={Instructing Large Language Models to Identify and Ignore Irrelevant Conditions},
  author={Wu, Zhenyu and Jiang, Meng and Shen, Chao},
  booktitle={Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)},
  year={2024}
}
```

## License

This project is licensed under the Apache-2.0 License.
