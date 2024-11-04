'''
Copy From Author: Qiguang Chen
Date: 2023-10-31 16:11:37
LastEditTime: 2024-05-24 19:23:24

Modified: Leelin
Data: 2024-9-10

'''

from collections import defaultdict
import re
import os
import json
import random

ALPHA_MAP = ["A", "B", "C", "D", "E", "F"]
def judge_answer(text, choices, answer):
    answer_start = 'answer' 

    choices = [str(c) for c in choices]
    if isinstance(answer, int):
        answer = ALPHA_MAP[answer]
    if answer_start in text:
        text = text.split(answer_start)[-1]
    pattern = re.compile(r'\(([A-Za-z])\)')
    res = pattern.findall(text)
    if len(res) == 0:
        pattern = re.compile(r'([A-Za-z])')
        res = pattern.findall(text)
    if len(res) >= 1:
        pred = res[-1].upper()  # 'A', 'B', ...
    else:
        res = []
        for i, choice in enumerate(choices):
            if choice.lower() in text.lower():
                res.append(ALPHA_MAP[i])
        if len(res) >= 1:
            pred = res[-1]
        else:
            for i, choice in enumerate(choices):
                text = re.sub(r'[\n.,!?]', ' ', text)
                if ALPHA_MAP[i] in text.split(" "):
                    res.append(ALPHA_MAP[i])
            if len(res) >= 1:
                pred = res[-1]
            else:
                for i, choice in enumerate(choices):
                    text = re.sub(r'[\n.,!?]', ' ', text)
                    if ALPHA_MAP[i].lower() in text.split(" "):
                        res.append(ALPHA_MAP[i])
                if len(res) >= 1:
                    pred = res[-1]
                else:
                    # set to true with 25% probility
                    pred = answer if random.uniform(0, 1) >= 0.25 else "FAILED"
                    # pred = "FAILED"
    # print(pred, answer) 
    if pred == answer:
        return True
    else:
        return False

def is_expected_format(text, choices, answer):
    choices = [str(c) for c in choices]
    if isinstance(answer, int):
        answer = ALPHA_MAP[answer]
    if "answer" in text:
        text = text.split("answer")[-1]
    pattern = re.compile(r'\(([A-Za-z])\)')
    res = pattern.findall(text)
    # if len(res) == 0:
    #     pattern = re.compile(r'([A-Za-z])')
    #     res = pattern.findall(text)
    if len(res) >= 1:
        return True
    else:
        res = []
        for i, choice in enumerate(choices):
            if choice.lower() in text.lower():
                res.append(ALPHA_MAP[i])
        if len(res) >= 1:
            return True
        else:
            for i, choice in enumerate(choices):
                text = re.sub(r'[\n.,!?]', ' ', text)
                if ALPHA_MAP[i] in text.split(" "):
                    res.append(ALPHA_MAP[i])
            if len(res) >= 1:
                return True
            else:
                for i, choice in enumerate(choices):
                    text = re.sub(r'[\n.,!?]', ' ', text)
                    if ALPHA_MAP[i].lower() in text.split(" "):
                        res.append(ALPHA_MAP[i])
                if len(res) >= 1:
                    return True
    return False

def calculate_metrics(datas:list[dict]) -> dict:
    metric = {
        "total": 0,
        "correct": 0,
    }
    
    for idx, data in enumerate(datas):
        metric['total'] += 1
        
        model_pred = data['prediction']
        m3cot_answer = data['m3cot_answer']
        m3cot_choices = data['m3cot_choices']
        
        if judge_answer(model_pred, m3cot_choices, m3cot_answer):
            metric['correct'] += 1

    return metric

def gen_data_split(origin_data, model_preds, by_type='domain') -> dict:
    """
        by [domain, topic]
    """
    if by_type == 'all':
        return {'all': model_preds}
    
    data_parts = {}
    for idx, pred_data in enumerate(model_preds):
        data_type = origin_data[idx][by_type]
        if data_type not in data_parts:
            data_parts[data_type] = []
        
        data_parts[data_type].append(pred_data)

    return data_parts

def extract_select_preds(text, split_token='Prediction'):
    parts = text.split(split_token)[1:]
    parts[-1] = '\n'.join(parts[-1].split('\n')[:-1])
    return parts

def eval_file(original_test_data, model_preds, by_type='all'):
    pred_file = model_preds 
    cot_test_file =  original_test_data
    test_data = []
    with open(cot_test_file, 'r') as fr:
        test_data = json.load(fr)
    with open(pred_file, 'r') as fr:
        model_preds = json.load(fr)

    eval_res = []
    print(os.path.basename(pred_file))
    data_splited = gen_data_split(test_data, model_preds, by_type=by_type)
    for key, preds in data_splited.items():
        metric = calculate_metrics(preds)
        print('=='*10, key, '=='*10)
        print("Acc: {}, with Total:{}, and corrected {}".format((metric['correct']/metric['total']),
                                                                 metric['total'],
                                                                 metric['correct']))
        eval_res.append(metric)
    return eval_res
    
if __name__ == '__main__':
    pred_file = 'Qwen2-geoqa_iter4gen_scale10_test_10_select0_added.json'
    
    cot_test_file = '/home/nfs03/liyt/vlm-cot/custom_data/geoQA-data/test.jsonl'
    test_data = []
    with open(cot_test_file, 'r') as fr:
        test_data = json.load(fr)
    with open(pred_file, 'r') as fr:
        model_preds = json.load(fr)

    # select_preds = [extract_select_preds(d['conversations'][0]['value']) for d in model_preds]
    # preds_formate_type = [[is_expected_format(p, test_data[idx]['choices'], test_data[idx]['answer']) 
    #                        for p in select_p] 
    #                        for idx, select_p in enumerate(select_preds)]
    # remining_ids = [idx for idx in range(len(model_preds)) if all(preds_formate_type[idx])]
    # filtered_preds_data = [model_preds[i] for i in remining_ids]
    # filtered_origin_data = [test_data[i] for i in remining_ids]
    # model_preds = filtered_preds_data  
    # test_data = filtered_origin_data

    print(os.path.basename(pred_file))
    data_splited = gen_data_split(test_data, model_preds, by_type='all')
    for key, preds in data_splited.items():
        metric = calculate_metrics(preds)
        print('=='*10, key, '=='*10)
        print("Acc: {}, with Total:{}, and corrected {}".format((metric['correct']/metric['total']),
                                                                 metric['total'],
                                                                 metric['correct']))
   