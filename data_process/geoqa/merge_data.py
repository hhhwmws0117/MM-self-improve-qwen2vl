import os
import json
import random
import re
import collections
import argparse

llava15_inst = '\nAnswer the question using a single word or phrase.'
llava_qa_inst = '\nAnswer with the option’s letter from the given choices directly.'

ALPHA_MAP = ["A", "B", "C", "D", "E", "F"]
def judge_answer(text, choices, answer):
    choices = [str(c) for c in choices]
    if isinstance(answer, int):
        answer = ALPHA_MAP[answer]
    if "Answer" in text:
        text = text.split("Answer")[-1]
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
                    # pred = answer if random.uniform(0, 1) >= 0.25 else "FAILED"
                    pred = "FAILED"
    
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

def gen_question(question: str, choices:list, cot=True) -> str:
    
    option_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    options = [f"({c}) {choice}" for c, choice in zip(option_list[:len(choices)], choices)]
    if cot:
        # task_prompt = "Firstly, generate a step-by-step reasoning process to justify your answer. Then, answer the geometry calculation question with the BEST option from given choices."
        task_prompt = "Let's think step by step!"
    else:
        task_prompt = "Answer the geometry calculation question with the BEST option from given choices."

    # user_prompt = "{question}\nChoices:\n{choices}\n"
    query = "{base_query}\n{task_prompt}\n{output_format}".format(
            base_query=question,
            # task_prompt="Let's think step by step.",
            task_prompt='Answer the question with Chain-of-Thought.',
            # output_format="Output reasoning step in order 1. 2. 3., and output choice option as answer at the end with 'Answer:\\n' ahead."
            output_format=''
    ).strip()
    question = query
    
    return question 


def judge_data(data: list[dict], origin_data: list[dict]) -> list[bool]:
    corrects = []
    for idx, d in enumerate(data):
        pred = d['prediction']
        # m3cot_answer = d['m3cot_answer'] 
        # m3cot_choices = d['m3cot_choices']
        m3cot_answer = origin_data[idx]['answer']
        m3cot_choices = origin_data[idx]['choices']
       
        correct = judge_answer(pred, m3cot_choices, m3cot_answer)
        corrects.append(correct)
    
    return corrects

def extract_answer(text, choices, answer):
    ALPHA_MAP = ["A", "B", "C", "D", "E", "F"]
    choices = [str(c) for c in choices]
    if isinstance(answer, int):
        answer = ALPHA_MAP[answer]
    if "Answer:" in text:
        text = text.split("Answer:")[-1]
    pattern = re.compile(r'\(([A-Za-z])\)')
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
                    pred = "FAILED"
    return pred

def check_data_id_match(data: list[dict]) -> bool:
    print([len(d) for d in data])
    total_length = len(data[0])
    
    for i in range(total_length):
        cur_ids = [d[i]['index'] for d in data]

        tmp_correct = cur_ids[0]
        for cur_correct in cur_ids:
            if cur_correct != tmp_correct:
                print(f'not match in {i}, got {cur_ids}')
                return False
    
    return True

def find_pos_neg_id(corrects: list) -> tuple:
    """
    return: posids, negids
    """
    pos_ids = []
    neg_ids = []
    for idx, cor in enumerate(corrects):
        if cor:
            pos_ids.append(idx)
        else:
            neg_ids.append(idx)
    return pos_ids, neg_ids

def gen_refine_data(merge_datas: list, merge_correct: list, origin_data, order_list) -> list[dict]:

    total_length = len(merge_datas[0])

    refine_data = []
    for i in range(total_length):
        cur_correct = [d_cor[i] for d_cor in merge_correct]
        pos_ids, neg_ids = find_pos_neg_id(cur_correct)
        if len(pos_ids) == 0 or len(neg_ids) == 0:
            # model do the same judge, no data to refine
            continue
        
        # generate refine data
        # pos_pred = merge_datas[random.choice(pos_ids)][i]
        # neg_pred = merge_datas[random.choice(neg_ids)][i]
        find_pos, find_neg = False, False
        for file_order in order_list:
            # choice the newest data
            if len(merge_datas[file_order][i]['prediction']) <= 50:
                continue
            if merge_datas[file_order][i]['prediction'].startswith('Answer'):
                continue
            if not find_pos and file_order in pos_ids:
                pos_pred = merge_datas[file_order][i]
                find_pos = True
            if not find_neg and file_order in neg_ids:
                neg_pred = merge_datas[file_order][i]
                find_neg = True
        if not find_pos or not find_neg:
            continue
        
        task_prompt = "\nJudge the correctness of the model's prediction and refine it."

        origin_question = origin_data[i]['question']
        origin_id = origin_data[i]['id']
        origin_img_pth = os.path.basename(origin_data[i]['image'].replace('\\', '/')) 
        origin_answer = origin_data[i]['answer']
        origin_choice = origin_data[i]['choices']

        # m3cot_question = "[Question] {}\n [Choices] {}\n".format(origin_question, origin_choice)
        problem = gen_question(origin_question, origin_choice)        
        refine_question = problem + "\nModel's Predicion: " + neg_pred['prediction'] + task_prompt
        refine_answer = pos_pred['prediction']
        
        conversations = []
        conversations.append({"from": "user", "value": refine_question})
        conversations.append({"from": "assistant", "value": refine_answer})
        refine_item = {
            "id": 'refine-' + origin_data[i]['id'],
            "image": origin_img_pth,
            "conversations": conversations,
            # "m3cot_answer": origin_answer,
            # "m3cot_choices": origin_choice,
        }

        refine_data.append(refine_item)

    return refine_data

import random
import os

def gen_select_data_n(merge_datas: list, merge_correct: list, origin_data: list, order_list, data_type, n_candidates=[3], max_trial=10) -> tuple:
    total_length = len(merge_datas[0])
    
    select_data = []
    metric = {'2pos1neg': 0, '1pos2neg': 0, 'total': 0}
    
    for n in n_candidates:
        for i in range(total_length):
            cur_correct = [d_cor[i] for d_cor in merge_correct]
            pos_ids, neg_ids = find_pos_neg_id(cur_correct)
            
            if data_type == 'train' and (len(pos_ids) == 0 or len(neg_ids) == 0):
                # If there are no correct or incorrect predictions, skip this iteration
                continue
            
            ordered_pos_ids = [file_order for file_order in order_list if file_order in pos_ids]
            ordered_neg_ids = [file_order for file_order in order_list if file_order in neg_ids]

            # filterd or non cot
            ordered_pos_ids = [file_order for file_order in ordered_pos_ids 
                               if len(merge_datas[file_order][i]['prediction']) > 50]
            ordered_neg_ids = [file_order for file_order in ordered_neg_ids
                               if len(merge_datas[file_order][i]['prediction']) > 50]
            ordered_pos_ids = [file_order for file_order in ordered_pos_ids 
                               if not merge_datas[file_order][i]['prediction'].startswith('Answer')]
            ordered_neg_ids = [file_order for file_order in ordered_neg_ids
                               if not merge_datas[file_order][i]['prediction'].startswith('Answer')]
            
            if len(ordered_neg_ids) == 0 or len(ordered_pos_ids) == 0:
                continue

            samples_generated = 0
            max_sample_per_item = 2
            for trail_time in range(max_trial):
                if samples_generated > max_sample_per_item:
                    break
                if len(ordered_pos_ids) + len(ordered_neg_ids) < n:
                    break  # Not enough data to choose n candidates

                # Ensure at least one positive in the candidates
                num_pos_samples = random.randint(1, min(len(ordered_pos_ids), n - 1))
                num_neg_samples = n - num_pos_samples
                if len(ordered_neg_ids) < num_neg_samples:
                    continue  # Not enough negative samples

                # Randomly select positive and negative samples
                selected_pos_indexes = random.sample(ordered_pos_ids, num_pos_samples)
                selected_neg_indexes = random.sample(ordered_neg_ids, num_neg_samples)
                selected_indexes = selected_pos_indexes + selected_neg_indexes
                random.shuffle(selected_indexes)

                # Extract predictions and verify consistent positive answers
                solutions_cand = [merge_datas[j][i]["prediction"] for j in selected_indexes]

                # Build user prompt and assistant response
                task_prompt = "\nWhich prediction is correct? Give the final answer for the beginning question by selecting the correct option."
                origin_question = origin_data[i]['question']
                problem = gen_question(origin_question, origin_data[i]['choices'])
                
                user_prompt = f"{problem}\n" + "\n".join(
                    [f"Model's Prediction {idx + 1}:\n{solution}" for idx, solution in enumerate(solutions_cand)]
                ) + f"\n{task_prompt}"
                origin_answer = origin_data[i]['answer']
                pos_answer = f"Answer:\n {origin_answer}"

                conversations = [
                    {"from": "user", "value": user_prompt},
                    {"from": "assistant", "value": pos_answer}
                ]
                
                select_item = {
                    "id": f'select-{origin_data[i]["id"]}-{n}-{samples_generated}',
                    "image": os.path.basename(origin_data[i]['image'].replace('\\', '/')),
                    "conversations": conversations
                }

                select_data.append(select_item)
                samples_generated += 1
                metric['total'] += 1


    return select_data, metric


def gen_select_data(merge_datas:list, merge_correct:list, origin_data:list, order_list, data_type) -> tuple:
    
    total_length = len(merge_datas[0])
    
    select_data = []
    metric = {'2pos1neg': 0, '1pos2neg': 0, 'total': 0}
    for i in range(total_length):
        cur_correct = [d_cor[i] for d_cor in merge_correct]
        pos_ids, neg_ids = find_pos_neg_id(cur_correct)
        if data_type == 'train':
            if len(pos_ids) == 0 or len(neg_ids) == 0:
            # model do the same judge, no data to refine
                continue
        
        ordered_pos_ids = [file_order for file_order in order_list if file_order in pos_ids]
        ordered_neg_ids = [file_order for file_order in order_list if file_order in neg_ids]
        # print(len(ordered_neg_ids), len(ordered_pos_ids))

        # # filterd or non cot
        if data_type == 'train':
            ordered_pos_ids = [file_order for file_order in ordered_pos_ids 
                               if len(merge_datas[file_order][i]['prediction']) > 50]
            ordered_neg_ids = [file_order for file_order in ordered_neg_ids
                               if len(merge_datas[file_order][i]['prediction']) > 50]
            ordered_pos_ids = [file_order for file_order in ordered_pos_ids 
                               if not merge_datas[file_order][i]['prediction'].startswith('Answer')]
            ordered_neg_ids = [file_order for file_order in ordered_neg_ids
                               if not merge_datas[file_order][i]['prediction'].startswith('Answer')]
            
            if len(ordered_neg_ids) == 0 or len(ordered_pos_ids) == 0:
                continue
        
        def gen_s_item():
            select_type = '2pos1neg' if len(pos_ids) > 1 else '1pos2neg'
            metric[select_type] += 1
            metric['total'] += 1
            
            ids_list = pos_ids.copy()
            ids_list.extend(neg_ids)
            random.shuffle(ids_list)
            pos_pred_orders = [i for i, idx in enumerate(ids_list) if idx in pos_ids]
            
            task_prompt = "\nWhich prediction is correct? Give the final answer for the beginning question by selecting the correct option."
            
            origin_question = origin_data[i]['question']
            origin_id = origin_data[i]['id']
            origin_img_pth = os.path.basename(origin_data[i]['image'].replace('\\', '/')) 
            origin_answer = origin_data[i]['answer']
            origin_choice = origin_data[i]['choices']

            # m3cot_question = "[Question] {}\n [Choices] {}\n".format(origin_question, origin_choice)
            problem = gen_question(origin_question, origin_choice)        
            model_preds = [merge_datas[idx][i]['prediction'] for idx in ids_list]
            # select_problem = "{question}\n" + [Model's Prediction 1: {}\n Model's Prediction 2: {}\n Model's Prediction 3: {}\n {task_prompt}"
            select_problem = f"{problem}\n" + '\n'.join([
                f"Model's Prediction {idx+1}: {pred}" for idx, pred in enumerate(model_preds)
            ]) + f'{task_prompt}'

            # select_problem = select_problem.format(
            #     *model_preds,
            #     question=problem,
            #     task_prompt=task_prompt
            # )
            # pos_ans_idx = random.choice(pos_pred_orders)
            option_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
            options = [f"({c}) {choice}" for c, choice in zip(option_list[:len(origin_choice)], origin_choice)]
            # select_answer = f"Answer:\n {options[ord(origin_answer) - ord('A')]}"
            select_answer = f"Answer:\n {origin_answer}"
           
            
            conversations = []
            conversations.append({"from": "user", "value": select_problem})
            conversations.append({"from": "assistant", "value": select_answer})
            select_item = {
                "id": f'select-' + origin_data[i]['id'],
                "image": origin_img_pth,
                "conversations": conversations,
                # "m3cot_answer": origin_answer,
                # "m3cot_choices": origin_choice,
            }

            select_data.append(select_item)

        if data_type == 'test':
            # print('Test gen')
            pos_ids = ordered_pos_ids
            neg_ids = ordered_neg_ids
            gen_s_item()
            continue
        
        if len(ordered_pos_ids) >= 2:
            # generate 2pos 1 neg
            pos_ids = ordered_pos_ids[:2]
            neg_ids = ordered_neg_ids[:1]
            
            gen_s_item()
        


        if len(ordered_neg_ids) >= 2:
            # generate 1pos 2 neg
            pos_ids = ordered_pos_ids[:1]
            neg_ids = ordered_neg_ids[:2]

            gen_s_item()

    return select_data, metric

def calculate_Mvote(merge_data: list, origin_data:list) -> dict:
    total_length = len(merge_data[0]) 
    
    correct_num = 0
    for i in range(total_length):
        origin_ans = origin_data[i]['answer']
        origin_choices = origin_data[i]['choices']
        cur_preds = [extract_answer(item[i]['prediction'], origin_choices, origin_ans) for item in merge_data]
        
        counts = collections.Counter(cur_preds)
        counts = sorted(counts.items(), key=lambda x: x[1])
        majority_pred, majority_count = counts[-1] 
        
        if judge_answer(majority_pred, origin_choices, origin_ans):
            correct_num += 1

    metric = {
        "acc": correct_num/total_length,
        "corrects": correct_num, 
        "total_num": total_length
    } 
    return metric

def gen_cot_data(merge_data: list, merge_corrects: list, origin_data: list, order_list: list):

    assert len(order_list) == len(merge_datas)
    total_length = len(merge_datas[0])
    
    cot_data = []
    for i in range(total_length):
        for idx in order_list:
            if not merge_corrects[idx][i] or len(merge_datas[idx][i]['prediction']) <= 50:
                continue
            if merge_datas[idx][i]['prediction'].startswith('Answer'):
                continue
            # if len(merge_datas[idx][i]['prediction']) <= 50:
            #     continue
            
            # gen cot data
            origin_question = origin_data[i]['question']
            origin_id = origin_data[i]['id']
            origin_img_pth = os.path.basename(origin_data[i]['image'].replace('\\', '/')) 
            origin_answer = origin_data[i]['answer']
            origin_choice = origin_data[i]['choices']
            
            
            problem = gen_question(origin_question, origin_choice)        
            cot_answer = merge_data[idx][i]['prediction']
            
            conversations = []
            conversations.append({"from": 'user', "value": problem})
            conversations.append({"from": 'assistant', "value": cot_answer})
            cot_item = {
                "id": 'cot-' + origin_id,
                "image": origin_img_pth,
                "conversations": conversations,
                # "m3cot_answer": origin_answer,
                # "m3cot_choices": origin_choice,
            }
            cot_data.append(cot_item)

            break

    return cot_data


if __name__ == '__main__':

    """
    base cmd
    --d_type=test --model_name=qwenvl --iter_num=
    """
    parser = argparse.ArgumentParser(description='merge data for new loop')
    parser.add_argument('--d_type', type=str, help="[train, test] for different mode")
    parser.add_argument('--iter_num', type=int, help='current iter number')
    parser.add_argument('--model_name', type=str, help='eval saving dir')
    parser.add_argument('--sample_prefix', type=str, help='experiment name')
    parser.add_argument('--save_dir', type=str, help='new data saving dir')
    parser.add_argument('--original_file', type=str, help='original data file path')

    args = parser.parse_args()
    
    data_type = args.d_type
    cur_iter = args.iter_num
    
    eval_pth = args.save_dir
    expr_name = args.sample_prefix
    origin_file = args.original_file

    # origin_file = f'/home/nfs03/liyt/vlm-cot/custom_data/geoQA-data/{data_type}.jsonl'
    merge_files = []
    if data_type == 'train':
        merge_files = [
            f'outputs/{expr_name}-iter{iter_num}_train_sample_{i}_added.json'
            for i in range(3) for iter_num in range(cur_iter+1)
        ]
    elif data_type == 'test':
        merge_files = [
            f'outputs/{expr_name}-iter{cur_iter}_test_sample_{i}_added.json'
            for i in range(3)
        ]
    else:
        raise ValueError(f'{data_type} if not in [train, test]')
    
    # merge_files = [
    #     # Sampled training samples or test samples path
    # ]

    merge_datas = [json.load(open(f, 'r')) for f in merge_files]
    if not check_data_id_match(merge_datas):
        raise ValueError('The id not match per file')
    origin_data = []
    with open(origin_file, 'r') as fr:
        origin_data = json.load(fr)
        

    # judge correctness for each file
    files_corrects = [judge_data(data, origin_data) for data in merge_datas]
    # file_format_type = [[is_expected_format(d['prediction'], origin_data[idx]['choices'], origin_data[idx]['answer']) 
    #                      for idx, d in enumerate(data)] 
    #                     for data in merge_datas]
    # remining_ids = [i for i in range(len(merge_datas[0])) if all([t[i] for t in file_format_type])]
    # filtered_merge_datas = [[data[i] for i in remining_ids] for data in merge_datas]
    # filtered_origin_data = [origin_data[i] for i in remining_ids]
    # merge_datas = filtered_merge_datas
    # origin_data = filtered_origin_data

    # test@M vote
    m_vote_metric = calculate_Mvote(merge_datas, origin_data)
    print("="*10, "M vote", "="*10)
    print(m_vote_metric)
    # exit()

    files_time = [(i, os.stat(merge_files[i]).st_mtime) for i in range(len(merge_files))]
    files_time = sorted(files_time, key=lambda x: x[1], reverse=True)
    file_order = [item[0] for item in files_time]
    print(file_order)
    
    cot_data, refine_data, select_data = [], [], []
    # build cot data
    cot_data = gen_cot_data(merge_data=merge_datas, merge_corrects=files_corrects, origin_data=origin_data, order_list=file_order)
    print('='*20)
    print("Generate cot data: ",  len(cot_data))
    print(cot_data[0])
    
    # build refine data
    refine_data = gen_refine_data(merge_datas=merge_datas, merge_correct=files_corrects, origin_data=origin_data, order_list=file_order)
    print('='*20)
    print('Generated New refine data: {}'.format(len(refine_data)))
    print(refine_data[0])
    
    # build select data
    select_data, select_metric = gen_select_data(merge_datas=merge_datas, merge_correct=files_corrects, origin_data=origin_data, order_list=file_order, data_type=data_type)
    print('='*20)
    print('Generated Select data:')
    print(select_metric)
    print(select_data[0])

    # ! For build mixed self-select training samples
    # # build n select data
    # select_data, select_metric = gen_select_data_n(merge_datas=merge_datas, merge_correct=files_corrects, origin_data=origin_data, order_list=file_order, data_type=data_type,
    #                                                 n_candidates=[2,3,4,5,6])
    # print('='*20)
    # print('Generated Select data:')
    # print(select_metric)
    # print(select_data[0])
    
    output_dir = f'data/{eval_pth}/self_{data_type}_data'
    iter_num = f'iter{cur_iter}'
    cot_out_file = f"{expr_name}-{iter_num}_cot.json"
    refine_out_file = f'{expr_name}-{iter_num}_refine.json'
    select_out_file = f'{expr_name}-{iter_num}_select.json'

    with open(os.path.join(output_dir, cot_out_file), 'w') as fw:
        json.dump(cot_data, fw, indent=4, ensure_ascii=False)
    with open(os.path.join(output_dir, refine_out_file), 'w') as fw:
        json.dump(refine_data, fw, indent=4, ensure_ascii=False)
    with open(os.path.join(output_dir, select_out_file), 'w') as fw:
        json.dump(select_data, fw, indent=4, ensure_ascii=False)