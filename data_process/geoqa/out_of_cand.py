import os
import json
import re

ALPHA_MAP = ["A", "B", "C", "D", "E", "F"]
def match_answer(text, choices, answer):
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
                    # pred = answer if random.uniform(0, 1) >= 0.25 else "FAILED"
                    pred = "FAILED"
    return pred

if __name__ == '__main__':
    select_file = 'Qwen2-geoqa_iter3_test_select0_added.json'

    select_data = json.load(open(select_file))

    out_of_cand = 0
    clean_crr = 0
    clean_total = 0
    for data in select_data:
        model_pred = data['prediction']
        query_text = data['conversations'][0]['value']
        query_parts = query_text.split("Model's Prediction")
        preds = query_parts[1:]
        preds[-1] = '\n'.join(preds[-1].split('\n')[:-1])

        context_cands = [match_answer(p, data['m3cot_choices'], data['m3cot_answer']) for p in preds]
        model_pred = match_answer(model_pred, data['m3cot_choices'], data['m3cot_answer'])

        if model_pred not in context_cands:
            out_of_cand += 1
        else:
            clean_total += 1
            clean_crr += 1 if model_pred == data['m3cot_answer'] else 0
    
    print(select_file)
    print(f"Out of candidats Num: {out_of_cand} ({out_of_cand/len(select_data)})")
    print(f"Clean acc: {clean_crr/clean_total}")


