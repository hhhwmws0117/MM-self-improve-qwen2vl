import os
import json


if __name__ == "__main__":
    cot_test_file = '/home/nfs03/liyt/vlm-cot/custom_data/geoQA-data/test.jsonl'
    # cot_test_file = '/home/nfs03/liyt/vlm-cot/custom_data/geoQA-data/train.jsonl'
    adding_file = 'Qwen2-geoqa_iter4gen_scale10_test_10_select0.json'
    new_file_name = os.path.basename(adding_file).replace('.json', '_added.json')
    new_file_pth = adding_file.replace(os.path.basename(adding_file), new_file_name)
    
    cot_test_data = [] 
    cot_test_id2index = {}
    with open(cot_test_file, 'r') as fr:
        cot_test_data = json.load(fr)
        # cot_test_id2index[item['id']] = idx
    cot_test_id2index = {item['id']: idx for idx, item in enumerate(cot_test_data)}

    with open(adding_file, 'r') as fr:
        tgt_datas = json.load(fr)
        
    # print(cot_test_id2index.keys())
        
    
    # generate file with mecot original informations
    new_data = []
    for data in tgt_datas: 
        # raw_id = os.path.basename(data['images'][0]).replace('.png', '')
        raw_id = '-'.join(data['id'].split('-')[1:]) # for select

        d_id = raw_id
        # print(raw_id)
        # d_id = '-'.join(d_id.split('-')[1:]) # for select test file
        cot_test_idx = cot_test_id2index[d_id]
        
        m3cot_ans = cot_test_data[cot_test_idx]['answer']
        m3cot_choices = cot_test_data[cot_test_idx]['choices']
        new_data.append({
            "index": d_id,
            **data,
            'm3cot_answer': m3cot_ans,
            'm3cot_choices': m3cot_choices
        })
        
    with open(new_file_pth, 'w') as fw:
        json.dump(new_data, fw, indent=4)
     
