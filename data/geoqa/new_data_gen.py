import copy
import os
import json
import random
import argparse

def formate_as_sharegpt(items, img_dir):
    new_data = [] 
    for item in items:
        messages = []
        input_text = item['conversations'][0]['value']
        if not input_text.startswith('<image>'):
            input_text = '<image>' + input_text
        messages.append({"role": "user", "content": input_text})
        messages.append({"role": "assistant", "content": item['conversations'][1]['value']})

        # img_dir = '/AIRvePFS/ai4science/users/chengkz/geoQA-data/images' 
        # abs_img_pth = os.path.join('/home/nfs03/liyt/vlm-cot/custom_data/geoQA-data/images', item['image'])
        abs_img_pth = os.path.join(img_dir, item['image'])
        
        new_data.append({
            "messages": messages,
            "images": [abs_img_pth]
        })
    return new_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate new data')
    parser.add_argument('--iter_num', type=str, required=True)
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--geoqa_dir', type=str)
    args = parser.parse_args()
    
    iter_num = args.iter_num

    # if 'llava' in args.model_name:
    #     eval_pth = 'geoQA'
    #     model_prefix = 'llava1_5-ep4'
    # elif 'qwen' in args.model_name:
    #     eval_pth = 'geoQA_qwen'
    #     model_prefix = 'qwen-ep3'
    # else:
    #     raise ValueError(f'{args.model_name} not in llava or qwen')
    eval_pth = 'data/geoqa'
    # model_prefix = 'qwen2-geoqa-iter3-scale
    model_prefix = args.prefix



    cot_file = f'{eval_pth}/self_train_data/{model_prefix}_{iter_num}_cot.json'
    refine_file = f'{eval_pth}/self_train_data/{model_prefix}_{iter_num}_refine.json'
    select_file = f'{eval_pth}/self_train_data/{model_prefix}_{iter_num}_select.json'

    train_qa_file = 'data/geoqa_train_base.json'
    # train_rationales_file = '/home/nfs03/liyt/vlm-cot/custom_data/geoQA-data/geoqa_sft_train_536-rationale_noInstruct_new.json'

    # origin_train_data = '/home/nfs03/liyt/vlm-cot/custom_data/geoQA-data/geoqa_sft_train_cot-rationales_noInstruct.json'with open(train_qa_file, r)
    with open(train_qa_file, 'r') as fr:
        train_qa_data = json.load(fr)
    # with open(train_rationales_file, 'r') as fr:
    #     train_rationales = json.load(fr)
    origin_train_data = train_qa_data
    # origin_train_data.extend(train_rationales)
    new_data = []
    for item in train_qa_data: 
        image = os.path.basename(item['images'][0])
        remote_image = os.path.join(args.geoqa_dir, 'images',image)
        item.update({
            "images": [remote_image]
        })
        new_data.append(item)
    origin_train_data = new_data

    
    origin_keys = [os.path.basename(item['images'][0]).replace('.png', '') for item in origin_train_data]
    origin_key2idx = {key:idx for idx, key in enumerate(origin_keys)}
    
    origin_train_data_copy = copy.deepcopy(origin_train_data)
    
    with open(cot_file, 'r') as fr:
        cot_data = json.load(fr)
    with open(refine_file, 'r') as fr:
        refine_data = json.load(fr)
    with open(select_file, 'r') as fr:
        select_data = json.load(fr)
    
    # convert to shareGPT format
    cot_data = formate_as_sharegpt(cot_data, os.path.join(args.geoqa_dir, 'images'))
    refine_data = formate_as_sharegpt(refine_data, os.path.join(args.geoqa_dir, 'images'))
    select_data = formate_as_sharegpt(select_data, os.path.join(args.geoqa_dir, 'images'))
        
    new_data = []
    # do metric 
    metric = {
        "cot": 0,
        "refine": 0,
        "select": 0,
    }
    # merge the cot_file
    # replace the exists older item
    for cot_item in cot_data:
        # cot_id = cot_item['id']
        # if cot_id in origin_keys:
        #     origin_train_data_copy[origin_key2idx[cot_id]] = cot_item
        # else:
            # origin_train_data_copy.append(cot_item)
        origin_train_data_copy.append(cot_item)
        metric['cot'] += 1
    
    # merge the refine data
    for refine_item in refine_data:
        # refine_id = refine_item['id']
        # if refine_id in origin_keys:
        #     origin_train_data_copy[origin_key2idx[refine_id]] = refine_item
        # else:
        #     origin_train_data_copy.append(refine_item)
        origin_train_data_copy.append(refine_item)     
        metric['refine'] += 1
    
    # merge the select data
    for select_item in select_data:
        # select_id = select_item['id']
        # if select_id in origin_keys:
        #     origin_train_data_copy[origin_key2idx[select_id]] = select_item
        # else:
        #     origin_train_data_copy.append(select_item)
        # new_select_id = select_id.replace('select', f"select_{iter_num}")
        # select_item['id'] = new_select_id
        origin_train_data_copy.append(select_item)
        metric['select'] += 1
        
    metric['origin'] = len(origin_train_data_copy) - sum(metric.values())
    # for new_data in origin_train_data_copy: 
    #     cur_id = new_data['id']
    #     d_type = cur_id.split('-')[0]
    #     if 'cot' in d_type:
    #         metric['cot'] += 1
    #     if 'refine' in d_type:
    #         metric['refine'] += 1
    #     if 'select' in d_type:
    #         metric['select'] += 1
    
    print(f'Gen new train data: {len(origin_train_data_copy)}')
    print(metric)

    random.shuffle(origin_train_data_copy) 

    with open(os.path.join(f'{eval_pth}', f"{model_prefix}_{iter_num}.json"), 'w') as fw: 
        json.dump(origin_train_data_copy, fw, indent=4, ensure_ascii=False)
            