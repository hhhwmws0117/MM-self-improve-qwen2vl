import os
import json

if __name__ == '__main__':
    tgt_files = [
        # 'Qwen2-geoqa_iter0_cot_test0_added.json',
        # 'Qwen2-geoqa_iter1_test0_added.json',
        # 'Qwen2-geoqa_iter2_cot_test0_added.json',
        # 'Qwen2-geoqa_iter3_cot_test0_added.json',
        # 'Qwen2-geoqa_iter3_cot_test_new0_added.json',

        # 'Qwen2-geoqa_iter0_train1_sample0_added.json',
        # 'Qwen2-geoqa_iter1_train_sample0_added.json',
        # 'Qwen2-geoqa_iter2_train_sample0_added.json',
        # 'Qwen2-geoqa_iter3_train_sample0_added.json',
        
        # 'data/geoqa/geoQA_train_iter0.json',
        # 'data/geoqa/geoQA_train_iter1.json',
        # 'data/geoqa/geoQA_train_iter2.json',
        
        'data/geoqa/self_train_data/qwen2-geoqa_iter0_cot.json',
        'data/geoqa/self_train_data/qwen2-geoqa_iter1_cot.json',
        'data/geoqa/self_train_data/qwen2-geoqa_iter2_cot.json',
        'data/geoqa/self_train_data/qwen2-geoqa_iter3_cot.json',

        'data/geoqa/self_train_data/qwen2-geoqa_iter0_select.json',
        'data/geoqa/self_train_data/qwen2-geoqa_iter1_select.json',
        'data/geoqa/self_train_data/qwen2-geoqa_iter2_select.json',
        'data/geoqa/self_train_data/qwen2-geoqa_iter3_select.json',

        'data/geoqa/self_train_data/qwen2-geoqa_iter0_refine.json',
        'data/geoqa/self_train_data/qwen2-geoqa_iter1_refine.json',
        'data/geoqa/self_train_data/qwen2-geoqa_iter2_refine.json',
        'data/geoqa/self_train_data/qwen2-geoqa_iter3_refine.json',
    ]
    
    predictions = [json.load(open(f)) for f in tgt_files]

    all_preds_len = []
    for idx, preds in enumerate(predictions):
        # preds_length = [len(d['prediction']) for d in preds]
        # preds_length = [sum([len(item['content']) for item in d['messages']]) for d in preds]
        preds_length = [sum([len(item['value']) for item in d['conversations']]) for d in preds]
        print('='*20)
        print(tgt_files[idx])
        print('min', min(preds_length))
        print('max', max(preds_length))
        print('mid', sorted(preds_length)[len(preds_length)//2])
        print('avg', sum(preds_length)/len(preds_length))

    
    print(*list(zip(tgt_files, all_preds_len)), sep='\n====\n')

