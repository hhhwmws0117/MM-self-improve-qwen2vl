import os
import json

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

import tqdm


model_path = '/home/nlper_data/liyt/Qwen2-VL-7B-Instruct'
# lora_path = 'LLaMA-Factory/saves/qwen2_vl-7b/lora/sft-iter3-new-qa-clean'
# lora_path = 'LLaMA-Factory/saves/qwen2_vl-7b/lora/sft-base'

# # default: Load the model on the available device(s)
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     model_path, torch_dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
# model.load_adapter(lora_path)


# default processer
# if isinstance(model, PeftModel):
#     print('Is peftModel, load base model processor')
#     model_path = model.peft_config["default"].base_model_name_or_path
processor = AutoProcessor.from_pretrained(model_path)
model.eval()
# processor.infer()


def load_data(path):
    data = json.load(open(path))

    new_data = [] 
    for item in data:
        # text = item['messages'][0]['content'].replace('<image>', '')
        # image_pth = item['images'][0]
        
        text = item['conversations'][0]['value']
        image_pth = os.path.join('/home/nfs03/liyt/vlm-cot/custom_data/geoQA-data/images', 
                                 item['image'])
        
        new_data.append([
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image_pth},
                    {"type": "text", "text": text}
                ] 
            }
        ])
    
    return new_data, data
    

if __name__ == '__main__':

    iter_times = 5
    with_sampling = False
    batch_size=16
    
    # load_data 
    datas, raw_datas = load_data('data/geoqa_train_cot.json')
    # datas, raw_datas = load_data('data/geoqa/self_test_data/qwen2-geoqa-clean-qa_iter3_select.json')
    print(f'Model: {model_path}')
    print(f'Lora: {None if not lora_path else lora_path}')
    print(f"sampleing: {with_sampling}")

    # bathc format
    b_datas = [datas[i:i+batch_size] for i in range(0, len(datas), batch_size)]
    b_raw_datas = [raw_datas[i:i+batch_size] for i in range(0, len(datas), batch_size)]
    
    for iter_idx in range(0, iter_times):
        
        model_predictions = []
        probess_bar = tqdm.tqdm(total=len(b_datas))
        for idx, messages in enumerate(b_datas):

            # prepare inputs 
            texts = [processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            ) for msg in messages]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
            # infer and save
            gen_kwargs = {}
            if with_sampling: 
                gen_kwargs.update({
                    'do_sample': True,
                    'top_k': 50,  # 选择 top_k 采样，可以调整 k 值
                    'top_p': 0.95,  # 选择 nucleus 采样，可以调整 p 值
                    'temperature': 0.9,  # 调整温度以控制生成文本的随机性
                    'max_new_tokens': 200
                })
            else:
                gen_kwargs.update({
                    'do_sample': False,
                    'top_k': 50,  # 选择 top_k 采样，可以调整 k 值
                    'temperature': 1,  # 调整温度以控制生成文本的随机性
                    'max_new_tokens': 200
                })
            generated_ids = model.generate(**inputs, **gen_kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            # print(output_texts)
            
            # save
            predictions = b_raw_datas[idx]
            for pred, output_text in zip(predictions, output_texts):
                pred.update({
                    "prediction": output_text,
                    # "answer": pred['messages'][1]['content'],
                    "answer": pred['conversations'][1]['value'],
                })
                model_predictions.append(pred)

            probess_bar.update(1)

            with open(f'Qwen2-geoqa_iter3_cot_qa_clean_test_select{iter_idx}.json', 'w') as fw:
                json.dump(model_predictions, fw, ensure_ascii=False, indent=2)
    
