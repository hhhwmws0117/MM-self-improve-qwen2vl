import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

import tqdm


model_path = '/home/nlper_data/liyt/Qwen2-VL-7B-Instruct'
lora_path = 'LLaMA-Factory/saves/sft-geoqa-iter4-scale-10-16bs/checkpoint-1215'
# lora_path = 'LLaMA-Factory/saves/qwen2_vl-7b/lora/geoqa-new-sft-iter2'
# lora_path = 'LLaMA-Factory/saves/qwen2_vl-7b/lora/sft-iter1'

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
    # device_map="auto",
)
model.load_adapter(lora_path)


# default processer
# if isinstance(model, PeftModel):
#     print('Is peftModel, load base model processor')
#     model_path = model.peft_config["default"].base_model_name_or_path
processor = AutoProcessor.from_pretrained(model_path)
model.eval()
# processor.infer()

distributed_state = Accelerator()
model.to(distributed_state.device)


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

class DummyDataset(Dataset):  # Make sure to inherit properly

    def __init__(self, data, raw_data):
        self.data = data
        self.raw_data = raw_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.raw_data[idx]
    
def process_qwen2(data):
    messages = [b[0] for b in data]
    answers = [b[1] for b in data]
    
    global processor
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
    return inputs, answers
    

if __name__ == '__main__':

    iter_times = 1
    with_sampling = False
    batch_size = 2

    if distributed_state.is_local_main_process:
        print('Model: ', model_path)
        print('Lora: ', None if not lora_path else lora_path)
        print('adapters ', model.active_adapters())
        print(model.generation_config)
    
    # load_data 
    datas, raw_datas = load_data('data/geoqa/self_test_data/qwen2-iter4-n3gen-10_iter4_select.json')
    ds = DummyDataset(datas, raw_datas)
    # print(len(ds))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=process_qwen2)
    dl = distributed_state.prepare(dl)
    
    for iter_idx in range(0, iter_times):
        if distributed_state.is_local_main_process:
            print(f"infer on iter{iter_idx}")
        model_predictions = []
        probess_bar = tqdm.tqdm(total=len(dl), disable=not distributed_state.is_local_main_process)

        with torch.inference_mode():
            for batch in dl:
                inputs, predictions = batch
                inputs = inputs.to(distributed_state.device)
                
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
                # make sure the outputs in order
                output_texts = distributed_state.gather_for_metrics(output_texts)
                predictions = distributed_state.gather_for_metrics(predictions)
                
                # save
                for pred, output_text in zip(predictions, output_texts):
                    pred.update({
                        "prediction": output_text,
                        # "answer": pred['messages'][1]['content'],
                        "answer": pred['conversations'][1]['value'],
                    })
                    # print(output_text)
                    model_predictions.append(pred)
    
        
                probess_bar.update(1)

                # if distributed_state.is_local_main_process:
                #     local_output_path = f'Qwen2-m3cot_iter0_train_sample{iter_idx}.json'
                #     with open(local_output_path, 'w') as fw:
                #         json.dump(model_predictions, fw, ensure_ascii=False, indent=2)
        local_output_path = f'Qwen2-geoqa_iter4gen_scale10_test_10_select{iter_idx}.json'
        with open(local_output_path, 'w') as fw:
            json.dump(model_predictions, fw, ensure_ascii=False, indent=2)