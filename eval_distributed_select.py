import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

import tqdm
import argparse

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
    parser = argparse.ArgumentParser(description="Process some parameters for your project.")
    
    parser.add_argument('--iter_times', type=int, default=1, help='Number of iterations (default: 1)')
    parser.add_argument('--with_sampling', type=bool, default=False, help='Use sampling (default: False)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (default: 2)')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the data test file')
    parser.add_argument('--save_name', type=str, required=True, help='tgt saving file pth')

    parser.add_argument('--lora_path', type=str, default=None, help='Path to the lora file')
    parser.add_argument('--base_model', type=str, default='Qwen2-VL-7B-Instruct', help='Base model name')

    args = parser.parse_args()
    
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # device_map="auto",
    )
    if args.lora_path:
        model.load_adapter(args.lora_path)
    processor = AutoProcessor.from_pretrained(args.base_model)
    model.eval()
    
    distributed_state = Accelerator()
    model.to(distributed_state.device)


    iter_times = args.iter_times
    with_sampling = args.with_sampling
    batch_size = args.batch_size

    if distributed_state.is_local_main_process:
        print('Model: ', args.base_model)
        print('Lora: ', None if not args.lora_path else args.lora_path)
        print('adapters ', model.active_adapters())
        print(model.generation_config)
    
    # load_data 
    # datas, raw_datas = load_data(os.path.join('data', args.data_file))
    datas, raw_datas = load_data(args.data_file)

    ds = DummyDataset(datas, raw_datas)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=process_qwen2)
    dl = distributed_state.prepare(dl)
    
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
                model_predictions.append(pred)
    
    
            probess_bar.update(1)

        local_output_path = os.path.join('outputs', f'{args.save_name}.json')
        with open(local_output_path, 'w') as fw:
            json.dump(model_predictions, fw, ensure_ascii=False, indent=2)