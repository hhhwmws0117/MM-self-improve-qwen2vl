import os
import json

data_dir = '/home/nfs03/liyt/vlm-cot/custom_data/geoQA-data'
img_dir = '/home/nfs03/liyt/vlm-cot/custom_data/geoQA-data/images'

tgt_file = 'train.jsonl'


def query_format(item):

    options = ['A', 'B', 'C', 'D', 'E'] 

    question = item['question']
    choices = item['choices']
    choices_with_option = [f"({op}) {choice}" for op, choice in 
                           zip(options[:len(choices)], choices)]
    
    query_text = "{question}\nChoices:\n{choices}".format(
        question=question,
        choices="\n".join(choices_with_option)
    )
    # answer = choices[ord(item['answer']) - ord('A')]
    answer = item['answer']

    return query_text, answer

if __name__ == '__main__':
    
    abs_data_file = os.path.join(data_dir, tgt_file)
    datas = json.load(open(abs_data_file))
    
    new_data = []
    for item in datas: 
        base_query, ground_truth = query_format(item)
        query = "{base_query}\n{output_format}".format(
            base_query=base_query,
            # task_prompt="Let's think step ,
            # task_prompt='',
            # output_format="Firstly, please give me your reasoning retionales in orderd, then answer the question with option of choices."
            output_format='Answer the question with Chain-of-Thought as short as possible without losing accuracy. Finally answer with option from choices with prefix "Answer: ".'
            # output_format='Answer the question with Chain-of-Thought.'
            # output_format=''
        ).strip()


        abs_image = os.path.join(img_dir, item['image'])

        messages = []        
        messages.append({"role": "user", "content": "<image>" + query})
        # messages.append({"role": "assistant", "content":  ground_truth})
        messages.append({"role": "assistant", "content":  'Answer: \n' + ground_truth})
        new_data.append({
            "messages": messages,
            "images": [abs_image]
        })
    
    with open(os.path.join('/home/nfs03/liyt/Qwen2_VL_run/data', 'geoqa_train_cot.json'), 
              'w') as fw:
        json.dump(new_data, fw, ensure_ascii=False, indent=2)


