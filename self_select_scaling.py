import os
import json
import subprocess
import re
import logging
import shutil
import argparse
import yaml

from data_process.geoqa.add_test_info import add_info
from data_process.geoqa.calculate import eval_file

const_configs = {
    "model_name": 'qwen2-vl-select-scaling',
    # "ckpt_path": 'saves',
    "base_model_path": 'Qwen2-VL-7B-Instruct', # ABS path to model checkpoint
    "dataset": 'geoqa',
    "data_file_pth": 'data',
    "data_utils_dir": 'data_process',
    "geoqa_data_dir": 'geoQA-data', # ABS path to processed geoQA data
    "CUDA_INFO": 'CUDA_VISIBLE_DEVICES=0,1,2,3',
    "train_batch_size": 2
}

# Configure the logging
logging.basicConfig(
    filename=f'self-train-{const_configs["model_name"]}.log',  # Specify the file to log to
    filemode='a',        # 'a' means append to the file (use 'w' to overwrite)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
)


def update_training_config(config_path: str, dataset_name: str, ckpt_dir: str):
    global const_configs
    # Load the YAML configuration file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update dataset and output_dir in the configuration
    config['dataset'] = dataset_name  # e.g., 'geoqa'
    config['output_dir'] = ckpt_dir   # e.g., 'saves/qwen2_vl-7b/sft-geoqa'
    config['model_name_or_path'] = const_configs['base_model_path'] 
    
    default_bs = 64 
    cuda_num = const_configs['CUDA_INFO'].count(',')+1
    gradient_cumulated_step = int(default_bs / cuda_num / const_configs['train_batch_size'])
    
    config['gradient_accumulation_steps'] = gradient_cumulated_step
    config['per_device_train_batch_size'] = 2
    

    logging.info(f"Train ckpt with 64 global batch_size on {cuda_num} GPUs, per device with BS: {const_configs['train_batch_size']}")


    
    # Optionally, add more fields here if needed
    # config['template'] = 'qwen2_vl'  # Ensure consistency with template if needed
    
    # Save the modified configuration back to the YAML file
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Updated training configuration at {config_path} with dataset {dataset_name} and output directory {ckpt_dir}.")


def do_train(iter_name:str,
             prefix: str,
             ckpt_dir:str,
             llama_factory_pth:str='LLaMA-Factory'):
    # inject the data and corresponding config for llama-factory requirement
    global const_configs
    train_data_name = f"{prefix}_iter{int(iter_name[-1])-1}"
    tgt_train_file = os.path.join(const_configs['data_file_pth'], 'geoqa', f"{train_data_name}.json")
    llama_factory_data_dir = os.path.join(llama_factory_pth, 'data')
    llama_factory_data_config = os.path.join(llama_factory_pth, 'data', 'dataset_info.json')
    new_data_config = {f"{train_data_name}": {
            "file_name": f"{train_data_name}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images"
            },
            "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant"
            }
        }
    }
        # Copy the data file to llama_factory data directory
    try:
        shutil.copy(tgt_train_file, llama_factory_data_dir)
        print(f"Copied {tgt_train_file} to {llama_factory_data_dir}.")
    except FileNotFoundError:
        print(f"Training data file not found: {tgt_train_file}")
        return
    except Exception as e:
        print(f"An error occurred while copying the file: {e}")
        return

    # Update the data_info.json configuration file
    try:
        # Load existing config data
        if os.path.exists(llama_factory_data_config):
            with open(llama_factory_data_config, 'r') as f:
                data_info = json.load(f)
        else:
            data_info = {}
        
        # Add new data configuration
        data_info.update(new_data_config)

        # Save the updated config
        with open(llama_factory_data_config, 'w') as f:
            json.dump(data_info, f, indent=4)
        print(f"Updated data_info.json with new configuration for {prefix}-{iter_name}.")
    except Exception as e:
        print(f"An error occurred while updating data_info.json: {e}")
    

    config_path = "qwen2vl_lora_sft_geoqa.yaml"  # Update this path
    dataset_name = f"{train_data_name}"  # Must match `data_info.json`
    ckpt_dir = os.path.join('saves', ckpt_dir)
    update_training_config(config_path, dataset_name, ckpt_dir)

    # Define your commands in a single bash call
    command = f"""
    cd {llama_factory_pth} &&
    export CUDA_VISIBLE_DEVICES={const_configs['CUDA_INFO']} &&
    llamafactory-cli train ../qwen2vl_lora_sft_geoqa.yaml
    """
    
    print("Executing command:", command)
    
    try:
        # Execute the entire command sequence in a single bash session
        subprocess.run(command, shell=True, check=True, executable="/bin/bash")
        print("Bash script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing bash script: {e}")
    except FileNotFoundError:
        print("Bash script not found or not executable.")

def do_infer(py_pth: str, base_model: str, lora_path: str, **kwargs):
    global const_configs
    cuda_num = const_configs['CUDA_INFO'].count(',') + 1
    # Build the command as a single string
    command = (
        f"accelerate launch --config_file accelerate_config.json --num_processes {cuda_num} {py_pth} "
        f"--base_model {base_model} "
    )
    if lora_path:
        command += f"--lora_path {lora_path} "

    for k, v in kwargs.items():
        command += f"--{k} {v} "

    print(command)

    try:
        # Use shell=True to run the command as a single shell command
        subprocess.run(command, check=True, shell=True)
        print("Bash script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing bash script: {e}")
        exit()
    except FileNotFoundError:
        print("Bash script not found or not executable.")
        exit()

def do_merge_data(cur_iter:int, config, d_type, save_dir, original_file, prefix):
    if config['select_scaling']:
        merge_data_py = os.path.join(config["data_utils_dir"], 'geoqa', 'merge_data_select.py')
    else:
        merge_data_py = os.path.join(config["data_utils_dir"], 'geoqa', 'merge_data.py')

    command = [
        'python', merge_data_py,
        f'--d_type={d_type}',
        f'--iter_num={cur_iter}',
        f'--sample_prefix={prefix}',
        f'--save_dir=geoqa',
        f'--original_file={original_file}',
    ]
    print(command)
    if const_configs['select_scaling']:
        command.append(f'--max_select_num={const_configs["max_select_num"]}')
    

    try:
        # 使用subprocess运行Bash脚本
        subprocess.run(command, check=True)
        print("Bash script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing bash script: {e}")
    except FileNotFoundError:
        print("Bash script not found or not executable.")

def do_gen_new_data(cur_iter_name, config, prefix):
    gen_data_py = os.path.join(config['data_file_pth'], 'geoqa', 'new_data_gen.py')
    command = [
        'python', gen_data_py,
        f'--iter_num={cur_iter_name}',
        f'--prefix={prefix}',
        f'--geoqa_dir={config["geoqa_data_dir"]}'
    ]
    print(command)

    try:
        # 使用subprocess运行Bash脚本
        subprocess.run(command, check=True)
        print("Bash script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing bash script: {e}")
    except FileNotFoundError:
        print("Bash script not found or not executable.")

def add_info_and_save(cur_iter_name, const_configs, mode='sampling_eval'):
    # add infor
    preds_files = os.listdir('outputs')
    preds_files = [f for f in preds_files if f.startswith(cur_iter_name) and f.endswith('json') and 'added' not in f]
    
    for pred_file in preds_files:
        pred_file = os.path.join('outputs', pred_file)
        data_type = "train" if "train" in pred_file else "test"
        if 'test_select' in pred_file:
            # self-select file have different id and needed to process
            add_info(os.path.join(const_configs["geoqa_data_dir"], f"{data_type}.jsonl"), pred_file, is_select=True)
        else:
            add_info(os.path.join(const_configs["geoqa_data_dir"], f"{data_type}.jsonl"), pred_file)
        
    if mode == 'test_eval':
        return

    # merge_data 
    prefix = '-'.join(cur_iter_name.split('-')[:-1])
    cur_iter_num = int(cur_iter_name[-1])
    d_types = ['train', 'test']
    if cur_iter_num == 0:
        d_types = ['train']
    if mode == 'select_scaling_test':
        d_types = ['test']

    for d_type in d_types:
        do_merge_data(cur_iter_num, const_configs, d_type, 'geoqa', 
                      os.path.join(const_configs['geoqa_data_dir'], f"{d_type}.jsonl"), prefix) 
    
    if mode == 'select_scaling_test':
        return
    
    # gen_new train_data 
    iter_name = cur_iter_name.split('-')[-1]
    do_gen_new_data(iter_name, const_configs, prefix+'-select-scaling')
    

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='Qwen2-VL-7B-Instruct')
    parser.add_argument('--geoqa_dir', type=str, default='geoQA-data')
    # parser.add_argument('--total_iters', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--ckpt_dir', type=str, default='Qwen2-VL-7B-Instruct')
    parser.add_argument('--max_select_num', type=int, default=6)

    args = parser.parse_args()
    
    global const_configs
    # check the CUDA_DEVICES_INFO
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        logging.info(f"Set CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        const_configs['CUDA_INFO'] = os.environ['CUDA_VISIBLE_DEVICES']
        
    const_configs['base_model_path'] = args.base_model 
    const_configs['geoqa_data_dir'] = args.geoqa_dir
    const_configs['select_scaling'] = True
    const_configs['max_select_num'] = args.max_select_num
    
    return args


if __name__ == '__main__':
    
    ckpt_dir_pattern = "Qwen2-VL-geoqa-{}"
    args = args_parser()

    match = re.search(r'iter(\d+)', args.ckpt_dir)
    iter_num = int(match.group(1))
    cur_iter = f'iter{iter_num}'
    ckpt_dir = ckpt_dir_pattern.format(cur_iter)
    # prefix_name = f"{ckpt_dir}"
    
    logging.info(f"==== Start Self-Select Scaling ====")
    logging.info(f"Base on ckpt : {args.ckpt_dir}")

    # Generate self-select-scaling data for SFT
    add_info_and_save(cur_iter_name=ckpt_dir, const_configs=const_configs)
    
    cur_iter = f'iter{iter_num+1}'
    ckpt_dir_pattern = "Qwen2-VL-geoqa-select-scaling-{}"
    ckpt_dir = ckpt_dir_pattern.format(cur_iter)
    print(cur_iter)

    # SFT with lora
    logging.info(f"Start SFT with lora for self-Select Scaling")
    logging.info(f"The checkpoint dir is LLaMA-Factory/saves/{ckpt_dir}")
    do_train(iter_name=cur_iter, prefix=ckpt_dir[:-6], ckpt_dir=ckpt_dir)
    
    
    # test data sampling for next iteration, with COT prompt
    sampling_config = {
        "iter_times": 10,
        "with_sampling": True,
        "batch_size": args.batch_size,
        "data_file": None,
        "save_name": None
    }
    logging.info(f"Start test data sampling for select scaling: with 10 times")
    sampling_config['data_file'] = os.path.join(const_configs['data_file_pth'],'geoqa_test_cot.json')
    sampling_config['save_name'] = ckpt_dir_pattern.format(f'{cur_iter}_test_sample')
    do_infer(py_pth='eval_distributed.py', base_model=const_configs['base_model_path'], lora_path=os.path.join('LLaMA-Factory/saves', ckpt_dir), **sampling_config)
    
    # Add info required for evaluation and generate the new train data
    add_info_and_save(cur_iter_name=ckpt_dir, const_configs=const_configs, mode='select_scaling_test')


    for i in range(2, args.max_select_num+1):
        infer_config = {
            "iter_times": 1,
            "with_sampling": False,
            "batch_size": args.batch_size,
            "data_file": None,
            "save_name": None
        }
        # Test time computation
        logging.info(f"Start test time computation for {cur_iter}, with self-select-{i}")
        infer_config['data_file'] = os.path.join(const_configs['data_file_pth'], const_configs['dataset'], 'self_test_data',
                                                 ckpt_dir_pattern.format(f'{cur_iter}_select_n{i}.json'))
        infer_config['save_name'] = ckpt_dir_pattern.format(f'{cur_iter}_test_select_n{i}')
        do_infer(py_pth='eval_distributed_select.py', base_model=const_configs['base_model_path'], lora_path=os.path.join('LLaMA-Factory/saves', ckpt_dir), **infer_config)

        # Evaluation
        add_info_and_save(cur_iter_name=ckpt_dir, const_configs=const_configs, mode='test_eval')
        self_select_file = os.path.join('outputs', f"{ckpt_dir}_test_select_n{i}_added.json")
        eval_res = eval_file(os.path.join(const_configs['geoqa_data_dir'], 'test.jsonl'), self_select_file)
        logging.info(f"Self-select scaling n@{i}: {eval_res}")
 