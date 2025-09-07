#!/usr/bin/env python3
"""
灵活的多GPU视频生成脚本 - 支持动态GPU选择、路径配置和增量实验
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import re
import argparse
import json
import time
import logging
import multiprocessing
from multiprocessing import Process, Queue, Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import subprocess
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from load_config import ExperimentConfig, load_config_with_args

def resolve_path(path, work_dir=None):
    """解析路径，支持相对路径和绝对路径"""
    if work_dir is None:
        work_dir = os.getcwd()
    
    if os.path.isabs(path):
        return path
    else:
        return os.path.join(work_dir, path)

def setup_logging(gpu_id, process_id, category, config_data):
    """设置日志配置"""
    work_dir = config_data['environment']['work_dir']
    log_dir = resolve_path(config_data['paths']['log_dir'], work_dir)
    log_dir = os.path.join(log_dir, f"gpu_{gpu_id}_process_{process_id}")
    
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"{log_dir}/{category}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    # 创建新的logger避免冲突
    logger = logging.getLogger(f'gpu_{gpu_id}_process_{process_id}')
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter(f'%(asctime)s - GPU{gpu_id}-P{process_id} - %(levelname)s - %(message)s')
    
    # 文件handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def detect_available_gpus():
    """检测可用的GPU"""
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                              capture_output=True, text=True, check=True)
        gpu_lines = result.stdout.strip().split('\n')
        return len(gpu_lines)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0

def get_gpu_memory_info(gpu_id):
    """获取GPU显存信息"""
    try:
        result = subprocess.run([
            'nvidia-smi', '--id=' + str(gpu_id), 
            '--query-gpu=memory.total,memory.used,memory.free', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        total, used, free = map(int, result.stdout.strip().split(', '))
        return {'total': total, 'used': used, 'free': free}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {'total': 0, 'used': 0, 'free': 0}

def select_gpus(config_data, specified_gpus=None):
    """智能选择GPU"""
    gpu_config = config_data['experiment']['gpu_config']
    
    # 检测可用GPU数量
    total_gpus = detect_available_gpus()
    if total_gpus == 0:
        raise RuntimeError("没有检测到可用的GPU")
    
    print(f"检测到 {total_gpus} 个GPU")
    
    # 如果手动指定了GPU
    if specified_gpus:
        selected_gpus = []
        for gpu_id in specified_gpus:
            if gpu_id < total_gpus:
                memory_info = get_gpu_memory_info(gpu_id)
                memory_gb = memory_info['total'] / 1024
                if memory_gb >= gpu_config['min_memory_gb']:
                    selected_gpus.append(gpu_id)
                    print(f"选择GPU {gpu_id}: {memory_gb:.1f}GB 显存")
                else:
                    print(f"跳过GPU {gpu_id}: 显存不足 ({memory_gb:.1f}GB < {gpu_config['min_memory_gb']}GB)")
            else:
                print(f"跳过GPU {gpu_id}: 超出可用范围 (0-{total_gpus-1})")
        return selected_gpus
    
    # 自动选择GPU
    if gpu_config['auto_detect']:
        selected_gpus = []
        exclude_gpus = set(gpu_config.get('exclude_gpus', []))
        
        for gpu_id in range(total_gpus):
            if gpu_id in exclude_gpus:
                print(f"跳过GPU {gpu_id}: 在排除列表中")
                continue
                
            memory_info = get_gpu_memory_info(gpu_id)
            memory_gb = memory_info['total'] / 1024
            
            if memory_gb >= gpu_config['min_memory_gb']:
                selected_gpus.append(gpu_id)
                print(f"自动选择GPU {gpu_id}: {memory_gb:.1f}GB 显存")
            else:
                print(f"跳过GPU {gpu_id}: 显存不足 ({memory_gb:.1f}GB)")
        
        return selected_gpus
    
    # 使用配置中的available_gpus
    return gpu_config.get('available_gpus', list(range(min(4, total_gpus))))

def get_next_video_path(output_dir="results", prefix="video", ext=".mp4"):
    """获取下一个可用的视频路径"""
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(ext)}$")
    nums = []
    for fn in os.listdir(output_dir):
        m = pattern.match(fn)
        if m:
            nums.append(int(m.group(1)))
    next_num = max(nums) + 1 if nums else 1
    return os.path.join(output_dir, f"{prefix}{next_num}{ext}")

def check_existing_results(output_dir, video_name, seed):
    """检查结果是否已存在"""
    pattern = f"{os.path.splitext(video_name)[0]}_seed_{seed}.mp4"
    target_file = os.path.join(output_dir, pattern)
    return os.path.exists(target_file)

def setup_gpu_device(gpu_id, memory_fraction=0.45):
    """设置GPU设备并限制显存使用"""
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        # 设置显存分配策略
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(memory_fraction, gpu_id)
        return f"cuda:{gpu_id}"
    else:
        return "cpu"

def load_video_files_from_category(category_path):
    """从类别目录加载所有视频文件"""
    video_files = []
    if os.path.exists(category_path):
        for file in os.listdir(category_path):
            if file.endswith(('.mp4', '.mov', '.avi')):
                video_files.append(os.path.join(category_path, file))
    return sorted(video_files)

def split_video_files(video_files, num_processes):
    """将视频文件分割为多个子集"""
    if not video_files:
        return [[] for _ in range(num_processes)]
    
    # 平均分配，处理余数
    chunk_size = len(video_files) // num_processes
    remainder = len(video_files) % num_processes
    
    chunks = []
    start_idx = 0
    
    for i in range(num_processes):
        # 前remainder个进程多分配一个文件
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        chunks.append(video_files[start_idx:end_idx])
        start_idx = end_idx
    
    return chunks

def worker_process(gpu_id, process_id, video_chunk, config_data, shared_results, mode='batch'):
    """工作进程函数"""
    logger = None
    try:
        # 获取类别信息
        category = config_data['experiment']['gpu_categories'][str(gpu_id)]
        
        # 设置日志
        logger = setup_logging(gpu_id, process_id, category, config_data)
        
        # 设置GPU设备和显存限制
        device = setup_gpu_device(gpu_id, config_data['experiment']['parallel_config']['memory_fraction_per_process'])
        
        logger.info(f"工作进程启动 - GPU: {gpu_id}, 进程: {process_id}, 设备: {device}, 模式: {mode}")
        logger.info(f"分配的视频数量: {len(video_chunk)}")
        
        if not video_chunk:
            logger.warning("没有分配到视频文件")
            return
        
        # 解析路径
        work_dir = config_data['environment']['work_dir']
        model_base_path = resolve_path(config_data['model']['base_path'], work_dir)
        
        # 加载模型
        logger.info("加载模型...")
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device=device)
        
        # 构建模型路径
        diffusion_models = [os.path.join(model_base_path, model) for model in config_data['model']['diffusion_models']]
        t5_model_path = os.path.join(model_base_path, config_data['model']['t5_model'])
        vae_model_path = os.path.join(model_base_path, config_data['model']['vae_model'])
        
        model_manager.load_models([
            diffusion_models,
            t5_model_path,
            vae_model_path
        ])
        
        # 创建pipeline
        pipeline = WanVideoPipeline.from_model_manager(
            model_manager,
            torch_dtype=torch.bfloat16,
            device=device
        )
        
        # 加载prompts
        prompt_file_path = resolve_path(config_data['paths']['prompt_file'], work_dir)
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)
        
        # 获取输出目录
        output_base_dir = resolve_path(config_data['paths']['output_dir'], work_dir)
        
        # 处理分配的视频
        process_results = []
        is_incremental = mode == 'incremental' and config_data['experiment']['modes']['incremental']['skip_existing']
        
        for video_idx, video_path in enumerate(video_chunk):
            try:
                video_name = os.path.basename(video_path)
                logger.info(f"处理视频 [{video_idx+1}/{len(video_chunk)}]: {video_name}")
                
                # 查找对应的prompt
                matching_prompt = None
                video_name_without_ext = os.path.splitext(video_name)[0]
                
                # 首先尝试新格式（嵌套结构）
                if isinstance(prompts_data, dict) and category in prompts_data:
                    category_prompts = prompts_data[category]
                    if video_name_without_ext in category_prompts:
                        prompts_list = category_prompts[video_name_without_ext]
                        if isinstance(prompts_list, list) and len(prompts_list) > 0:
                            matching_prompt = prompts_list[0]  # 使用第一个prompt
                
                # 如果没找到，尝试旧格式（列表结构）
                if not matching_prompt and isinstance(prompts_data, list):
                    for prompt_item in prompts_data:
                        if prompt_item['video_name'] == video_name:
                            matching_prompt = prompt_item['prompt']
                            break
                
                if not matching_prompt:
                    logger.warning(f"未找到视频 {video_name} 对应的prompt，跳过")
                    continue
                
                # 使用配置的seeds生成视频
                for seed in config_data['experiment']['seeds']:
                    # 设置输出路径
                    output_subdir = os.path.join(
                        output_base_dir,
                        category,
                        f"gpu_{gpu_id}_process_{process_id}"
                    )
                    os.makedirs(output_subdir, exist_ok=True)
                    
                    # 检查是否已存在（增量模式）
                    if is_incremental and check_existing_results(output_subdir, video_name, seed):
                        logger.info(f"跳过已存在的结果: {video_name}_seed_{seed}")
                        continue
                    
                    start_time = time.time()
                    
                    output_path = os.path.join(
                        output_subdir,
                        f"{os.path.splitext(video_name)[0]}_seed_{seed}.mp4"
                    )
                    
                    logger.info(f"生成视频 - Seed: {seed}, 输出: {output_path}")
                    
                    # 生成视频
                    video_data = VideoData(video_file=video_path)
                    
                    result_video = pipeline(
                        prompt=matching_prompt,
                        num_inference_steps=config_data['experiment']['generation_params']['num_inference_steps'],
                        num_frames=config_data['experiment']['generation_params']['num_frames'],
                        height=config_data['experiment']['generation_params']['height'],
                        width=config_data['experiment']['generation_params']['width'],
                        fps=config_data['experiment']['generation_params']['fps'],
                        generator=torch.Generator(device=device).manual_seed(seed),
                        negative_prompt=config_data['experiment']['generation_params']['negative_prompt']
                    )
                    
                    # 保存视频
                    save_video(
                        result_video,
                        output_path,
                        fps=config_data['experiment']['generation_params']['fps'],
                        quality=config_data['experiment']['generation_params']['quality']
                    )
                    
                    elapsed_time = time.time() - start_time
                    
                    # 记录结果
                    result_info = {
                        'gpu_id': gpu_id,
                        'process_id': process_id,
                        'video_name': video_name,
                        'seed': seed,
                        'output_path': output_path,
                        'elapsed_time': elapsed_time,
                        'mode': mode,
                        'success': True
                    }
                    
                    process_results.append(result_info)
                    logger.info(f"视频生成完成 - 耗时: {elapsed_time:.2f}s")
                    
                    # 清理GPU缓存
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"处理视频 {video_path} 时出错: {str(e)}")
                result_info = {
                    'gpu_id': gpu_id,
                    'process_id': process_id,
                    'video_name': video_name,
                    'error': str(e),
                    'mode': mode,
                    'success': False
                }
                process_results.append(result_info)
        
        # 保存进程结果到共享存储
        shared_results[f"gpu_{gpu_id}_process_{process_id}"] = process_results
        logger.info(f"进程完成 - 总共处理: {len(process_results)} 个任务")
        
    except Exception as e:
        error_msg = f"工作进程异常退出: {str(e)}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        if 'shared_results' in locals():
            shared_results[f"gpu_{gpu_id}_process_{process_id}"] = [{'error': str(e), 'success': False}]

def run_experiments(config_data, selected_gpus, categories=None, mode='batch'):
    """运行实验"""
    parallel_config = config_data['experiment']['parallel_config']
    num_processes = parallel_config['processes_per_gpu']
    
    print(f"启动 {mode} 模式实验")
    print(f"使用GPU: {selected_gpus}")
    print(f"每GPU进程数: {num_processes}")
    
    # 获取工作目录
    work_dir = config_data['environment']['work_dir']
    
    # 创建共享存储用于收集结果
    manager = Manager()
    shared_results = manager.dict()
    
    # 启动所有GPU的进程
    all_processes = []
    
    for gpu_id in selected_gpus:
        # 确定该GPU处理的类别
        if categories:
            # 如果指定了类别，使用指定的类别
            category = categories[0] if len(categories) == 1 else categories[gpu_id % len(categories)]
        else:
            # 使用配置中的类别映射
            category = config_data['experiment']['gpu_categories'].get(str(gpu_id))
        
        if not category:
            print(f"跳过GPU {gpu_id}: 未配置类别")
            continue
        
        print(f"GPU {gpu_id} 处理类别: {category}")
        
        # 获取该类别的所有视频文件
        category_path = resolve_path(config_data['paths']['categories'][category], work_dir)
        video_files = load_video_files_from_category(category_path)
        
        if not video_files:
            print(f"警告: 类别 {category} 中没有找到视频文件")
            continue
        
        print(f"类别 {category} 找到 {len(video_files)} 个视频文件")
        
        # 分割视频文件
        video_chunks = split_video_files(video_files, num_processes)
        
        # 启动该GPU的多个进程
        for process_id in range(num_processes):
            if video_chunks[process_id]:  # 只启动有任务的进程
                p = Process(
                    target=worker_process,
                    args=(gpu_id, process_id, video_chunks[process_id], config_data, shared_results, mode)
                )
                all_processes.append(p)
                p.start()
                print(f"启动GPU {gpu_id} 进程 {process_id} (PID: {p.pid})")
    
    # 等待所有进程完成
    for p in all_processes:
        p.join()
    
    # 收集和保存结果
    all_results = []
    for key, results in shared_results.items():
        all_results.extend(results)
    
    # 保存实验结果
    output_dir = resolve_path(config_data['paths']['output_dir'], work_dir)
    results_file = os.path.join(output_dir, f"experiment_results_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    successful_tasks = len([r for r in all_results if r.get('success', False)])
    print(f"\n{mode} 模式实验完成!")
    print(f"结果已保存到: {results_file}")
    print(f"总共完成: {successful_tasks} 个成功任务")
    
    return all_results

def main():
    # 设置多进程启动方式为spawn以支持CUDA
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='灵活的多GPU视频生成实验')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--gpus', type=int, nargs='+', help='手动指定使用的GPU ID列表')
    parser.add_argument('--categories', type=str, nargs='+', help='指定处理的类别')
    parser.add_argument('--mode', type=str, choices=['batch', 'incremental', 'single'], 
                       default='batch', help='运行模式')
    parser.add_argument('--new-data-dir', type=str, help='新增数据目录（增量模式）')
    parser.add_argument('--new-prompts', type=str, help='新增prompts文件（增量模式）')
    
    args = parser.parse_args()
    
    print(f"启动灵活实验系统")
    print(f"配置文件: {args.config}")
    print(f"运行模式: {args.mode}")
    
    # 加载配置
    config = load_config_with_args(args.config, argparse.Namespace(**vars(args)))
    config_data = config.config  # 获取原始配置字典
    
    # 选择GPU
    selected_gpus = select_gpus(config_data, args.gpus)
    
    if not selected_gpus:
        print("错误: 没有可用的GPU")
        return
    
    # 运行实验
    results = run_experiments(config_data, selected_gpus, args.categories, args.mode)
    
    print(f"\n实验系统完成! 共处理 {len(results)} 个任务")

if __name__ == "__main__":
    main()
