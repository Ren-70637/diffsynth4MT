#!/usr/bin/env python3
"""
Wan2.1-T2V-14B 多GPU实验工具函数
提供任务分配、文件管理、进度跟踪等功能
"""

import os
import json
import re
import time
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
import multiprocessing as mp
from dataclasses import dataclass

@dataclass
class TaskInfo:
    """单个任务信息"""
    video_category: str
    video_filename: str
    prompt_idx: int
    prompt_text: str
    seed: int
    output_path: str
    reference_video_path: str

class ExperimentManager:
    """实验管理器"""
    
    def __init__(self, base_dir: str = "/root/autodl-tmp"):
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / "Final_Dataset"
        self.results_dir = self.base_dir / "experiment_results"
        self.logs_dir = self.results_dir / "logs"
        
        # 创建输出目录
        self.results_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # 加载prompt数据
        with open(self.dataset_dir / "prompt.json", 'r', encoding='utf-8') as f:
            self.prompt_data = json.load(f)
            
        # GPU-类别映射
        self.gpu_categories = {
            0: "camera_motion",
            1: "single_object", 
            2: "multiple_objects",
            3: "complex_human_motion"
        }
        
        # 实验配置
        self.seeds = [42, 142]
        self.num_inference_steps = 50
        self.num_frames = 81  # 修改为81帧
        
    def get_video_files_for_category(self, category: str) -> List[str]:
        """获取指定类别的所有视频文件"""
        category_dir = self.dataset_dir / category
        video_files = []
        
        if category_dir.exists():
            for video_file in category_dir.glob("*.mp4"):
                # 去掉.mp4后缀作为基础名称
                base_name = video_file.stem
                video_files.append(base_name)
                
        return sorted(video_files)
    
    def get_prompts_for_video(self, video_base_name: str) -> List[str]:
        """获取指定视频的所有prompt"""
        # 在prompt.json中查找匹配的prompts
        for category_data in self.prompt_data.values():
            if isinstance(category_data, dict):
                for prompt_key, prompts in category_data.items():
                    if prompt_key == video_base_name:
                        return prompts
        return []
    
    def generate_all_tasks(self) -> Dict[int, List[TaskInfo]]:
        """生成所有任务，按GPU分组"""
        gpu_tasks = {gpu_id: [] for gpu_id in range(4)}
        
        for gpu_id, category in self.gpu_categories.items():
            video_files = self.get_video_files_for_category(category)
            
            for video_base_name in video_files:
                prompts = self.get_prompts_for_video(video_base_name)
                reference_video_path = str(self.dataset_dir / category / f"{video_base_name}.mp4")
                
                # 为每个prompt和seed组合创建任务
                for prompt_idx, prompt_text in enumerate(prompts):
                    for seed in self.seeds:
                        # 创建输出目录
                        output_dir = self.results_dir / category / video_base_name
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        output_filename = f"seed_{seed}_prompt_{prompt_idx}.mp4"
                        output_path = str(output_dir / output_filename)
                        
                        task = TaskInfo(
                            video_category=category,
                            video_filename=video_base_name,
                            prompt_idx=prompt_idx,
                            prompt_text=prompt_text,
                            seed=seed,
                            output_path=output_path,
                            reference_video_path=reference_video_path
                        )
                        
                        gpu_tasks[gpu_id].append(task)
        
        return gpu_tasks
    
    def split_tasks_for_processes(self, tasks: List[TaskInfo], num_processes: int) -> List[List[TaskInfo]]:
        """将任务列表分割为多个进程"""
        if num_processes <= 0:
            return [tasks]
            
        chunk_size = len(tasks) // num_processes
        remainder = len(tasks) % num_processes
        
        chunks = []
        start_idx = 0
        
        for i in range(num_processes):
            # 如果有余数，前几个进程多分配一个任务
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size
            
            chunks.append(tasks[start_idx:end_idx])
            start_idx = end_idx
            
        return chunks
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """获取任务统计信息"""
        gpu_tasks = self.generate_all_tasks()
        stats = {}
        
        total_tasks = 0
        for gpu_id, tasks in gpu_tasks.items():
            category = self.gpu_categories[gpu_id]
            video_count = len(set(task.video_filename for task in tasks))
            task_count = len(tasks)
            
            stats[f"GPU_{gpu_id}_{category}"] = {
                "video_count": video_count,
                "task_count": task_count,
                "prompts_per_video": task_count // video_count if video_count > 0 else 0
            }
            total_tasks += task_count
            
        stats["total_tasks"] = total_tasks
        return stats
    
    def check_task_completion(self, task: TaskInfo) -> bool:
        """检查任务是否已完成（输出文件是否存在）"""
        return os.path.exists(task.output_path) and os.path.getsize(task.output_path) > 0
    
    def get_incomplete_tasks(self, tasks: List[TaskInfo]) -> List[TaskInfo]:
        """获取未完成的任务列表"""
        return [task for task in tasks if not self.check_task_completion(task)]
    
    def setup_logging(self, gpu_id: int, process_id: int) -> logging.Logger:
        """设置日志记录"""
        log_filename = self.logs_dir / f"gpu{gpu_id}_process{process_id}.log"
        
        logger = logging.getLogger(f"GPU{gpu_id}_P{process_id}")
        logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if not logger.handlers:
            handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # 同时输出到控制台
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger

def get_next_video_path(output_dir: str = "results", prefix: str = "video", ext: str = ".mp4") -> str:
    """
    获取下一个可用的视频文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(ext)}$")
    nums = []
    
    for fn in os.listdir(output_dir):
        m = pattern.match(fn)
        if m:
            nums.append(int(m.group(1)))
            
    next_num = max(nums) + 1 if nums else 1
    return os.path.join(output_dir, f"{prefix}{next_num}{ext}")

def print_experiment_summary(manager: ExperimentManager):
    """打印实验概要信息"""
    stats = manager.get_task_statistics()
    
    print("=" * 80)
    print("Wan2.1-T2V-14B 多GPU实验概要")
    print("=" * 80)
    
    for key, value in stats.items():
        if key.startswith("GPU_"):
            print(f"{key}:")
            print(f"  视频数量: {value['video_count']}")
            print(f"  任务数量: {value['task_count']}")
            print(f"  每视频prompt数: {value['prompts_per_video']}")
            print()
    
    print(f"总任务数量: {stats['total_tasks']}")
    print(f"预计总生成视频数: {stats['total_tasks']}")
    print("=" * 80)

if __name__ == "__main__":
    # 测试功能
    manager = ExperimentManager()
    print_experiment_summary(manager)
    
    # 生成任务示例
    gpu_tasks = manager.generate_all_tasks()
    for gpu_id, tasks in gpu_tasks.items():
        print(f"\nGPU {gpu_id} ({manager.gpu_categories[gpu_id]}) - 前3个任务:")
        for i, task in enumerate(tasks[:3]):
            print(f"  {i+1}. {task.video_filename} | prompt_{task.prompt_idx} | seed_{task.seed}")
        print(f"  ... 总计 {len(tasks)} 个任务")
