#!/usr/bin/env python3
"""
Wan2.1-T2V-14B 多GPU分布式实验主控脚本
负责启动和管理所有GPU工作进程
"""

import os
import sys
import time
import subprocess
import argparse
import signal
from pathlib import Path
from typing import List, Dict
import multiprocessing as mp

from experiment_utils import ExperimentManager, print_experiment_summary

class ExperimentController:
    """实验控制器"""
    
    def __init__(self, num_processes_per_gpu: int = 1):
        self.manager = ExperimentManager()
        self.num_processes_per_gpu = num_processes_per_gpu
        self.num_gpus = 4
        self.processes = []
        self.tmux_sessions = []
        
    def check_prerequisites(self):
        """检查实验前置条件"""
        print("检查实验前置条件...")
        
        # 检查GPU
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError("nvidia-smi命令失败")
            print(f"✓ GPU检查通过")
        except Exception as e:
            raise RuntimeError(f"GPU检查失败: {e}")
        
        # 检查模型文件
        model_base_path = Path("/root/autodl-tmp/pretrained_model/Wan-AI/Wan2.1-T2V-14B")
        required_files = [
            "diffusion_pytorch_model-00001-of-00006.safetensors",
            "diffusion_pytorch_model-00002-of-00006.safetensors", 
            "diffusion_pytorch_model-00003-of-00006.safetensors",
            "diffusion_pytorch_model-00004-of-00006.safetensors",
            "diffusion_pytorch_model-00005-of-00006.safetensors",
            "diffusion_pytorch_model-00006-of-00006.safetensors",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.1_VAE.pth"
        ]
        
        for file_name in required_files:
            file_path = model_base_path / file_name
            if not file_path.exists():
                raise RuntimeError(f"模型文件不存在: {file_path}")
        print(f"✓ 模型文件检查通过")
        
        # 检查数据集
        prompt_file = self.manager.dataset_dir / "prompt.json"
        if not prompt_file.exists():
            raise RuntimeError(f"Prompt文件不存在: {prompt_file}")
        print(f"✓ 数据集检查通过")
        
        # 检查参考视频
        missing_videos = []
        for gpu_id in range(self.num_gpus):
            category = self.manager.gpu_categories[gpu_id]
            video_files = self.manager.get_video_files_for_category(category)
            
            for video_name in video_files:
                video_path = self.manager.dataset_dir / category / f"{video_name}.mp4"
                if not video_path.exists():
                    missing_videos.append(str(video_path))
        
        if missing_videos:
            print(f"警告: 发现缺失的参考视频文件:")
            for video in missing_videos[:5]:  # 只显示前5个
                print(f"  - {video}")
            if len(missing_videos) > 5:
                print(f"  ... 和其他 {len(missing_videos)-5} 个文件")
        else:
            print(f"✓ 参考视频文件检查通过")
        
        # 检查存储空间
        statvfs = os.statvfs("/root/autodl-tmp")
        free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        if free_space_gb < 50:
            print(f"警告: 可用存储空间仅 {free_space_gb:.1f}GB，建议至少50GB")
        else:
            print(f"✓ 存储空间充足 ({free_space_gb:.1f}GB)")
        
        print("前置条件检查完成\n")
        
    def create_tmux_session(self, session_name: str, command: str):
        """创建tmux会话"""
        # 检查会话是否已存在
        check_cmd = ["tmux", "has-session", "-t", session_name]
        result = subprocess.run(check_cmd, capture_output=True)
        
        if result.returncode == 0:
            print(f"tmux会话 {session_name} 已存在，跳过创建")
            return
        
        # 创建新会话
        create_cmd = ["tmux", "new-session", "-d", "-s", session_name, command]
        result = subprocess.run(create_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"创建tmux会话: {session_name}")
            self.tmux_sessions.append(session_name)
        else:
            print(f"创建tmux会话失败: {session_name}")
            print(f"错误信息: {result.stderr}")
    
    def start_gpu_workers_in_tmux(self):
        """在tmux中启动所有GPU工作进程"""
        print("在tmux中启动GPU工作进程...")
        
        # 激活conda环境的命令
        conda_activate = "source /root/miniconda3/etc/profile.d/conda.sh && conda activate diffsynth"
        
        for gpu_id in range(self.num_gpus):
            category = self.manager.gpu_categories[gpu_id]
            
            for process_id in range(self.num_processes_per_gpu):
                session_name = f"wan_gpu{gpu_id}_proc{process_id}"
                
                # 构建工作命令
                work_cmd = (
                    f"cd /root/autodl-tmp && "
                    f"python gpu_worker.py "
                    f"--gpu_id {gpu_id} "
                    f"--process_id {process_id} "
                    f"--num_processes {self.num_processes_per_gpu}"
                )
                
                # 完整命令：激活环境 + 工作命令
                full_cmd = f"{conda_activate} && {work_cmd}"
                
                self.create_tmux_session(session_name, full_cmd)
                
        print(f"启动了 {self.num_gpus * self.num_processes_per_gpu} 个工作进程")
        
    def start_monitoring_session(self):
        """启动监控会话"""
        session_name = "wan_monitor"
        
        # 监控脚本
        monitor_script = '''
        watch -n 10 "
        echo '=== GPU状态 ==='
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
        echo ''
        echo '=== 实验进度 ==='
        find /root/autodl-tmp/experiment_results -name '*.mp4' 2>/dev/null | wc -l | xargs echo '已生成视频数:'
        echo ''
        echo '=== 各类别进度 ==='
        for category in camera_motion single_object multiple_objects complex_human_motion; do
            count=$(find /root/autodl-tmp/experiment_results/$category -name '*.mp4' 2>/dev/null | wc -l)
            echo \"$category: $count\"
        done
        echo ''
        echo '=== tmux会话状态 ==='
        tmux list-sessions 2>/dev/null | grep wan_ || echo '无活动会话'
        "
        '''
        
        conda_activate = "source /root/miniconda3/etc/profile.d/conda.sh && conda activate diffsynth"
        full_cmd = f"{conda_activate} && {monitor_script}"
        
        self.create_tmux_session(session_name, full_cmd)
        
    def list_tmux_sessions(self):
        """列出所有相关的tmux会话"""
        try:
            result = subprocess.run(["tmux", "list-sessions"], capture_output=True, text=True)
            if result.returncode == 0:
                sessions = [line for line in result.stdout.split('\n') if 'wan_' in line]
                if sessions:
                    print("活动的tmux会话:")
                    for session in sessions:
                        print(f"  {session}")
                else:
                    print("没有找到相关的tmux会话")
            else:
                print("没有活动的tmux会话")
        except Exception as e:
            print(f"获取tmux会话列表失败: {e}")
    
    def stop_all_sessions(self):
        """停止所有相关的tmux会话"""
        try:
            result = subprocess.run(["tmux", "list-sessions", "-F", "#{session_name}"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                sessions = [s.strip() for s in result.stdout.split('\n') if s.strip().startswith('wan_')]
                
                for session in sessions:
                    subprocess.run(["tmux", "kill-session", "-t", session], 
                                 capture_output=True)
                    print(f"停止会话: {session}")
                    
                if sessions:
                    print(f"已停止 {len(sessions)} 个会话")
                else:
                    print("没有找到要停止的会话")
            else:
                print("没有活动的tmux会话")
        except Exception as e:
            print(f"停止会话失败: {e}")
    
    def show_progress(self):
        """显示实验进度"""
        print("=" * 60)
        print("实验进度报告")
        print("=" * 60)
        
        try:
            # 总体进度
            results_dir = Path("/root/autodl-tmp/experiment_results")
            total_generated = len(list(results_dir.glob("**/*.mp4")))
            
            # 各类别进度
            categories_progress = {}
            total_expected = 0
            
            for gpu_id in range(self.num_gpus):
                category = self.manager.gpu_categories[gpu_id]
                category_dir = results_dir / category
                
                if category_dir.exists():
                    generated = len(list(category_dir.glob("**/*.mp4")))
                else:
                    generated = 0
                
                # 计算预期数量
                video_files = self.manager.get_video_files_for_category(category)
                expected = 0
                for video_name in video_files:
                    prompts = self.manager.get_prompts_for_video(video_name)
                    expected += len(prompts) * len(self.manager.seeds)
                
                categories_progress[category] = {
                    'generated': generated,
                    'expected': expected,
                    'progress': generated / expected * 100 if expected > 0 else 0
                }
                total_expected += expected
            
            print(f"总体进度: {total_generated}/{total_expected} ({total_generated/total_expected*100:.1f}%)")
            print()
            
            for category, stats in categories_progress.items():
                print(f"{category:20s}: {stats['generated']:3d}/{stats['expected']:3d} ({stats['progress']:5.1f}%)")
            
            print("=" * 60)
            
        except Exception as e:
            print(f"获取进度信息失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="Wan2.1-T2V-14B 多GPU实验控制器")
    parser.add_argument("--action", choices=['start', 'stop', 'status', 'progress', 'summary'], 
                       default='start', help="执行的操作")
    parser.add_argument("--processes_per_gpu", type=int, default=1, 
                       help="每个GPU的进程数 (默认: 1)")
    args = parser.parse_args()
    
    controller = ExperimentController(args.processes_per_gpu)
    
    if args.action == 'summary':
        print_experiment_summary(controller.manager)
        
    elif args.action == 'start':
        try:
            # 显示实验概要
            print_experiment_summary(controller.manager)
            
            # 检查前置条件
            controller.check_prerequisites()
            
            # 启动实验
            print("开始启动实验...")
            controller.start_gpu_workers_in_tmux()
            controller.start_monitoring_session()
            
            print("\n实验启动完成！")
            print("\n使用以下命令监控实验:")
            print("  查看监控界面: tmux attach -t wan_monitor")
            print("  查看特定进程: tmux attach -t wan_gpu0_proc0")
            print("  查看所有会话: tmux list-sessions")
            print("  查看进度: python run_experiment.py --action progress")
            print("  停止实验: python run_experiment.py --action stop")
            
        except Exception as e:
            print(f"启动失败: {e}")
            sys.exit(1)
    
    elif args.action == 'stop':
        print("停止所有实验会话...")
        controller.stop_all_sessions()
        
    elif args.action == 'status':
        controller.list_tmux_sessions()
        
    elif args.action == 'progress':
        controller.show_progress()

if __name__ == "__main__":
    main()
