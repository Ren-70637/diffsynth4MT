#!/usr/bin/env python3
"""
单GPU工作进程脚本
负责在指定GPU上运行Wan2.1-T2V-14B模型进行视频生成
"""

import os
import sys
import torch
import argparse
import time
import traceback
from pathlib import Path
from typing import List

# 添加diffsynth到路径
sys.path.append("/root/autodl-tmp/diffsynth4MT-main")

from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from experiment_utils import ExperimentManager, TaskInfo

class GPUWorker:
    """单GPU工作器"""
    
    def __init__(self, gpu_id: int, process_id: int):
        self.gpu_id = gpu_id
        self.process_id = process_id
        self.device = f"cuda:{gpu_id}"
        
        # 设置CUDA设备和优化参数
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.set_device(0)  # 由于设置了CUDA_VISIBLE_DEVICES，这里使用0
        
        # 初始化实验管理器
        self.manager = ExperimentManager()
        self.logger = self.manager.setup_logging(gpu_id, process_id)
        
        # 模型相关
        self.model_manager = None
        self.pipeline = None
        self.model_loaded = False
        
    def load_models(self):
        """加载Wan2.1-T2V-14B模型"""
        try:
            self.logger.info(f"开始在GPU {self.gpu_id}上加载模型...")
            start_time = time.time()
            
            # 初始化模型管理器
            self.model_manager = ModelManager(device="cpu")
            
            # 加载模型文件
            model_paths = [
                [
                    "/root/autodl-tmp/pretrained_model/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
                    "/root/autodl-tmp/pretrained_model/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
                    "/root/autodl-tmp/pretrained_model/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
                    "/root/autodl-tmp/pretrained_model/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
                    "/root/autodl-tmp/pretrained_model/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
                    "/root/autodl-tmp/pretrained_model/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
                ],
                "/root/autodl-tmp/pretrained_model/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
                "/root/autodl-tmp/pretrained_model/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
            ]
            
            self.model_manager.load_models(
                model_paths,
                torch_dtype=torch.bfloat16  # 使用bfloat16以节省显存
            )
            
            # 创建pipeline
            self.pipeline = WanVideoPipeline.from_model_manager(
                self.model_manager, 
                torch_dtype=torch.bfloat16, 
                device="cuda"  # 由于设置了CUDA_VISIBLE_DEVICES，这里使用cuda
            )
            
            # 启用显存管理 - 设置较小的持久参数数量以节省显存
            self.pipeline.enable_vram_management(num_persistent_param_in_dit=10)
            
            # 启用CPU offload来动态管理模型组件
            self.pipeline.enable_cpu_offload()
            
            load_time = time.time() - start_time
            self.logger.info(f"模型加载完成，耗时: {load_time:.2f}秒")
            self.model_loaded = True
            
            # 打印显存使用情况
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                self.logger.info(f"当前显存使用: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
                
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise e
    
    def process_single_task(self, task: TaskInfo) -> bool:
        """处理单个任务"""
        try:
            # 检查输出文件是否已存在
            if self.manager.check_task_completion(task):
                self.logger.info(f"任务已完成，跳过: {task.output_path}")
                return True
            
            self.logger.info(f"开始处理任务: {task.video_filename} | prompt_{task.prompt_idx} | seed_{task.seed}")
            start_time = time.time()
            
            # 加载参考视频
            if not os.path.exists(task.reference_video_path):
                self.logger.error(f"参考视频不存在: {task.reference_video_path}")
                return False
                
            input_video = VideoData(task.reference_video_path, height=480, width=832)
            
            # 推理前：卸载所有模型到CPU以释放显存
            self.pipeline.load_models_to_device([])
            torch.cuda.empty_cache()
            
            # 生成视频
            output_video = self.pipeline(
                prompt=task.prompt_text,
                negative_prompt="光线混乱，变形，扭曲，几何畸变，发光边缘，光晕，运动模糊过曝，静态，画面扭曲，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                num_inference_steps=self.manager.num_inference_steps,
                input_video=input_video,
                seed=task.seed,
                tiled=True,
                num_frames=self.manager.num_frames,
            )
            
            # 保存视频
            os.makedirs(os.path.dirname(task.output_path), exist_ok=True)
            save_video(output_video, task.output_path, fps=15, quality=5)
            
            process_time = time.time() - start_time
            self.logger.info(f"任务完成: {task.output_path}, 耗时: {process_time:.2f}秒")
            
            # 推理后：卸载所有模型到CPU并清理显存
            self.pipeline.load_models_to_device([])
            torch.cuda.empty_cache()
            
            # 额外的显存清理
            import gc
            gc.collect()
            torch.cuda.empty_cache()
                
            return True
            
        except Exception as e:
            self.logger.error(f"任务处理失败: {task.output_path}")
            self.logger.error(f"错误信息: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def process_task_list(self, tasks: List[TaskInfo]):
        """处理任务列表"""
        if not self.model_loaded:
            self.logger.error("模型未加载，无法处理任务")
            return
            
        total_tasks = len(tasks)
        completed_tasks = 0
        failed_tasks = 0
        
        self.logger.info(f"开始处理 {total_tasks} 个任务")
        
        for i, task in enumerate(tasks, 1):
            self.logger.info(f"进度: {i}/{total_tasks} ({i/total_tasks*100:.1f}%)")
            
            success = self.process_single_task(task)
            if success:
                completed_tasks += 1
            else:
                failed_tasks += 1
                
            # 打印当前统计
            self.logger.info(f"当前统计 - 完成: {completed_tasks}, 失败: {failed_tasks}, 剩余: {total_tasks-i}")
            
        self.logger.info(f"所有任务处理完成 - 成功: {completed_tasks}, 失败: {failed_tasks}")

def main():
    parser = argparse.ArgumentParser(description="GPU工作进程")
    parser.add_argument("--gpu_id", type=int, required=True, help="GPU ID")
    parser.add_argument("--process_id", type=int, required=True, help="进程ID")
    parser.add_argument("--num_processes", type=int, default=2, help="每GPU进程数")
    args = parser.parse_args()
    
    # 创建工作器
    worker = GPUWorker(args.gpu_id, args.process_id)
    
    try:
        # 加载模型
        worker.load_models()
        
        # 获取该GPU的所有任务
        manager = ExperimentManager()
        all_gpu_tasks = manager.generate_all_tasks()
        gpu_tasks = all_gpu_tasks[args.gpu_id]
        
        # 过滤掉已完成的任务
        incomplete_tasks = manager.get_incomplete_tasks(gpu_tasks)
        worker.logger.info(f"GPU {args.gpu_id} 总任务数: {len(gpu_tasks)}, 未完成: {len(incomplete_tasks)}")
        
        # 将任务分配给当前进程
        process_chunks = manager.split_tasks_for_processes(incomplete_tasks, args.num_processes)
        
        if args.process_id < len(process_chunks):
            my_tasks = process_chunks[args.process_id]
            worker.logger.info(f"进程 {args.process_id} 分配到 {len(my_tasks)} 个任务")
            
            # 处理任务
            worker.process_task_list(my_tasks)
        else:
            worker.logger.info(f"进程 {args.process_id} 没有分配到任务")
            
    except Exception as e:
        worker.logger.error(f"工作进程异常退出: {str(e)}")
        worker.logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
