#!/usr/bin/env python3
"""
配置文件加载工具模块
用于加载实验配置文件并提供参数访问接口
"""

import json
import os
import argparse
from typing import Dict, Any, List


class ExperimentConfig:
    """实验配置管理类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认为 ./config.json
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"成功加载配置文件: {self.config_path}")
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误: {e}")
    
    # 环境配置
    @property
    def conda_path(self) -> str:
        return self.config["environment"]["conda_path"]
    
    @property
    def conda_env(self) -> str:
        return self.config["environment"]["conda_env"]
    
    @property
    def work_dir(self) -> str:
        return self.config["environment"]["work_dir"]
    
    # 路径配置
    @property
    def base_path(self) -> str:
        return self.config["paths"]["base_path"]
    
    @property
    def prompt_file(self) -> str:
        return self.config["paths"]["prompt_file"]
    
    @property
    def output_dir(self) -> str:
        return self.config["paths"]["output_dir"]
    
    @property
    def log_dir(self) -> str:
        return self.config["paths"]["log_dir"]
    
    def get_category_path(self, category: str) -> str:
        """获取指定类别的路径"""
        return self.config["paths"]["categories"][category]
    
    @property
    def categories(self) -> Dict[str, str]:
        return self.config["paths"]["categories"]
    
    # 模型配置
    @property
    def model_base_path(self) -> str:
        return self.config["model"]["base_path"]
    
    @property
    def model_diffusion_files(self) -> List[str]:
        """获取完整的diffusion模型文件路径列表"""
        base_path = self.model_base_path
        files = self.config["model"]["diffusion_models"]
        return [os.path.join(base_path, f) for f in files]
    
    @property
    def model_t5_file(self) -> str:
        """获取T5模型文件完整路径"""
        return os.path.join(self.model_base_path, self.config["model"]["t5_model"])
    
    @property
    def model_vae_file(self) -> str:
        """获取VAE模型文件完整路径"""
        return os.path.join(self.model_base_path, self.config["model"]["vae_model"])
    
    # 实验配置
    @property
    def gpu_ids(self) -> List[int]:
        return self.config["experiment"]["gpu_ids"]
    
    @property
    def gpu_categories(self) -> Dict[str, str]:
        return self.config["experiment"]["gpu_categories"]
    
    def get_gpu_category(self, gpu_id: int) -> str:
        """获取指定GPU对应的类别"""
        return self.gpu_categories[str(gpu_id)]
    
    @property
    def seeds(self) -> List[int]:
        return self.config["experiment"]["seeds"]
    
    # 生成参数
    @property
    def generation_params(self) -> Dict[str, Any]:
        return self.config["experiment"]["generation_params"]
    
    @property
    def num_inference_steps(self) -> int:
        return self.generation_params["num_inference_steps"]
    
    @property
    def num_frames(self) -> int:
        return self.generation_params["num_frames"]
    
    @property
    def height(self) -> int:
        return self.generation_params["height"]
    
    @property
    def width(self) -> int:
        return self.generation_params["width"]
    
    @property
    def fps(self) -> int:
        return self.generation_params["fps"]
    
    @property
    def quality(self) -> int:
        return self.generation_params["quality"]
    
    @property
    def num_persistent(self) -> int:
        return self.generation_params["num_persistent"]
    
    @property
    def negative_prompt(self) -> str:
        return self.generation_params["negative_prompt"]
    
    def update_from_args(self, args: argparse.Namespace):
        """从命令行参数更新配置"""
        # 更新路径配置
        if hasattr(args, 'base_path') and args.base_path:
            self.config["paths"]["base_path"] = args.base_path
            # 同时更新相关路径
            self.config["paths"]["prompt_file"] = os.path.join(args.base_path, "prompt.json")
            for category in self.config["paths"]["categories"]:
                self.config["paths"]["categories"][category] = os.path.join(args.base_path, category)
        
        if hasattr(args, 'output_dir') and args.output_dir:
            self.config["paths"]["output_dir"] = args.output_dir
        
        if hasattr(args, 'ref_prompts_path') and args.ref_prompts_path:
            self.config["paths"]["prompt_file"] = args.ref_prompts_path
        
        # 更新模型路径
        if hasattr(args, 'model_path') and args.model_path:
            self.config["model"]["base_path"] = args.model_path
        
        # 更新环境配置
        if hasattr(args, 'conda_env') and args.conda_env:
            self.config["environment"]["conda_env"] = args.conda_env
        
        # 更新生成参数
        if hasattr(args, 'num_persistent') and args.num_persistent:
            self.config["experiment"]["generation_params"]["num_persistent"] = args.num_persistent
    
    def validate_paths(self) -> bool:
        """验证关键路径是否存在"""
        paths_to_check = [
            (self.base_path, "数据集基础路径"),
            (self.prompt_file, "prompt文件"),
            (self.model_base_path, "模型基础路径"),
        ]
        
        all_valid = True
        for path, description in paths_to_check:
            if not os.path.exists(path):
                print(f"错误: {description}不存在 - {path}")
                all_valid = False
        
        # 检查类别目录
        for category, path in self.categories.items():
            if not os.path.exists(path):
                print(f"警告: 类别目录不存在 - {category}: {path}")
        
        # 检查模型文件
        for model_file in self.model_diffusion_files:
            if not os.path.exists(model_file):
                print(f"错误: 模型文件不存在 - {model_file}")
                all_valid = False
        
        if not os.path.exists(self.model_t5_file):
            print(f"错误: T5模型文件不存在 - {self.model_t5_file}")
            all_valid = False
        
        if not os.path.exists(self.model_vae_file):
            print(f"错误: VAE模型文件不存在 - {self.model_vae_file}")
            all_valid = False
        
        return all_valid
    
    def create_directories(self):
        """创建必要的输出目录"""
        dirs_to_create = [
            self.output_dir,
            self.log_dir,
        ]
        
        # 为每个GPU创建日志目录
        for gpu_id in self.gpu_ids:
            dirs_to_create.append(os.path.join(self.log_dir, f"gpu_{gpu_id}"))
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
            print(f"创建目录: {dir_path}")


def load_config_with_args(config_path: str = None, args: argparse.Namespace = None) -> ExperimentConfig:
    """
    加载配置文件并应用命令行参数覆盖
    
    Args:
        config_path: 配置文件路径
        args: 命令行参数
    
    Returns:
        ExperimentConfig: 配置管理器实例
    """
    config = ExperimentConfig(config_path)
    
    if args:
        config.update_from_args(args)
    
    return config


def create_default_config(output_path: str = "config.json"):
    """创建默认配置文件"""
    default_config = {
        "environment": {
            "conda_path": "/home/Wind645/miniconda3/etc/profile.d/conda.sh",
            "conda_env": "diffsynth",
            "work_dir": "/home/Wind645/code/diffsynth4MT/ICLR批量实验脚本"
        },
        "paths": {
            "base_path": "/home/Wind645/code/diffsynth4MT/ICLR批量实验脚本/Final_Dataset",
            "prompt_file": "/home/Wind645/code/diffsynth4MT/ICLR批量实验脚本/Final_Dataset/prompt.json",
            "output_dir": "/home/Wind645/code/diffsynth4MT/ICLR批量实验脚本/results_final",
            "log_dir": "/home/Wind645/code/diffsynth4MT/ICLR批量实验脚本/logs",
            "categories": {
                "camera_motion": "/home/Wind645/code/diffsynth4MT/ICLR批量实验脚本/Final_Dataset/camera_motion",
                "single_object": "/home/Wind645/code/diffsynth4MT/ICLR批量实验脚本/Final_Dataset/single_object",
                "multiple_objects": "/home/Wind645/code/diffsynth4MT/ICLR批量实验脚本/Final_Dataset/multiple_objects",
                "complex_human_motion": "/home/Wind645/code/diffsynth4MT/ICLR批量实验脚本/Final_Dataset/complex_human_motion"
            }
        },
        "model": {
            "base_path": "/data/Wind645/Wan-AI/Wan2.1-T2V-14B",
            "diffusion_models": [
                "diffusion_pytorch_model-00001-of-00006.safetensors",
                "diffusion_pytorch_model-00002-of-00006.safetensors",
                "diffusion_pytorch_model-00003-of-00006.safetensors",
                "diffusion_pytorch_model-00004-of-00006.safetensors",
                "diffusion_pytorch_model-00005-of-00006.safetensors",
                "diffusion_pytorch_model-00006-of-00006.safetensors"
            ],
            "t5_model": "models_t5_umt5-xxl-enc-bf16.pth",
            "vae_model": "Wan2.1_VAE.pth"
        },
        "experiment": {
            "gpu_ids": [0, 1, 2, 3],
            "gpu_categories": {
                "0": "camera_motion",
                "1": "single_object",
                "2": "multiple_objects",
                "3": "complex_human_motion"
            },
            "seeds": [42, 123],
            "generation_params": {
                "num_inference_steps": 50,
                "num_frames": 41,
                "height": 480,
                "width": 832,
                "fps": 15,
                "quality": 5,
                "num_persistent": 50000000,
                "negative_prompt": "光线混乱，变形，扭曲，几何畸变，发光边缘，光晕，运动模糊过曝，静态，画面扭曲，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
            }
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, ensure_ascii=False, indent=2)
    
    print(f"已创建默认配置文件: {output_path}")


if __name__ == "__main__":
    # 测试配置加载
    parser = argparse.ArgumentParser(description="配置文件测试")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--create-default", action="store_true", help="创建默认配置文件")
    parser.add_argument("--validate", action="store_true", help="验证配置文件")
    
    args = parser.parse_args()
    
    if args.create_default:
        create_default_config(args.config)
    elif args.validate:
        config = ExperimentConfig(args.config)
        if config.validate_paths():
            print("配置验证通过!")
        else:
            print("配置验证失败!")
    else:
        config = ExperimentConfig(args.config)
        print("配置加载成功!")
        print(f"工作目录: {config.work_dir}")
        print(f"数据集路径: {config.base_path}")
        print(f"模型路径: {config.model_base_path}")
        print(f"输出目录: {config.output_dir}")
