#!/usr/bin/env python3
"""
显存使用监控工具
用于分析WAN2.1-T2V-14B模型的显存使用组成和峰值
"""

import torch
import os
import time
import json
import argparse
import psutil
import subprocess
from datetime import datetime
from diffsynth import ModelManager, WanVideoPipeline, VideoData

def get_gpu_memory_info(gpu_id=0):
    """获取GPU显存信息"""
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        # PyTorch显存信息
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3   # GB
        max_allocated = torch.cuda.max_memory_allocated(gpu_id) / 1024**3  # GB
        
        # nvidia-smi显存信息
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                                   '--format=csv,noheader,nounits', '-i', str(gpu_id)], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                used_mb, total_mb = map(int, result.stdout.strip().split(', '))
                nvidia_used_gb = used_mb / 1024
                nvidia_total_gb = total_mb / 1024
            else:
                nvidia_used_gb = nvidia_total_gb = 0
        except:
            nvidia_used_gb = nvidia_total_gb = 0
        
        return {
            'torch_allocated': allocated,
            'torch_reserved': reserved,
            'torch_max_allocated': max_allocated,
            'nvidia_used': nvidia_used_gb,
            'nvidia_total': nvidia_total_gb,
            'nvidia_free': nvidia_total_gb - nvidia_used_gb
        }
    return None

def log_memory_state(stage, gpu_id=0, log_file=None):
    """记录当前显存状态"""
    memory_info = get_gpu_memory_info(gpu_id)
    if memory_info:
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = {
            'timestamp': timestamp,
            'stage': stage,
            'gpu_id': gpu_id,
            **memory_info
        }
        
        print(f"[{timestamp}] {stage}:")
        print(f"  PyTorch - 已分配: {memory_info['torch_allocated']:.2f}GB, "
              f"已保留: {memory_info['torch_reserved']:.2f}GB, "
              f"峰值: {memory_info['torch_max_allocated']:.2f}GB")
        print(f"  NVIDIA  - 已使用: {memory_info['nvidia_used']:.2f}GB, "
              f"总计: {memory_info['nvidia_total']:.2f}GB, "
              f"空闲: {memory_info['nvidia_free']:.2f}GB")
        print("-" * 60)
        
        if log_file:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        return log_entry
    return None

def analyze_model_components(gpu_id=0, log_file=None):
    """分析模型各组件的显存使用"""
    
    print("🔍 开始显存使用分析...")
    print("=" * 80)
    
    # 重置显存统计
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(gpu_id)
    
    # 基准状态
    baseline = log_memory_state("01_基准状态(空载)", gpu_id, log_file)
    
    try:
        # 1. 创建ModelManager
        print("📦 加载ModelManager...")
        model_manager = ModelManager(device="cpu")
        log_memory_state("02_ModelManager创建", gpu_id, log_file)
        
        # 2. 加载模型文件
        print("📂 加载模型文件...")
        model_base_dir = os.environ.get('MODEL_BASE_DIR', '/root/autodl-tmp/pretrained_models')
        model_dir = os.path.join(model_base_dir, "Wan-AI/Wan2.1-T2V-14B")
        
        model_manager.load_models(
            [
                [
                    os.path.join(model_dir, "diffusion_pytorch_model-00001-of-00006.safetensors"),
                    os.path.join(model_dir, "diffusion_pytorch_model-00002-of-00006.safetensors"),
                    os.path.join(model_dir, "diffusion_pytorch_model-00003-of-00006.safetensors"),
                    os.path.join(model_dir, "diffusion_pytorch_model-00004-of-00006.safetensors"),
                    os.path.join(model_dir, "diffusion_pytorch_model-00005-of-00006.safetensors"),
                    os.path.join(model_dir, "diffusion_pytorch_model-00006-of-00006.safetensors"),
                ],
                os.path.join(model_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
                os.path.join(model_dir, "Wan2.1_VAE.pth"),
            ],
            torch_dtype=torch.bfloat16,
        )
        log_memory_state("03_模型文件加载完成", gpu_id, log_file)
        
        # 3. 创建Pipeline
        print("🔧 创建Pipeline...")
        pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=f"cuda:{gpu_id}")
        log_memory_state("04_Pipeline创建完成", gpu_id, log_file)
        
        # 4. 启用VRAM管理
        print("💾 启用VRAM管理...")
        pipe.enable_vram_management(num_persistent_param_in_dit=50000000)
        log_memory_state("05_VRAM管理启用", gpu_id, log_file)
        
        # 5. 加载测试视频
        print("🎬 加载测试视频...")
        video_path = "/root/autodl-tmp/Final_Dataset/camera_motion/ref_镜头仰角旋转_832x480.mp4"
        if os.path.exists(video_path):
            video = VideoData(video_path, height=480, width=832)
            log_memory_state("06_视频加载完成", gpu_id, log_file)
        else:
            print(f"⚠️ 测试视频不存在: {video_path}")
            return
        
        # 6. 执行推理(小规模测试)
        print("🚀 执行推理测试...")
        test_prompt = "A simple test prompt for memory analysis"
        
        # 推理前显存状态
        log_memory_state("07_推理前状态", gpu_id, log_file)
        
        # 执行推理 - 先尝试81帧，如果失败则尝试其他帧数
        test_frames = [81, 85, 77, 73]  # 尝试不同的帧数来找到合适的值
        
        for frames in test_frames:
            try:
                print(f"  尝试 {frames} 帧...")
                video_result = pipe(
                    prompt=test_prompt,
                    negative_prompt="low quality, blurred",
                    num_inference_steps=10,  # 减少步数用于测试
                    input_video=video,
                    seed=42,
                    tiled=True,
                    num_frames=frames,
                )
                print(f"  ✅ {frames} 帧推理成功")
                break
            except Exception as e:
                print(f"  ❌ {frames} 帧失败: {str(e)[:100]}...")
                if frames == test_frames[-1]:  # 最后一次尝试
                    raise e
                continue
        
        # 推理中峰值显存
        log_memory_state("08_推理完成(峰值)", gpu_id, log_file)
        
        # 7. 清理缓存后
        print("🧹 清理GPU缓存...")
        torch.cuda.empty_cache()
        log_memory_state("09_缓存清理后", gpu_id, log_file)
        
        return True
        
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        log_memory_state("ERROR_错误状态", gpu_id, log_file)
        return False

def calculate_memory_breakdown(log_file):
    """计算显存使用分解"""
    if not os.path.exists(log_file):
        return
    
    print("\n📊 显存使用分解分析")
    print("=" * 80)
    
    entries = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                entries.append(json.loads(line.strip()))
            except:
                continue
    
    if len(entries) < 2:
        print("数据不足，无法进行分解分析")
        return
    
    baseline = entries[0]
    peak = max(entries, key=lambda x: x['torch_allocated'])
    
    print(f"基准显存使用: {baseline['nvidia_used']:.2f}GB")
    print(f"峰值显存使用: {peak['nvidia_used']:.2f}GB")
    print(f"模型总占用: {peak['nvidia_used'] - baseline['nvidia_used']:.2f}GB")
    
    # 计算各阶段增量
    prev_used = baseline['nvidia_used']
    print(f"\n各阶段显存增量:")
    
    stage_mapping = {
        "02_ModelManager创建": "ModelManager",
        "03_模型文件加载完成": "模型权重",
        "04_Pipeline创建完成": "Pipeline初始化",
        "05_VRAM管理启用": "VRAM管理",
        "06_视频加载完成": "视频数据",
        "08_推理完成(峰值)": "推理过程"
    }
    
    for entry in entries[1:]:
        stage = entry['stage']
        if stage in stage_mapping:
            increment = entry['nvidia_used'] - prev_used
            print(f"  {stage_mapping[stage]:12}: +{increment:6.2f}GB (总计: {entry['nvidia_used']:6.2f}GB)")
            prev_used = entry['nvidia_used']
    
    # 估算并行能力
    print(f"\n🎯 并行能力估算:")
    total_memory = entries[0]['nvidia_total']
    model_memory = peak['nvidia_used'] - baseline['nvidia_used']
    
    # 考虑一些内存开销和安全边际
    safe_margin = 2.0  # 2GB安全边际
    available_memory = total_memory - safe_margin
    
    max_parallel = int(available_memory / model_memory)
    print(f"  GPU总显存: {total_memory:.1f}GB")
    print(f"  单模型占用: {model_memory:.1f}GB")
    print(f"  安全边际: {safe_margin:.1f}GB")
    print(f"  理论最大并行数: {max_parallel}")
    
    return {
        'total_memory': total_memory,
        'model_memory': model_memory,
        'max_parallel': max_parallel,
        'baseline_memory': baseline['nvidia_used'],
        'peak_memory': peak['nvidia_used']
    }

def main():
    parser = argparse.ArgumentParser(description="WAN2.1-T2V-14B显存使用分析")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU设备ID")
    parser.add_argument("--output", type=str, default="vram_analysis.jsonl", help="输出日志文件")
    parser.add_argument("--summary", type=str, default="vram_summary.json", help="摘要输出文件")
    
    args = parser.parse_args()
    
    # 清空之前的日志
    if os.path.exists(args.output):
        os.remove(args.output)
    
    # 执行分析
    success = analyze_model_components(args.gpu_id, args.output)
    
    if success:
        # 分解分析
        summary = calculate_memory_breakdown(args.output)
        
        # 保存摘要
        if summary:
            with open(args.summary, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n💾 分析结果已保存到: {args.summary}")
        
        print(f"\n📋 详细日志文件: {args.output}")
        print("\n✅ 显存分析完成!")
    else:
        print("\n❌ 显存分析失败!")

if __name__ == "__main__":
    main()