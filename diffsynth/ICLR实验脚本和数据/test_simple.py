#!/usr/bin/env python3
"""
简化的测试脚本 - 用于诊断问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import multiprocessing
import argparse
from load_config import ExperimentConfig

def test_config():
    """测试配置加载"""
    print("1. 测试配置加载...")
    config = ExperimentConfig("config.json")
    print(f"   配置加载成功: {config.work_dir}")
    return config.config

def test_gpu():
    """测试GPU"""
    print("2. 测试GPU...")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"   发现 {gpu_count} 个GPU")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"   GPU {i}: {props.name} - {memory_gb:.1f}GB")
        return True
    else:
        print("   错误: CUDA不可用")
        return False

def simple_worker(gpu_id):
    """简单的工作进程"""
    try:
        print(f"工作进程启动 - GPU {gpu_id}")
        
        # 尝试设置GPU
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        print(f"成功设置设备: {device}")
        
        # 简单的tensor操作测试
        x = torch.randn(100, 100).to(device)
        y = torch.matmul(x, x.t())
        print(f"GPU {gpu_id} 张量操作成功")
        
        return {"gpu_id": gpu_id, "success": True}
        
    except Exception as e:
        print(f"GPU {gpu_id} 工作进程失败: {str(e)}")
        return {"gpu_id": gpu_id, "success": False, "error": str(e)}

def test_multiprocessing():
    """测试多进程"""
    print("3. 测试多进程...")
    
    # 设置spawn方法
    multiprocessing.set_start_method('spawn', force=True)
    
    try:
        with multiprocessing.Pool(processes=2) as pool:
            results = pool.map(simple_worker, [0, 1])
        
        print("   多进程测试结果:")
        for result in results:
            print(f"   {result}")
        
        return all(r.get('success', False) for r in results)
        
    except Exception as e:
        print(f"   多进程测试失败: {str(e)}")
        return False

def test_imports():
    """测试关键模块导入"""
    print("4. 测试关键模块导入...")
    
    try:
        from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
        print("   diffsynth 模块导入成功")
        return True
    except Exception as e:
        print(f"   diffsynth 模块导入失败: {str(e)}")
        return False

def main():
    print("=== 简化测试脚本 ===")
    
    # 测试步骤
    tests = [
        ("配置加载", test_config),
        ("GPU可用性", test_gpu),
        ("关键模块", test_imports),
        ("多进程", test_multiprocessing),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = result
            status = "✅ 通过" if result else "❌ 失败"
            print(f"   {status}")
        except Exception as e:
            results[name] = False
            print(f"   ❌ 异常: {str(e)}")
        print()
    
    print("=== 测试总结 ===")
    for name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    if all(results.values()):
        print("\n🎉 所有测试通过！系统准备就绪。")
        return 0
    else:
        print("\n⚠️  部分测试失败，需要修复问题。")
        return 1

if __name__ == "__main__":
    exit(main())


