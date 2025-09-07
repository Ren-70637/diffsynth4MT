import torch
import os
import re
import argparse
import json
import time
import logging
import random
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData

def setup_logging(gpu_id, category, log_base_dir=None):
    """设置日志配置"""
    if log_base_dir is None:
        # 尝试从环境变量获取，否则使用当前工作目录
        log_base_dir = os.environ.get('EXPERIMENT_LOG_DIR', os.path.join(os.getcwd(), 'logs'))
    
    log_dir = os.path.join(log_base_dir, f"gpu_{gpu_id}")
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"{log_dir}/incremental_{category}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - GPU{} - %(levelname)s - %(message)s'.format(gpu_id),
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_next_video_path(output_dir="results", prefix="video", ext=".mp4"):
    """获取下一个可用的视频文件路径"""
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(ext)}$")
    nums = []
    for fn in os.listdir(output_dir):
        m = pattern.match(fn)
        if m:
            nums.append(int(m.group(1)))
    next_num = max(nums) + 1 if nums else 1
    return os.path.join(output_dir, f"{prefix}{next_num}{ext}")

def check_if_completed(output_dir, video_key, prompt_idx):
    """检查任务是否已完成(检查是否存在该prompt的任意2个seed结果)"""
    import glob
    pattern = f"{video_key}_prompt{prompt_idx+1}_seed*_video*.mp4"
    existing_files = glob.glob(os.path.join(output_dir, pattern))
    # 如果已存在2个或以上该prompt的结果，认为已完成
    return len(existing_files) >= 2

def get_existing_results(output_base_dir, category):
    """获取已存在的实验结果"""
    existing_results = set()
    category_dir = os.path.join(output_base_dir, category)
    
    if not os.path.exists(category_dir):
        return existing_results
    
    for video_dir in os.listdir(category_dir):
        video_path = os.path.join(category_dir, video_dir)
        if os.path.isdir(video_path):
            for prompt_dir in os.listdir(video_path):
                prompt_path = os.path.join(video_path, prompt_dir)
                if os.path.isdir(prompt_path):
                    # 检查是否有完成的视频文件
                    if any(f.endswith('.mp4') for f in os.listdir(prompt_path)):
                        result_key = f"{video_dir}_{prompt_dir}"
                        existing_results.add(result_key)
    
    return existing_results

def setup_gpu_device(gpu_id):
    """设置指定的GPU设备"""
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        logger.info(f"使用GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        logger.info(f"GPU {gpu_id} 显存: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.1f} GB")
        return device
    else:
        logger.error("CUDA不可用")
        return "cpu"

def load_model_with_retry(model_base_dir=None, max_retries=3):
    """带重试的模型加载"""
    
    # 获取模型基础路径
    if model_base_dir is None:
        model_base_dir = os.environ.get('MODEL_BASE_DIR', '/root/autodl-tmp/pretrained_models')
    
    model_dir = os.path.join(model_base_dir, "Wan-AI/Wan2.1-T2V-14B")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"尝试加载模型 (第{attempt + 1}次)")
            logger.info(f"模型路径: {model_dir}")
            
            model_manager = ModelManager(device="cpu")
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
            
            pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
            logger.info("模型加载成功")
            return pipe
            
        except Exception as e:
            logger.error(f"模型加载失败 (第{attempt + 1}次): {str(e)}")
            if attempt < max_retries - 1:
                logger.info("等待10秒后重试...")
                time.sleep(10)
            else:
                logger.error("模型加载最终失败")
                raise

def main(args):
    global logger
    logger = setup_logging(args.gpu_id, args.category, args.log_base_dir)
    
    logger.info("="*50)
    logger.info("开始增量视频生成实验")
    logger.info(f"GPU ID: {args.gpu_id}")
    logger.info(f"处理类别: {args.category}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"增量模式: {args.incremental_mode}")
    logger.info("="*50)
    
    # 设置GPU
    device = setup_gpu_device(args.gpu_id)
    
    # 读取prompts
    try:
        with open(args.ref_prompts_path, 'r', encoding='utf-8') as f:
            ref_prompts = json.load(f)
        logger.info(f"成功读取prompts文件: {args.ref_prompts_path}")
    except Exception as e:
        logger.error(f"读取prompts文件失败: {e}")
        return
    
    # 获取已存在的结果（如果是增量模式）
    existing_results = set()
    if args.incremental_mode:
        existing_results = get_existing_results(args.output_dir, args.category)
        logger.info(f"检测到已存在的结果数量: {len(existing_results)}")
    
    # 加载模型
    pipe = load_model_with_retry()
    
    # 启用VRAM管理
    logger.info(f"启用VRAM管理，persistent参数: {args.num_persistent}")
    pipe.enable_vram_management(num_persistent_param_in_dit=args.num_persistent)
    
    # 处理指定类别
    if args.category not in ref_prompts:
        logger.error(f"类别 {args.category} 在prompts中不存在")
        return
    
    videos = ref_prompts[args.category]
    logger.info(f"开始处理类别: {args.category}")
    logger.info(f"视频数量: {len(videos)}")
    
    # 计算总任务数
    total_tasks = 0
    new_tasks = 0
    for video_key, prompts in videos.items():
        for i in range(len(prompts)):
            total_tasks += 2  # 两个不同的seed
            
            # 检查是否为新任务(随机种子模式下检查是否已有结果)
            if args.incremental_mode and not check_if_completed(args.output_dir, video_key, i):
                new_tasks += 2
            elif not args.incremental_mode:
                new_tasks += 2
    
    logger.info(f"总任务数: {total_tasks}")
    logger.info(f"新任务数: {new_tasks}")
    
    if new_tasks == 0:
        logger.info("没有新任务需要处理")
        return
    
    # 处理视频
    processed_count = 0
    # 生成随机种子
    seeds = [random.randint(1, 10000), random.randint(1, 10000)]
    
    for video_key, prompts in videos.items():
        # 根据category确定视频文件路径
        input_video_path = os.path.join(args.base_path, video_key + ".mp4")
        
        logger.info(f"处理视频: {input_video_path}")
        if not os.path.exists(input_video_path):
            logger.warning(f"视频文件不存在: {input_video_path}")
            continue
        
        try:
            video = VideoData(input_video_path, height=480, width=832)
            logger.info(f"成功加载视频: {video_key}")
        except Exception as e:
            logger.error(f"加载视频失败 {video_key}: {e}")
            continue
        
        for i in range(len(prompts)):
            prompt = prompts[i]
            
            # 检查是否需要跳过(随机种子模式下检查是否已完成)
            if args.incremental_mode and check_if_completed(args.output_dir, video_key, i):
                logger.info(f"跳过已完成的任务: {video_key}_prompt{i+1}")
                continue
            
            for seed in seeds:
                try:
                    logger.info(f"生成视频: {video_key} - prompt{i+1} - seed{seed}")
                    logger.info(f"Prompt: {prompt[:100]}...")
                    
                    # 创建输出路径
                    output_path = os.path.join(args.output_dir, f"{video_key}_prompt{i+1}_seed{seed}_video.mp4")
                    os.makedirs(args.output_dir, exist_ok=True)
                    
                    video_result = pipe(
                        prompt=prompt,
                        negative_prompt="deformation, distortion, geometric aberration, glowing edges, halo, motion blur, whitening, glare, halo, lens flare, glowing edges, sharpening artifacts, overexposure, static, warped image, unclear details, subtitles, style, work, painting, still frame, overall grayish tone, worst quality, low quality, JPEG compression artifacts, ugly, defective, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, completely still image, cluttered background, three legs, crowded background, walking backwards",
                        num_inference_steps=50,
                        input_video=video,
                        seed=seed,
                        tiled=True,
                        num_frames=81,  # 使用81帧
                    )
                    
                    save_video(video_result, output_path, fps=15, quality=5)
                    
                    processed_count += 1
                    progress = (processed_count / new_tasks) * 100
                    logger.info(f"成功生成 {video_key}-prompt{i+1}-seed{seed} | 进度: {progress:.1f}% ({processed_count}/{new_tasks})")
                    
                except Exception as e:
                    logger.error(f"生成失败 {video_key}-prompt{i+1}-seed{seed}: {e}")
                    continue
    
    logger.info("="*50)
    logger.info(f"增量实验完成! 总共处理 {processed_count} 个任务")
    logger.info("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WanVideo 增量实验")
    parser.add_argument("--gpu_id", type=int, required=True, help="使用的GPU ID")
    parser.add_argument("--category", type=str, required=True, help="要处理的类别")
    parser.add_argument("--output_dir", type=str, default=os.environ.get('EXPERIMENT_OUTPUT_DIR', './results_final_2'), help="输出目录")
    parser.add_argument("--ref_prompts_path", type=str, required=True, help="prompts文件路径")
    parser.add_argument("--base_path", type=str, required=True, help="视频文件基础路径")
    parser.add_argument("--num_persistent", type=int, default=50000000, help="VRAM管理参数")
    parser.add_argument("--incremental_mode", action="store_true", help="启用增量模式（跳过已存在的结果）")
    # 在参数解析部分添加
    parser.add_argument("--model_base_dir", type=str, 
                    default=os.environ.get('MODEL_BASE_DIR', '/root/autodl-tmp/pretrained_models'),
                    help="模型文件基础目录")
    parser.add_argument("--log_base_dir", type=str,
                    default=os.environ.get('EXPERIMENT_LOG_DIR', './logs'),
                    help="日志文件基础目录")
    args = parser.parse_args()
    main(args)
