import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
import os, re
import argparse
import json


def get_next_video_path(output_dir="results", prefix="video", ext=".mp4"):
    """
    在 output_dir 下查找所有 prefixN.ext，返回下一个可用的完整路径。
    例如已有 video1.mp4、video2.mp4，则返回 results/video3.mp4
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

def main(args):
    # Load 14B models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            [
                "/mnt/data1/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
                "/mnt/data1/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
                "/mnt/data1/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
                "/mnt/data1/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
                "/mnt/data1/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
                "/mnt/data1/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
            ],
            "/mnt/data1/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
            "/mnt/data1/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16,
    )
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    
    # 启用 VRAM 管理，14B 模型建议设置较小的 num_persistent_param_in_dit
    # 可以根据你的显存情况调整（0-1e8 范围内）
    pipe.enable_vram_management(num_persistent_param_in_dit=args.num_persistent if args.num_persistent else 50000000)
    
    #ref_prompts_path = "/home/Wind645/code/diffsynth4MT/41_ref/prompts.json"
    #base = "/home/Wind645/code/diffsynth4MT/41_ref"
    ref_prompts_path = args.ref_prompts_path
    base = args.base_path
    
    with open(ref_prompts_path, 'r', encoding='utf-8') as f:
        ref_prompts = json.load(f)
    
    for key, value in ref_prompts.items():
        input_video_path = os.path.join(base, key + ".mp4")
        video = VideoData(input_video_path, height=480, width=832)
        
        for i in range(5):
            prompt = value[i] if isinstance(value, list) else value
            output_dir = os.path.join(args.output_dir, key, "prompt"+str(i+1))
            
            video_result = pipe(
                prompt=prompt,
                negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                num_inference_steps=50,
                input_video=video,
                seed=args.seed, 
                tiled=True,
                num_frames=41,
                sf=args.sf,
                test_latency=args.test_latency,
                latency_dir=args.latency_dir,
                mode=args.mode,
            )
            save_video(video_result, get_next_video_path(output_dir=output_dir), fps=15, quality=5)
            
            print(f"Generated video for {key} - prompt {i+1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WanVideo 14B Text-to-Video DAVIS Sampling")
    parser.add_argument("--output_dir", type=str, default="results_14b", help="Directory to save output videos")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--sf", type=int, default=4, help="Spatial factor for AMF computation (default: 4)")
    parser.add_argument("--test_latency", action="store_true", help="Test latency of the model")
    parser.add_argument("--latency_dir", type=str, default=None, help="Directory to save latency logs")
    parser.add_argument("--mode", type=str, default="effi_AMF", choices=['No_transfer', 'AMF', 'effi_AMF'], 
                       help="Mode for the video generation (default: effi_AMF)")
    parser.add_argument("--num_persistent", type=int, default=None, 
                       help="Number of persistent parameters in DiT (for VRAM management)")
    parser.add_argument("--ref_prompts_path", type=str, default="/home/Wind645/code/diffsynth4MT/41_ref/prompts.json",
                        help="Path to the reference prompts JSON file")
    parser.add_argument("--base_path", type=str, default="/home/Wind645/code/diffsynth4MT/41_ref",
                        help="Base path for input videos")
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    main(args)