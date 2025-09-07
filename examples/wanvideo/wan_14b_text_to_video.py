import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download
import os, re, argparse


# Download models
#snapshot_download("Wan-AI/Wan2.1-T2V-14B", local_dir="models/Wan-AI/Wan2.1-T2V-14B")
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
    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
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
        ],
        #torch_dtype=torch.float8_e4m3fn, # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

    video = VideoData("/root/autodl-tmp/ref_单个豹步_832x480.mp4", height=480, width=832)
    # Text-to-video
    video = pipe(
        prompt="Documentary photography style. An African elephant walking steadily from right to left, trunk swinging slowly, in front of towering red sand dunes and scattered desert shrubs, under clear desert sunlight, realistic documentary style, tracking shot from behind side angle.",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        num_inference_steps=50,
        input_video=video,
        seed=args.seed, 
        tiled=True,
        num_frames=81,
        sf=args.sf,
        test_latency=args.test_latency,
        latency_dir=args.latency_dir,
        mode=args.mode,
    )
    save_video(video, get_next_video_path(output_dir=args.output_dir), fps=15, quality=5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WanVideo Text-to-Video Example")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save output videos")
    #parser.add_argument("--prompt", type=str, help="Text prompt for video generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--sf", type=int, default=4, help="Spatial factor for AMF computation (default: 1)")
    parser.add_argument("--test_latency", action="store_true", help="Test latency of the model")
    parser.add_argument("--latency_dir", type=str, default=None, help="Directory to save latency logs")
    parser.add_argument("--mode", type=str, default=None, choices=['No_transfer', 'AMF', 'effi_AMF'],help="Mode for the video generation, e.g., 'No_transfer'")
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    main(args)






