import torch
import yaml
import argparse
import os
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from tqdm import tqdm

def generate_avatar(gender, output_name=None):
    # 1. Load configuration
    with open("configs/prompts.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Setup Pipeline
    print("Loading models (this may take a while)...")
    motion_adapter = MotionAdapter.from_pretrained(config['models']['motion_module'], torch_dtype=torch.float16)
    
    pipeline = AnimateDiffPipeline.from_pretrained(
        config['models']['base_model'],
        motion_adapter=motion_adapter,
        torch_dtype=torch.float16,
    ).to(device)

    # Use DDIM scheduler for better animation stability
    pipeline.scheduler = DDIMScheduler.from_config(
        pipeline.scheduler.config,
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )

    # Enable memory saving optimizations for RTX 3080
    pipeline.enable_vae_slicing()
    pipeline.enable_model_cpu_offload() # Switch to this if OOM occurs

    # 3. Prepare Prompts
    prompt_cfg = config['prompts'][gender]
    prompt = prompt_cfg['positive']
    negative_prompt = prompt_cfg['negative']

    # 4. Generate
    print(f"Generating {gender} avatar GIF...")
    output = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=config['settings']['num_frames'],
        guidance_scale=config['settings']['guidance_scale'],
        num_inference_steps=config['settings']['num_inference_steps'],
        generator=torch.manual_seed(42),
    )

    frames = output.frames[0]
    
    # 5. Optional Resize (e.g., to 268x268)
    resize_to = config['settings'].get('resize_to')
    if resize_to:
        print(f"Resizing frames to {resize_to}x{resize_to}...")
        from PIL import Image
        frames = [frame.resize((resize_to, resize_to), Image.LANCZOS) for frame in frames]
    
    # 6. Save
    if output_name is None:
        output_name = f"outputs/avatar_{gender}_hr_manager.gif"
    
    os.makedirs("outputs", exist_ok=True)
    export_to_gif(frames, output_name, fps=config['settings']['fps'])
    print(f"Successfully saved to {output_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AvatarGen: HR Dynamic Avatar Generator")
    parser.add_argument("--gender", type=str, choices=["male", "female"], default="female", help="Gender of the avatar")
    parser.add_argument("--output", type=str, default=None, help="Output filename")
    
    args = parser.parse_args()
    
    generate_avatar(args.gender, args.output)
