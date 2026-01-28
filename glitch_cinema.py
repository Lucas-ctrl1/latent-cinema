import torch
import torch.nn as nn
from diffusers import TextToVideoSDPipeline
from diffusers.utils import export_to_video
import os
import warnings

warnings.filterwarnings("ignore")

# --- LO1: TEMPORAL HACK ---
class TemporalJitter:
    def __init__(self, intensity=0.1):
        self.intensity = intensity

    def hook_fn(self, module, input, output):
        """Handle both tensor and tuple outputs"""
        if self.intensity > 0:
            if isinstance(output, tuple):
                # Usually (hidden_states, attention_weights)
                if len(output) > 0 and isinstance(output[0], torch.Tensor):
                    hidden_states = output[0]
                    energy = torch.norm(hidden_states) / torch.sqrt(torch.tensor(hidden_states.numel()))
                    noise = torch.randn_like(hidden_states) * (self.intensity * energy)
                    return (hidden_states + noise,) + output[1:]
                return output
            elif isinstance(output, torch.Tensor):
                energy = torch.norm(output) / torch.sqrt(torch.tensor(output.numel()))
                noise = torch.randn_like(output) * (self.intensity * energy)
                return output + noise
        return output

# --- M2 MEMORY FIX ---
def decode_on_cpu(pipe, latents):
    print("--- Moving to CPU for safe decoding ---")
    
    # Move VAE to CPU
    pipe.vae = pipe.vae.to("cpu")
    latents = latents.to("cpu")
    
    # Scale latents
    latents = 1 / pipe.vae.config.scaling_factor * latents
    
    # Reshape for decoding
    batch, channel, frames, height, width = latents.shape
    latents = latents.permute(0, 2, 1, 3, 4).reshape(batch * frames, channel, height, width)
    
    # Decode in chunks
    decoded_frames = []
    chunk_size = 2  # Even smaller chunks for safety
    
    with torch.no_grad():
        for i in range(0, latents.shape[0], chunk_size):
            chunk = latents[i:i+chunk_size]
            decoded = pipe.vae.decode(chunk).sample
            decoded_frames.append(decoded)
    
    # Combine
    image = torch.cat(decoded_frames, dim=0)
    
    # Reshape back
    video = image.reshape(batch, frames, 3, image.shape[2], image.shape[3])
    video = video.permute(0, 2, 1, 3, 4)
    
    # Normalize
    video = (video / 2 + 0.5).clamp(0, 1)
    video = video.cpu().permute(0, 2, 3, 4, 1).numpy()
    
    return video

def generate_glitch_film():
    print("=== LATENT CINEMATOGRAPHER v3 (Fixed Hooks) ===")
    
    device = "cpu"
    model_id = "cerspense/zeroscope_v2_576w" 
    
    print(f"Loading {model_id} on {device.upper()}...")
    
    pipe = TextToVideoSDPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    
    # User Control
    print("\nSelect Instability Level:")
    print("0.0 = Stable (Reference)")
    print("0.2 = Metamorphosis (The Goal)")
    print("0.5 = Chaos (Extreme)")
    
    try:
        val = input("Enter value (Default 0.0): ")
        jitter_val = float(val) if val.strip() else 0.0
    except ValueError:
        jitter_val = 0.0
    
    # Register hooks only on TEMPORAL attention layers
    jitter = TemporalJitter(intensity=jitter_val)
    hooks = []
    count = 0
    
    # More targeted hooking - only temporal attention
    for name, module in pipe.unet.named_modules():
        if "temp" in name.lower() and "attn" in name.lower():
            hook = module.register_forward_hook(jitter.hook_fn)
            hooks.append(hook)
            count += 1
    
    if count == 0:
        print("No temporal layers found. Hooking all attention layers...")
        for name, module in pipe.unet.named_modules():
            if "attn" in name.lower():
                hook = module.register_forward_hook(jitter.hook_fn)
                hooks.append(hook)
                count += 1
                if count >= 10:  # Limit to 10 hooks
                    break
    
    print(f"→ Injected instability into {count} layers.")

    # Generate
    prompt = "A cyberpunk city at night, cinematic lighting, 4k"
    print(f"\nGenerating: '{prompt}'...")
    print("(Using CPU - this will take 15-20 minutes)")
    
    try:
        with torch.no_grad():
            latents = pipe(
                prompt=prompt,
                num_frames=8,
                height=256,
                width=384,
                num_inference_steps=20,
                guidance_scale=7.5,
                output_type="latent"
            ).frames
        
        # Decode and save
        video_frames = decode_on_cpu(pipe, latents)
        
        os.makedirs("output", exist_ok=True)
        filename = f"output/glitch_s{jitter_val:.1f}.mp4"
        export_to_video(video_frames[0], filename, fps=6)
        
        print(f"\n✅ SUCCESS: Created {filename}")
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        print("Trying with even simpler settings...")
        
        # Emergency fallback
        try:
            with torch.no_grad():
                latents = pipe(
                    prompt="cyberpunk city",
                    num_frames=4,  # Even smaller
                    height=192,
                    width=256,
                    num_inference_steps=15,
                    guidance_scale=7.0,
                    output_type="latent"
                ).frames
            
            video_frames = decode_on_cpu(pipe, latents)
            filename = f"output/fallback_s{jitter_val:.1f}.mp4"
            export_to_video(video_frames[0], filename, fps=4)
            print(f"✅ Created fallback: {filename}")
            
        except Exception as e2:
            print(f"❌ Could not generate video: {e2}")
    
    finally:
        # Cleanup hooks
        for hook in hooks:
            hook.remove()
    
    return None

if __name__ == "__main__":
    generate_glitch_film()