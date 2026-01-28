# AI-4-Creativity-Project-Template (25/26)



## Student name:Lucas Shrestha
## Student number: 2317991
## Project title: The Latent Cinematographer: Exploring Temporal Hallucinations
## Link to project video recording: https://drive.google.com/file/d/11aVo2CxuMfImaQxpCIF8s15tDCGHQkfn/view?usp=sharing

## 🎥 Project Concept
This project explores **"Temporal Incoherence"** in AI video generation. Unlike a traditional camera that captures physical continuity, this system leverages the **ZeroScope_v2** model to visualize the "Latent Walk"—a journey through the mathematical space of the model where objects morph and melt in a surreal "dream logic" aesthetic.

The goal is not stability, but **instability**. By hacking the model's attention layers, we create a "One-Shot Cinema" experience where the glitch is the primary feature.

---

## 🛠 Technical Implementation (LO1: Intervention)
To demonstrate technical depth beyond standard prompting, this project implements a custom Python pipeline that intervenes in the diffusion process:

### 1. Temporal Jitter Injection
Located in `glitch_cinema.py`, the `TemporalJitter` class registers forward hooks into the U-Net's **temporal attention layers**.
* **Mechanism:** It calculates the energy of the hidden states and injects calculated Gaussian noise during inference.
* **Result:** This forces the model to "hallucinate" temporal changes, creating the signature melting effect without destroying individual frame quality.

### 2. Apple Silicon Optimization
The pipeline is specifically optimized for MacBook Air M1 hardware:
* **MPS Backend:** Utilizes Metal Performance Shaders for GPU acceleration.
* **CPU Offloading:** Implements a custom `decode_on_cpu` function to handle VAE decoding, preventing memory overflows (OOM) typical on 8GB/16GB Unified Memory systems.

---

## 📊 Quantifying Instability (LO3: Testing & Evaluation)
Following instructor feedback, this project includes a robust analysis suite (`complete_analysis.py`) to provide objective evidence of the generated effects.

We measure instability using two distinct metrics:
1.  **Optical Flow (Farneback Method):** Tracks the magnitude of movement vectors between adjacent frames to measure "fluidity."
2.  **Pixel Difference (MSE):** Calculates the Mean Squared Error between frames to quantify raw visual change.

**Validation:**
The system generates a correlation graph mapping the **Noise Injection Level** (0.0 - 0.5) against the **Combined Instability Score**, proving a direct mathematical relationship between the code intervention and the visual output.

---

## 🚀 Installation & Usage

### Prerequisites
* Python 3.10+
* FFmpeg (for video processing)
* MacBook with M1/M2/M3 chip (Recommended)

### 1. Setup

# 1. Clone the repository
```
git clone https://github.com/Lucas-ctrl1/latent-cinema.git
cd latent-cinema
```

# 2. Create environment
```
conda create -n latent_cinema python=3.10
conda activate latent_cinema
```
# 3. Install dependencies
```
pip install -r requirements.txt
``` 
### 2. Generate "Glitch" Cinema
Run the main generator script. You will be prompted to select an instability level (0.0 to 0.5).
``` 
python glitch_cinema.py
``` 
Output: Videos are saved to the output/ folder (e.g., output/glitch_s0.2.mp4).

### 3. Analyze Results
Run the analysis suite to generate charts and quantify the effect.
``` 
python complete_analysis.py
``` 
### File Structure
``` 
latent-cinema/
├── README.md                 # Project documentation and instructions
├── requirements.txt          # List of dependencies (Torch, Diffusers, OpenCV)
├── glitch_cinema.py          # LO1: The Generator (Main code with Temporal Jitter)
├── complete_analysis.py      # LO3: The Analyst (Optical Flow & MSE scoring)
├── .gitignore                # Tells Git to ignore large video files
└── output/                   # Auto-generated folder for results
    ├── glitch_s0.0.mp4       # Stable reference video
    ├── glitch_s0.2.mp4       # The "Latent Film" (Metamorphosis)
    ├── instability_comparison.png  # Bar chart proving instability
    └── correlation_analysis.png    # Graph showing noise vs. glitch level

``` 

