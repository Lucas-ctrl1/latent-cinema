# complete_analysis.py - WORKING VERSION
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_optical_flow(video_path):
    """Calculate optical flow instability using Farneback method"""
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        return None
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flow_scores = []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= 30:  # Limit to 30 frames for speed
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate magnitude
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_scores.append(np.mean(magnitude))
        
        prev_gray = gray
        frame_count += 1
    
    cap.release()
    
    if len(flow_scores) == 0:
        return None
    
    return {
        'avg_flow': np.mean(flow_scores),
        'std_flow': np.std(flow_scores),
        'max_flow': np.max(flow_scores)
    }

def calculate_pixel_difference(video_path):
    """Calculate frame-to-frame pixel differences"""
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    pixel_scores = []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= 30:  # Limit to 30 frames
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            pixel_scores.append(np.mean(diff))
        
        prev_frame = gray
        frame_count += 1
    
    cap.release()
    
    if len(pixel_scores) == 0:
        return None
    
    return {
        'avg_pixel': np.mean(pixel_scores),
        'std_pixel': np.std(pixel_scores),
        'max_pixel': np.max(pixel_scores)
    }

def analyze_video(video_path):
    """Complete analysis of a single video"""
    print(f"  Analyzing: {os.path.basename(video_path)}")
    
    # Calculate both metrics
    flow_data = calculate_optical_flow(video_path)
    pixel_data = calculate_pixel_difference(video_path)
    
    if not flow_data or not pixel_data:
        print(f"    ❌ Could not analyze video")
        return None
    
    # Combined score
    combined = (flow_data['avg_flow'] * 0.6 + pixel_data['avg_pixel'] * 0.4)
    
    # Extract injection level from filename
    try:
        filename = os.path.basename(video_path)
        level = float(filename.split('_s')[1].replace('.mp4', ''))
    except:
        level = 0.0
    
    return {
        'filename': filename,
        'level': level,
        'flow_avg': flow_data['avg_flow'],
        'pixel_avg': pixel_data['avg_pixel'],
        'combined': combined,
        'details': {
            'flow_std': flow_data['std_flow'],
            'flow_max': flow_data['max_flow'],
            'pixel_std': pixel_data['std_pixel'],
            'pixel_max': pixel_data['max_pixel']
        }
    }

def main():
    print("=== TEMPORAL INSTABILITY QUANTIFICATION ===")
    print("Addressing Tutor Feedback with Optical Flow Analysis\n")
    
    # Look for generated videos
    files = [
        "output/glitch_s0.0.mp4",
        "output/glitch_s0.2.mp4", 
        "output/glitch_s0.5.mp4"
    ]
    
    results = []
    
    # Analyze each file
    for f in files:
        if os.path.exists(f):
            analysis = analyze_video(f)
            if analysis:
                results.append(analysis)
                print(f"    Optical Flow: {analysis['flow_avg']:.4f}")
                print(f"    Pixel Difference: {analysis['pixel_avg']:.2f}")
                print(f"    Combined Score: {analysis['combined']:.3f}")
        else:
            print(f"  ⚠️  File not found: {f}")
    
    if not results:
        print("\n❌ No videos found for analysis!")
        print("   Generate videos first: python glitch_cinema.py")
        return
    
    # Create visualization
    create_comparison_chart(results)
    create_correlation_chart(results)
    
    # Print summary
    print_summary(results)

def create_comparison_chart(results):
    """Create bar chart comparing all videos"""
    names = [r['filename'].replace('.mp4', '') for r in results]
    flow_scores = [r['flow_avg'] for r in results]
    pixel_scores = [r['pixel_avg'] for r in results]
    combined_scores = [r['combined'] for r in results]
    
    x = np.arange(len(names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, flow_scores, width, label='Optical Flow', color='blue', alpha=0.7)
    bars2 = ax.bar(x, pixel_scores, width, label='Pixel Difference', color='green', alpha=0.7)
    bars3 = ax.bar(x + width, combined_scores, width, label='Combined Score', color='red', alpha=0.7)
    
    ax.set_xlabel('Video Version')
    ax.set_ylabel('Instability Score')
    ax.set_title('Quantitative Instability Analysis\n(Optical Flow + Pixel Difference)')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('output/instability_comparison.png', dpi=100)
    print(f"\n✅ Comparison chart saved: output/instability_comparison.png")

def create_correlation_chart(results):
    """Show correlation between injection level and instability"""
    levels = [r['level'] for r in results]
    combined_scores = [r['combined'] for r in results]
    
    if len(levels) < 2:
        return
    
    # Calculate correlation
    correlation = np.corrcoef(levels, combined_scores)[0, 1]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Scatter plot
    scatter = ax.scatter(levels, combined_scores, s=200, alpha=0.7, 
                        c=levels, cmap='viridis', edgecolors='black')
    
    # Add trendline
    z = np.polyfit(levels, combined_scores, 1)
    p = np.poly1d(z)
    ax.plot(levels, p(levels), "r--", alpha=0.5, linewidth=2, 
            label=f'Correlation: r = {correlation:.3f}')
    
    ax.set_xlabel('Noise Injection Level')
    ax.set_ylabel('Combined Instability Score')
    ax.set_title('Correlation: Injection Level vs Temporal Instability')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add labels to points
    for i, (x, y) in enumerate(zip(levels, combined_scores)):
        ax.annotate(f'({x}, {y:.3f})', (x, y), 
                   textcoords="offset points", 
                   xytext=(0,10), 
                   ha='center', fontsize=9)
    
    plt.colorbar(scatter, label='Injection Level')
    plt.tight_layout()
    plt.savefig('output/correlation_analysis.png', dpi=100)
    print(f"✅ Correlation chart saved: output/correlation_analysis.png")
    print(f"📈 Correlation coefficient: r = {correlation:.3f}")

def print_summary(results):
    """Print formatted summary table"""
    print(f"\n{'='*60}")
    print("SUMMARY OF TEMPORAL INSTABILITY ANALYSIS")
    print(f"{'='*60}")
    print(f"{'Video':<20} {'Injection':<10} {'Optical Flow':<15} {'Pixel Diff':<12} {'Combined':<10}")
    print(f"{'-'*60}")
    
    for r in results:
        print(f"{r['filename']:<20} {r['level']:<10.1f} {r['flow_avg']:<15.4f} "
              f"{r['pixel_avg']:<12.2f} {r['combined']:<10.3f}")
    
    print(f"{'='*60}")
    
    # Find which has highest instability
    if results:
        max_instability = max(results, key=lambda x: x['combined'])
        print(f"🎯 Highest Instability: {max_instability['filename']} "
              f"(Score: {max_instability['combined']:.3f})")
    
    print(f"\n📊 Charts saved to 'output/' folder:")
    print(f"   1. instability_comparison.png - Bar chart comparison")
    print(f"   2. correlation_analysis.png - Correlation analysis")

if __name__ == "__main__":
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Check if OpenCV is installed
    try:
        import cv2
        main()
    except ImportError:
        print("❌ OpenCV not installed!")
        print("Install it with: pip install opencv-python")
        print("Then run: python complete_analysis.py")