import time
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Import project modules
# فرض بر این است که این فایل در کنار main.py قرار دارد
from src.vectorization import ManualTFIDF
from src.graph_utils import calculate_cosine_similarity_matrix, build_graph
from src.textrank import run_pagerank
from src.utils import load_config, setup_logger

# ---------------------------------------------------------
# 1. Helper Function to Generate Synthetic Data
# ---------------------------------------------------------
def generate_synthetic_sentences(num_sentences: int) -> List[str]:
    """
    تولید جملات مصنوعی برای تست استرس الگوریتم.
    این تابع تضمین می‌کند که کد بدون نیاز به فایل‌های متنی واقعی اجرا شود.
    """
    base_sentences = [
        "Software architecture has evolved significantly over the past few decades.",
        "Microservices offer advantages like resilience and independent deployment.",
        "A monolithic application is built as a single unified unit.",
        "Scaling a monolith can be inefficient because the entire application must be scaled.",
        "These services communicate with each other through well-defined APIs."
    ]
    
    generated = []
    for i in range(num_sentences):
        # تکرار چرخشی جملات پایه برای ساخت حجم دلخواه
        base = base_sentences[i % len(base_sentences)]
        # اضافه کردن اندیس برای متفاوت شدن متن‌ها (جلوگیری از شباهت ۱۰۰ درصدی همه)
        generated.append(f"{base} (sentence id {i})")
    
    return generated

# ---------------------------------------------------------
# 2. Benchmarking Function
# ---------------------------------------------------------
def run_benchmark(config, sizes: List[int]) -> Tuple[List[int], List[float]]:
    """
    الگوریتم TextRank را روی تعداد جملات مختلف اجرا کرده و زمان را اندازه می‌گیرد.
    """
    times = []
    logger = setup_logger("Benchmark")
    
    logger.info("Starting Performance Analysis...")
    logger.info(f"Testing input sizes (sentences): {sizes}")

    for n in sizes:
        # 1. Generate Data
        sentences = generate_synthetic_sentences(n)
        
        # 2. Start Timer
        start_time = time.time()
        
        try:
            # --- Core Algorithm (Phase 1 Logic) ---
            # A. Vectorization
            vectorizer = ManualTFIDF()
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # B. Graph Construction
            sim_matrix = calculate_cosine_similarity_matrix(tfidf_matrix)
            graph = build_graph(sim_matrix, config['textrank']['similarity_threshold'])
            
            # C. PageRank
            _ = run_pagerank(
                graph, 
                d=config['textrank']['damping_factor'],
                max_iter=config['textrank']['max_iterations'],
                tol=config['textrank']['convergence_threshold']
            )
            # --------------------------------------
            
        except Exception as e:
            logger.error(f"Error at size {n}: {e}")
            times.append(0)
            continue

        # 3. Stop Timer
        end_time = time.time()
        duration = end_time - start_time
        times.append(duration)
        
        logger.info(f"Size: {n} sentences | Time: {duration:.4f}s")

    return sizes, times

# ---------------------------------------------------------
# 3. Plotting and Saving Results
# ---------------------------------------------------------
def save_performance_plot(sizes, times, output_dir):
    """
    رسم نمودار رشد زمانی و ذخیره آن.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot Actual Data
    plt.plot(sizes, times, marker='o', linestyle='-', color='b', label='Actual Runtime')
    
    # Optional: Plot O(N^2) Theoretical Curve for comparison
    # نرمال‌سازی برای هم‌مقیاس شدن با زمان واقعی
    if len(sizes) > 0 and len(times) > 0:
        theoretical = [n**2 for n in sizes]
        scale_factor = times[-1] / theoretical[-1] if theoretical[-1] != 0 else 1
        theoretical_scaled = [t * scale_factor for t in theoretical]
        plt.plot(sizes, theoretical_scaled, linestyle='--', color='r', alpha=0.5, label='Theoretical O(N^2)')

    plt.title('Algorithm Performance Analysis (TextRank)')
    plt.xlabel('Number of Sentences (N)')
    plt.ylabel('Execution Time (Seconds)')
    plt.grid(True)
    plt.legend()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "performance_analysis.png")
    plt.savefig(output_path)
    print(f"\n[Success] Plot saved to: {output_path}")
    plt.close()

# ---------------------------------------------------------
# 4. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # Load Config
    try:
        config = load_config()
    except Exception as e:
        # Fallback config if file missing (to prevent crash)
        print(f"Warning: Could not load config ({e}). Using defaults.")
        config = {
            'textrank': {
                'similarity_threshold': 0.1,
                'damping_factor': 0.85,
                'max_iterations': 50,
                'convergence_threshold': 0.0001
            },
            'io': {'output_dir': 'data/outputs'}
        }

    # Define Test Cases (Input Sizes)
    # تست با تعداد جملات کم تا زیاد برای نشان دادن رشد نمایی
    input_sizes = [10, 50, 100, 200, 300, 500] 
    
    # Run Benchmark
    sizes, times = run_benchmark(config, input_sizes)
    
    # Save Results
    save_performance_plot(sizes, times, config['io']['output_dir'])