import time
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# Import project modules
from src.utils import load_config, setup_logger
from src.classic_strategies import get_strategy

# ---------------------------------------------------------
# 1. Helper Function to Generate Synthetic Data
# ---------------------------------------------------------
def generate_synthetic_sentences(num_sentences: int) -> List[str]:
    base_sentences = [
        "Software architecture has evolved significantly over the past few decades.",
        "Microservices offer advantages like resilience and independent deployment.",
        "A monolithic application is built as a single unified unit.",
        "Scaling a monolith can be inefficient because the entire application must be scaled.",
        "These services communicate with each other through well-defined APIs."
    ]
    
    generated = []
    for i in range(num_sentences):
        base = base_sentences[i % len(base_sentences)]
        generated.append(f"{base} (sentence id {i})")
    
    return generated

# ---------------------------------------------------------
# 2. Benchmarking Function (Updated for Comparison)
# ---------------------------------------------------------
def run_comparative_benchmark(config, sizes: List[int], strategies: List[str]) -> Dict[str, List[float]]:

    results = {name: [] for name in strategies}
    logger = setup_logger("Benchmark")
    
    logger.info("Starting Comparative Performance Analysis...")
    logger.info(f"Strategies to test: {strategies}")
    logger.info(f"Input sizes (sentences): {sizes}")

    for n in sizes:
        # 1. Generate Data
        sentences = generate_synthetic_sentences(n)
        logger.info(f"--- Benchmarking N={n} ---")
        
        for strategy_name in strategies:
            try:
                # Get Strategy Instance
                strategy = get_strategy(strategy_name)
                
                # Start Timer
                start_time = time.time()
                
                # Execution
                _ = strategy.calculate_scores(sentences, config)
                
                # Stop Timer
                duration = time.time() - start_time
                results[strategy_name].append(duration)
                
                # logger.info(f"   -> {strategy_name}: {duration:.4f}s")
                
            except Exception as e:
                logger.error(f"Error in {strategy_name} at size {n}: {e}")
                results[strategy_name].append(0)

    return results


def save_comparative_plot(sizes, results, output_dir):
    plt.figure(figsize=(12, 7))
    
    colors = {'textrank': 'blue', 'frequency': 'green', 'greedy': 'orange'}
    markers = {'textrank': 'o', 'frequency': 's', 'greedy': '^'}
    
    # Plot Actual Data for each strategy
    for name, times in results.items():
        if not times: continue
        plt.plot(sizes, times, 
                 marker=markers.get(name, 'x'), 
                 linestyle='-', 
                 color=colors.get(name, 'black'), 
                 label=f'{name.capitalize()} (Actual)',
                 linewidth=2)

    # --- Plot Theoretical Curves (Comparison with Big-O) ---
    # TextRank is usually O(N^2) due to similarity matrix
    if 'textrank' in results and len(sizes) > 0:
        tr_times = results['textrank']
        if tr_times[-1] > 0:
            theoretical_n2 = [n**2 for n in sizes]
            # Normalize to match the last point of actual data
            scale_n2 = tr_times[-1] / theoretical_n2[-1]
            plt.plot(sizes, [t * scale_n2 for t in theoretical_n2], 
                     linestyle='--', color='blue', alpha=0.4, 
                     label='Theoretical O(N^2)')

    # Frequency is usually O(N) (Linear scan)
    if 'frequency' in results and len(sizes) > 0:
        fr_times = results['frequency']
        if fr_times[-1] > 0:
            theoretical_n = [n for n in sizes]
            # Normalize
            scale_n = fr_times[-1] / theoretical_n[-1]
            plt.plot(sizes, [t * scale_n for t in theoretical_n], 
                     linestyle='--', color='green', alpha=0.4, 
                     label='Theoretical O(N)')

    plt.title('Algorithm Performance Comparison: TextRank vs Frequency')
    plt.xlabel('Number of Sentences (N)')
    plt.ylabel('Execution Time (Seconds)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "comparative_performance.png")
    plt.savefig(output_path)
    print(f"\n[Success] Comparative plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    # Load Config
    try:
        config = load_config()
    except Exception as e:
        print(f"Warning: Could not load config ({e}). Using defaults.")
        # Minimal config mock
        config = {
            'textrank': {'similarity_threshold': 0.1, 'damping_factor': 0.85, 'max_iterations': 50, 'convergence_threshold': 0.0001},
            'io': {'output_dir': 'data/outputs'}
        }

    # Define Test Cases (Input Sizes)
    input_sizes = [10, 50, 100, 200, 300, 500, 800] 
    
    # Define Strategies to Compare
    strategies_to_test = ['textrank', 'frequency']
    
    # Run Benchmark
    results = run_comparative_benchmark(config, input_sizes, strategies_to_test)
    
    # Save Results
    save_comparative_plot(input_sizes, results, config['io']['output_dir'])