import sys
import time
import os
import numpy as np
from src.utils import load_config, setup_logger, read_text_file, save_summary
from src.preprocessing import split_into_sentences, filter_sentences
from src.llm_oracle import LLMOracle
from src.hybrid_merge import HybridMerger
from src.classic_strategies import get_strategy 
import warnings

warnings.filterwarnings("ignore")

def main():
    # 1. Setup
    config = load_config()
    logger = setup_logger()
    logger.info("Starting Algorithmic Summarization Engine (Phase 2)")

    # 2. Load Data
    input_file = os.path.join(config['io']['input_dir'], config['io']['default_input_file'])
    try:
        text = read_text_file(input_file)
        logger.info(f"Loaded input file: {input_file} ({len(text)} chars)")
    except Exception as e:
        logger.error(f"Failed to load input: {e}")
        return

    # 3. Preprocessing
    sentences = split_into_sentences(text)
    sentences = filter_sentences(sentences, config['preprocessing']['min_sentence_length'])
    logger.info(f"Preprocessed: {len(sentences)} sentences.")

    if len(sentences) < 2:
        logger.warning("Not enough sentences to summarize.")
        return

    # 4. Phase 1: Classic Algorithm (Dynamic Selection)
    method_name = config.get('classic', {}).get('method', 'textrank')
    logger.info(f"Running Phase 1 with Strategy: {method_name.upper()}...")
    
    start_time = time.time()
    
    try:
        strategy = get_strategy(method_name)
        classic_scores = strategy.calculate_scores(sentences, config)
        
        logger.info(f"Classic algorithm ({method_name}) completed in {time.time() - start_time:.4f}s")
    except Exception as e:
        logger.error(f"Error in classic algorithm: {e}")
        return

    # Generate Classic Summary Text (Top N based on classic scores)
    top_n = config['hybrid']['final_summary_count']
    ranked_indices = np.argsort(classic_scores)[::-1][:top_n]
    ranked_indices = sorted(ranked_indices)
    classic_summary_text = " ".join([sentences[i] for i in ranked_indices])

    # 5. Phase 2: LLM Oracle (Semantic)
    logger.info("Querying LLM Oracle...")
    oracle = LLMOracle(config['llm'])
    llm_summary_text = oracle.get_abstractive_summary(text)
    
    # 6. Hybrid Merge
    logger.info("Merging Scores...")
    merger = HybridMerger(
        alpha=config['hybrid']['alpha'],
        beta=config['hybrid']['beta']
    )
    
    results = merger.merge_scores(sentences, classic_scores, llm_summary_text)
    final_hybrid_summary = merger.get_top_n(results, n=config['hybrid']['final_summary_count'])

    # 7. Output
    print("\n" + "="*50)
    print(f" 1. CLASSICAL SUMMARY ({method_name.upper()})")
    print("="*50)
    print(classic_summary_text)
    
    print("\n" + "="*50)
    print(" 2. SEMANTIC SUMMARY (LLM Oracle)")
    print("="*50)
    print(llm_summary_text)
    
    print("\n" + "="*50)
    print(" 3. HYBRID SUMMARY (Merged)")
    print("="*50)
    print(final_hybrid_summary)
    print("\n" + "="*50 + "\n")
    
    # Save files
    output_dir = config['io']['output_dir']
    save_summary(output_dir, f"summary_classic_{method_name}.txt", classic_summary_text)
    save_summary(output_dir, "summary_llm.txt", llm_summary_text)
    save_summary(output_dir, "summary_hybrid.txt", final_hybrid_summary)
    
    logger.info(f"Summaries saved to {output_dir}/")

if __name__ == "__main__":
    main()