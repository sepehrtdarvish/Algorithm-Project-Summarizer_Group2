import sys
import time
import os
import re
import numpy as np
from datetime import datetime
from src.utils import load_config, setup_logger, save_summary
from src.preprocessing import split_into_sentences, filter_sentences
from src.llm_oracle import LLMOracle
from src.hybrid_merge import HybridMerger
from src.classic_strategies import get_strategy
import warnings

warnings.filterwarnings("ignore")

def get_multiline_input():
    """
    Captures multiline input from the user via the console.
    Ends when the user types 'END' on a new line.
    """
    print("\n" + "-"*60)
    print(" INPUT REQUIRED")
    print("-"*60)
    print("Please paste your text below.")
    print("Type 'END' on a new line and press Enter to start processing.")
    print("Type 'EXIT' to quit the program.")
    print("-"*60)
    
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
            
        if line.strip().upper() == 'END':
            break
        if line.strip().upper() == 'EXIT':
            return None
            
        lines.append(line)
    
    return "\n".join(lines)

def evaluate_summaries_with_llm(oracle, original_text, candidates):
    """
    Uses the LLM to judge the quality of the generated summaries.
    Returns a dictionary of scores (0-100).
    """
    print("\n[Judge] Asking LLM to evaluate the summaries...")
    
    # Constructing a prompt for the LLM to act as a judge
    prompt = f"""
    You are an expert evaluator of text summarization systems.
    
    ORIGINAL TEXT:
    "{original_text}"
    
    Please evaluate the following 3 summaries based on Accuracy, Coherence, and Coverage.
    Assign a score from 0 to 100 for each.
    
    SUMMARY 1 (Classic Strategy):
    "{candidates['Classic']}"
    
    SUMMARY 2 (Semantic/LLM):
    "{candidates['Semantic']}"
    
    SUMMARY 3 (Hybrid Merged):
    "{candidates['Hybrid']}"
    
    RESPONSE FORMAT:
    You must return the result strictly in this format:
    Classic: <score>
    Semantic: <score>
    Hybrid: <score>
    
    Do not add any explanation, just the scores.
    """
    
    try:
        # We reuse the oracle's generation method to get the evaluation
        # Assuming get_abstractive_summary sends text to LLM
        response_text = oracle.get_abstractive_summary(prompt)
        
        # Parse scores using Regex to be robust
        scores = {}
        patterns = {
            'Classic': r"Classic:\s*(\d+)",
            'Semantic': r"Semantic:\s*(\d+)",
            'Hybrid': r"Hybrid:\s*(\d+)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response_text)
            if match:
                scores[key] = int(match.group(1))
            else:
                scores[key] = 0 # Default if parsing fails
                
        return scores, response_text
        
    except Exception as e:
        print(f"[Judge Error] Could not evaluate: {e}")
        return {'Classic': 0, 'Semantic': 0, 'Hybrid': 0}, "Error"

def process_text(text, config, logger, oracle):
    """
    Main processing pipeline: Preprocessing -> Classic -> LLM -> Hybrid -> Evaluation.
    """
    if not text or not text.strip():
        print("❌ No text entered. Please try again.")
        return

    start_total = time.time()

    # 1. Preprocessing
    logger.info("Processing input text...")
    sentences = split_into_sentences(text)
    # Relax filter for very short inputs to avoid errors
    min_len = 5 if len(text) < 300 else config['preprocessing']['min_sentence_length']
    sentences = filter_sentences(sentences, min_len)

    if len(sentences) < 2:
        print("⚠️ Input text is too short or has too few sentences to summarize.")
        return

    # 2. Phase 1: Classic Algorithm
    method_name = config.get('classic', {}).get('method', 'textrank')
    logger.info(f"Phase 1: Running {method_name.upper()}...")
    
    try:
        strategy = get_strategy(method_name)
        classic_scores = strategy.calculate_scores(sentences, config)
    except Exception as e:
        logger.error(f"Error in classic algorithm: {e}")
        print(f"❌ Error in Classic Algorithm: {e}")
        return

    # Generate Classic Summary Text
    top_n = config['hybrid']['final_summary_count']
    ranked_indices = np.argsort(classic_scores)[::-1][:top_n]
    ranked_indices = sorted(ranked_indices)
    classic_summary_text = " ".join([sentences[i] for i in ranked_indices])

    # 3. Phase 2: LLM Oracle
    logger.info("Phase 2: Querying LLM Oracle...")
    try:
        llm_summary_text = oracle.get_abstractive_summary(text)
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        llm_summary_text = "Error getting LLM summary."

    # 4. Hybrid Merge
    logger.info("Phase 3: Merging Scores...")
    merger = HybridMerger(
        alpha=config['hybrid']['alpha'],
        beta=config['hybrid']['beta']
    )
    
    results = merger.merge_scores(sentences, classic_scores, llm_summary_text)
    final_hybrid_summary = merger.get_top_n(results, n=config['hybrid']['final_summary_count'])

    # 5. LLM Judge Evaluation (New Feature)
    candidates = {
        "Classic": classic_summary_text,
        "Semantic": llm_summary_text,
        "Hybrid": final_hybrid_summary
    }
    
    scores, judge_reasoning = evaluate_summaries_with_llm(oracle, text, candidates)

    # 6. Output to Console
    print("\n" + "▒"*20 + "  FINAL RESULTS  " + "▒"*20)
    
    print(f"\n🔹 1. CLASSICAL SUMMARY ({method_name.upper()})")
    print("-" * 50)
    print(classic_summary_text)
    
    print("\n🔹 2. SEMANTIC SUMMARY (LLM Oracle)")
    print("-" * 50)
    print(llm_summary_text)
    
    print("\n✅ 3. HYBRID SUMMARY (Merged)")
    print("-" * 50)
    print(final_hybrid_summary)
    
    print("\n" + "="*50)
    print("⚖️  LLM JUDGE SCORECARD (0-100)")
    print("="*50)
    print(f"{'Method':<20} | {'Score':<10}")
    print("-" * 35)
    print(f"{'Classic':<20} | {scores['Classic']}")
    print(f"{'Semantic':<20} | {scores['Semantic']}")
    print(f"{'Hybrid':<20} | {scores['Hybrid']}")
    
    
    print("\n" + "▒"*60 + "\n")

    # Save logic
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = config['io']['output_dir']
        save_summary(output_dir, f"summary_{timestamp}_hybrid.txt", final_hybrid_summary)
        logger.info(f"Processing finished in {time.time() - start_total:.2f}s")
    except Exception as e:
        logger.warning(f"Could not save file: {e}")

def main():
    # Setup - Load once
    print("Initializing Engine... Please wait.")
    config = load_config()
    logger = setup_logger()
    
    # Initialize LLM Oracle
    try:
        oracle = LLMOracle(config['llm'])
    except Exception as e:
        logger.error(f"Failed to initialize LLM Oracle: {e}")
        print(f"Fatal Error: {e}")
        return

    logger.info("Interactive Summarization Engine Ready.")

    # Main Loop
    while True:
        try:
            user_text = get_multiline_input()
            
            if user_text is None:
                print("Exiting program. Goodbye!")
                break
                
            process_text(user_text, config, logger, oracle)
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"An unexpected error occurred: {e}")
            print("Restarting loop...")

if __name__ == "__main__":
    main()