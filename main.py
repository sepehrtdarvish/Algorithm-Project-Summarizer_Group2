# main.py
import sys
import time
from src.utils import load_config, setup_logger, read_text_file, save_summary
from src.preprocessing import split_into_sentences, filter_sentences
from src.vectorization import ManualTFIDF
from src.graph_utils import calculate_cosine_similarity_matrix, build_graph
from src.textrank import run_pagerank
from src.llm_oracle import LLMOracle
from src.hybrid_merge import HybridMerger
import os

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

    # 4. Phase 1: TextRank (Statistical)
    start_time = time.time()
    
    # Vectorization (TF-IDF)
    vectorizer = ManualTFIDF()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Graph Construction
    sim_matrix = calculate_cosine_similarity_matrix(tfidf_matrix)
    graph = build_graph(sim_matrix, config['textrank']['similarity_threshold'])
    
    # PageRank
    tr_scores = run_pagerank(
        graph, 
        d=config['textrank']['damping_factor'],
        max_iter=config['textrank']['max_iterations'],
        tol=config['textrank']['convergence_threshold']
    )
    
    logger.info(f"TextRank completed in {time.time() - start_time:.4f}s")

    # 5. Phase 2: LLM Oracle (Semantic)
    logger.info("Querying LLM Oracle...")
    oracle = LLMOracle(config['llm'])
    llm_summary = oracle.get_abstractive_summary(text)
    
    # 6. Hybrid Merge
    logger.info("Merging Statistical and Semantic Scores...")
    merger = HybridMerger(
        alpha=config['hybrid']['alpha'],
        beta=config['hybrid']['beta']
    )
    
    results = merger.merge_scores(sentences, tr_scores, llm_summary)
    final_summary = merger.get_top_n(results, n=config['hybrid']['final_summary_count'])

    # 7. Output
    logger.info("Final Summary Generated.")
    print("\n--- FINAL SUMMARY ---\n")
    print(final_summary)
    print("\n---------------------\n")
    
    output_filename = "summary_output.txt"
    save_summary(config['io']['output_dir'], output_filename, final_summary)
    logger.info(f"Summary saved to {config['io']['output_dir']}/{output_filename}")

if __name__ == "__main__":
    import os
    main()