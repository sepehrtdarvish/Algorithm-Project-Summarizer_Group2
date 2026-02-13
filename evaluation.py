import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.utils import load_config, setup_logger
from src.preprocessing import split_into_sentences, filter_sentences
from src.classic_strategies import get_strategy
from src.llm_oracle import LLMOracle
from src.hybrid_merge import HybridMerger


TEST_CASES = [
    # --- LEVEL 1-3: SIMPLE ---
    {
        "id": 1, "difficulty": "Simple", "topic": "Story",
        "text": "Once upon a time, there was a little cat named Luna. Luna loved to play with yarn. One day, she found a big red ball of yarn under the sofa. She rolled it all over the house. Her owner laughed when he saw the mess. Luna was a happy cat."
    },
    {
        "id": 2, "difficulty": "Simple", "topic": "Repetitive",
        "text": "Apples are a healthy fruit. Apples are red or green. Many people like to eat apples every day. Eating apples is good for your health. Doctors say an apple a day keeps the doctor away. So, you should eat apples."
    },
    {
        "id": 3, "difficulty": "Simple", "topic": "Instruction",
        "text": "To make a cup of tea, first boil some water. Put a tea bag in a cup. Pour the hot water into the cup. Let it sit for three minutes. Remove the tea bag. Add sugar or milk if you like. Stir and enjoy your hot tea."
    },

    # --- LEVEL 4-7: MEDIUM ---
    {
        "id": 4, "difficulty": "Medium", "topic": "News",
        "text": "Local authorities announced a new plan to improve city traffic today. The plan involves building two new subway lines and expanding bus routes. Officials say this will reduce congestion by 30% over the next five years. However, some residents are concerned about the construction noise and cost. The mayor has promised to hold a public hearing next week."
    },
    {
        "id": 5, "difficulty": "Medium", "topic": "Technology",
        "text": "Cloud computing allows users to store and access data over the internet instead of on a hard drive. It offers flexibility, scalability, and cost savings for businesses. Major providers like AWS, Google Cloud, and Azure dominate the market. While convenient, security remains a top concern for many organizations moving sensitive data to the cloud."
    },
    {
        "id": 6, "difficulty": "Medium", "topic": "Biology",
        "text": "Photosynthesis is the process by which green plants create food. Using sunlight, water, and carbon dioxide, plants produce glucose and oxygen. This process takes place in the chloroplasts, which contain chlorophyll. Photosynthesis is vital for life on Earth as it provides the oxygen that humans and animals need to breathe."
    },
    {
        "id": 7, "difficulty": "Medium", "topic": "History",
        "text": "The Industrial Revolution marked a major turning point in history. Starting in Britain in the late 18th century, it shifted production from hand tools to complex machines. Steam engines powered factories, and urbanization increased rapidly. While it brought economic growth, it also led to poor working conditions and pollution."
    },

    # --- LEVEL 8-10: HARD ---
    {
        "id": 8, "difficulty": "Hard", "topic": "Physics",
        "text": "Quantum entanglement is a physical phenomenon that occurs when a group of particles interacts in such a way that the quantum state of each particle cannot be described independently. Measurements of physical properties such as position, momentum, spin, and polarization performed on entangled particles are found to be perfectly correlated, even when the particles are separated by large distances."
    },
    {
        "id": 9, "difficulty": "Hard", "topic": "Philosophy",
        "text": "Existentialism is a philosophical inquiry that explores the problem of human existence and centers on the lived experience of the thinking, feeling, and acting individual. In the view of the existentialist, the individual's starting point has been called 'the existential angst', a sense of dread, disorientation, confusion, or anxiety in the face of an apparently meaningless or absurd world."
    },
    {
        "id": 10, "difficulty": "Hard", "topic": "Project Doc",
        "text": "The Algorithm Design project requires students to implement a real-world problem using a complete algorithmic system. Phase 1 focuses on problem analysis and base algorithm design, including time complexity calculations. Phase 2 involves implementation, testing, and performance evaluation. Students must use an LLM as an oracle or judge to enhance the system's capabilities, specifically for tasks like semantic similarity or summarization scoring."
    }
]

class SummarizationBenchmark:
    def __init__(self):
        self.config = load_config()
        self.logger = setup_logger("Benchmark")
        self.oracle = LLMOracle(self.config['llm'])
        
        # Load Strategies
        self.tr_strategy = get_strategy('textrank')
        self.freq_strategy = get_strategy('frequency')
        
        # Hybrid Merger
        self.merger = HybridMerger(
            alpha=self.config['hybrid']['alpha'],
            beta=self.config['hybrid']['beta']
        )
        
        self.results_log = []

    def _generate_candidate_summaries(self, text):
        """Generates 5 types of summaries for a single text."""
        candidates = {}
        
        # 1. Preprocessing
        sentences = split_into_sentences(text)
        # For very short texts (Simple), we relax the filter
        min_len = 5 if len(text) < 200 else self.config['preprocessing']['min_sentence_length']
        sentences = filter_sentences(sentences, min_length=min_len)
        
        if not sentences:
            return None

        summary_count = min(len(sentences), self.config['hybrid']['final_summary_count'])

        # --- A. Pure LLM ---
        llm_summary_text = self.oracle.get_abstractive_summary(text)
        candidates['Pure LLM'] = llm_summary_text

        # --- B. Classic: TextRank ---
        try:
            tr_scores = self.tr_strategy.calculate_scores(sentences, self.config)
            top_idx = np.argsort(tr_scores)[::-1][:summary_count]
            top_idx = sorted(top_idx)
            candidates['TextRank'] = " ".join([sentences[i] for i in top_idx])
            
            # --- C. Hybrid: TextRank ---
            tr_merged = self.merger.merge_scores(sentences, tr_scores, llm_summary_text)
            candidates['Hybrid TR'] = self.merger.get_top_n(tr_merged, n=summary_count)
            
        except Exception as e:
            self.logger.error(f"TextRank failed: {e}")
            candidates['TextRank'] = ""
            candidates['Hybrid TR'] = ""

        # --- D. Classic: Frequency ---
        try:
            freq_scores = self.freq_strategy.calculate_scores(sentences, self.config)
            top_idx = np.argsort(freq_scores)[::-1][:summary_count]
            top_idx = sorted(top_idx)
            candidates['Frequency'] = " ".join([sentences[i] for i in top_idx])
            
            # --- E. Hybrid: Frequency ---
            freq_merged = self.merger.merge_scores(sentences, freq_scores, llm_summary_text)
            candidates['Hybrid Freq'] = self.merger.get_top_n(freq_merged, n=summary_count)
            
        except Exception as e:
            self.logger.error(f"Frequency failed: {e}")
            candidates['Frequency'] = ""
            candidates['Hybrid Freq'] = ""

        return candidates

    def run_benchmark(self):
        self.logger.info("Starting LLM-as-a-Judge Benchmark on 10 Scenarios...")
        
        for case in TEST_CASES:
            case_id = case['id']
            difficulty = case['difficulty']
            topic = case['topic']
            text = case['text']
            
            print(f"\n--- Processing Case {case_id}: {topic} ({difficulty}) ---")
            
            # 1. Generate Summaries
            summaries = self._generate_candidate_summaries(text)
            if not summaries:
                self.logger.warning(f"Skipping case {case_id} (Not enough sentences).")
                continue
                
            # 2. Batch Evaluate (LLM Judge)
            scores = self.oracle.evaluate_batch_summaries(text, summaries)
            
            # 3. Print Results in Console
            print(f"{'Method':<20} | Score")
            print("-" * 30)
            for method, score in scores.items():
                print(f"{method:<20} | {score}%")
            
            # 4. Store Data
            row = {
                "id": case_id,
                "difficulty": difficulty,
                "topic": topic,
                "TextRank": scores.get('TextRank', 0),
                "Frequency": scores.get('Frequency', 0),
                "Pure LLM": scores.get('Pure LLM', 0),
                "Hybrid TR": scores.get('Hybrid TR', 0),
                "Hybrid Freq": scores.get('Hybrid Freq', 0)
            }
            self.results_log.append(row)
            
            # Small sleep to avoid API rate limits
            time.sleep(2)

    def save_plot(self):
        if not self.results_log:
            print("No results to plot.")
            return

        df = pd.DataFrame(self.results_log)
        output_dir = self.config['io']['output_dir']
        
        plt.figure(figsize=(12, 6))
        
        methods = ['TextRank', 'Frequency', 'Pure LLM', 'Hybrid TR', 'Hybrid Freq']
        markers = ['o', 's', '^', 'D', 'x']
        
        for i, method in enumerate(methods):
            plt.plot(df['id'], df[method], marker=markers[i], linewidth=2, label=method)

        # Formatting
        plt.title('Algorithm Performance: LLM Judge Scores across Difficulty Levels')
        plt.xlabel('Test Cases (1-3: Simple, 4-7: Medium, 8-10: Hard)')
        plt.ylabel('Quality Score (0-100)')
        plt.xticks(df['id'], [f"{row['topic']}\n({row['difficulty']})" for _, row in df.iterrows()], rotation=45)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "benchmark_quality_scores.png")
        plt.savefig(plot_path)
        print(f"\n[Success] Benchmark plot saved to: {plot_path}")

if __name__ == "__main__":
    benchmark = SummarizationBenchmark()
    benchmark.run_benchmark()
    benchmark.save_plot()