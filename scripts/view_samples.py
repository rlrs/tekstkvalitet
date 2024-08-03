import argparse
import gzip
import json
import random
import glob
import os

def load_evaluated_texts(file_path):
    """Load evaluated texts from a gzipped JSONL file."""
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def get_text_snippet(text, max_length=1000):
    """Get a snippet of the text, not cutting words in half."""
    if len(text) <= max_length:
        return text
    snippet = text[:max_length]
    last_space = snippet.rfind(' ')
    return snippet[:last_space] + '...'

def view_random_samples(input_dir, num_samples):
    """View random samples of evaluated texts from all files in the input directory."""
    all_files = glob.glob(os.path.join(input_dir, '*_evaluated.jsonl.gz'))
    all_texts = []

    for file in all_files:
        all_texts.extend(load_evaluated_texts(file))
    print(f"Loaded {len(all_texts)} texts")

    if not all_texts:
        print("No evaluated texts found in the specified directory.")
        return

    samples = random.sample(all_texts, min(num_samples, len(all_texts)))

    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        text_snippet = get_text_snippet(sample['text'])
        print(f"Text snippet: {text_snippet}")
        print(f"Explanation: {sample.get('quality_explanation', 'N/A')}")
        print(f"Score: {sample.get('quality_score', 'N/A')}")

        # Add a check to see if the explanation mentions the content of the text snippet
        if 'quality_explanation' in sample:
            explanation_lower = sample['quality_explanation'].lower()
            snippet_words = set(text_snippet.lower().split())
            if any(word in explanation_lower for word in snippet_words if len(word) > 3):
                print("Validation: Explanation appears to reference the text content.")
            else:
                print("Validation: Warning - Explanation may not reference the text content.")

        print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="View random samples of evaluated texts")
    parser.add_argument("input_dir", type=str, help="Directory containing evaluated JSONL.gz files")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random samples to view")
    args = parser.parse_args()

    view_random_samples(args.input_dir, args.num_samples)

if __name__ == "__main__":
    main()
