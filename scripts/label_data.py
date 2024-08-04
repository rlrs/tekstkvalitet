"""
Generate quality labels for text data using OpenAI API.
This is the second step in the pipeline.
"""
import argparse
import openai
from json import loads, dumps
import gzip
from tqdm import tqdm
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import glob
import sqlite3
import tiktoken
import json
import queue
import threading
from dotenv import load_dotenv
load_dotenv()

client = openai.Client()

def create_database(db_path):
    """Create SQLite database and table for storing evaluated texts with additional fields."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS evaluated_texts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  file_name TEXT,
                  text_hash TEXT,
                  text TEXT UNIQUE,
                  quality_explanation TEXT,
                  quality_score INTEGER,
                  metadata TEXT,
                  evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_file_name_text_hash ON evaluated_texts(file_name, text_hash)')
    conn.commit()
    conn.close()


def load_data_from_jsonl_gz(file_path: str):
    """Load data from a gzipped JSONL file."""
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield loads(line)

def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def generate_prompt(text: str) -> str:
    """Generate a prompt for the OpenAI model."""
    return (
    "Nedenfor er et ekstrakt af tekst fra en web-side. Vurder tekstens kvalitet ved hjælp af det additive 5-punkts scoringssystem beskrevet nedenfor. Der akkumuleres point baseret på opfyldelsen af hvert kriterium.\n"
    "Tilføj 1 point for grundlæggende relevans: Teksten indeholder information der er relevant for et eller flere konkrete, interessante emner. Dette udelukker spam, reklamer, og andet indhold som ofte består af løst sammensatte emner. Dette punkt fokuserer udelukkende på tilstedeværelsen af relevant information, uanset kvaliteten eller præsentationen.\n"
    "Tilføj endnu et point for sammenhæng og struktur: Teksten præsenterer information på en logisk og organiseret måde. Den har en klar struktur med en indledning, hoveddel og konklusion. Den er formateret på en letlæselig måde, fx anvender den afsnit og paragraffer, eller andre måder at opbryde teksten på. Dette punkt vurderer tekstens overordnede opbygning og flow, uafhængigt af indholdet.\n"
    "Giv et tredje point for sproglig kvalitet: Teksten udviser klar og præcis sprogbrug. Sætninger er velformulerede, grammatisk korrekte, og der bruges et passende ordforråd. Dette punkt fokuserer udelukkende på det sproglige aspekt, uanset indholdet eller strukturen.\n"
    "Giv et fjerde point for dybde og nuancer: Teksten går ud over overfladisk behandling af emnet og udforsker det i dybden. Den præsenterer forskellige perspektiver, understøtter påstande med beviser eller eksempler, og viser en nuanceret forståelse af emnet. Dette punkt vurderer indholdslagets kvalitet, uanset tekstens struktur eller sprogbrug.\n"
    "Giv et femte point for originalitet og indsigt: Teksten bidrager med ny viden, originale ideer eller unikke perspektiver til emnet. Den går ud over at gentage kendt information og demonstrerer kritisk tænkning, analytiske færdigheder eller kreativ problemløsning. Dette punkt fokuserer på tekstens intellektuelle bidrag, uafhængigt af de andre kriterier.\n"
    f"Teksten:\n{text}\n"
    "Efter at have undersøgt teksten:\n"
    "- Forklar i én kort paragraf din totale score, højst 50 ord.\n"
    "- Konkluder med scoren i formatet: \"Kvalitet: <point>\""
    )

def get_quality_evaluation(text: str, model: str, max_retries: int = 3) -> dict[str, str] | None:
    """Get quality evaluation with retry mechanism."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that evaluates the quality of web documents."},
                    {"role": "user", "content": generate_prompt(text)}
                ],
                temperature=0,
                max_tokens=256,
            )
            generated_text = response.choices[0].message.content.strip()

            matches = re.findall(r"(.+)\s*Kvalitet: (\d)", generated_text)
            if matches:
                explanation, score = matches[0]
                return {
                    "explanation": explanation,
                    "score": score
                }
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error after {max_retries} attempts: {e}")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff

    return None

def split_text(text: str, encoder, max_tokens: int):
    """Split text into chunks that don't exceed max_tokens, preserving formatting and creating even-sized chunks."""
    # Encode the entire text to get total tokens
    total_tokens = len(encoder.encode(text))

    # If the text is short enough, return it as is
    if total_tokens <= max_tokens:
        return [text]

    # Calculate the ideal chunk size
    num_chunks = (total_tokens + max_tokens - 1) // max_tokens  # Round up division
    ideal_chunk_tokens = total_tokens // num_chunks

    chunks = []
    current_chunk = []
    current_chunk_tokens = 0

    # Split the text into paragraphs, preserving empty lines
    paragraphs = re.split(r'(\n\s*\n)', text)

    for paragraph in paragraphs:
        paragraph_tokens = len(encoder.encode(paragraph))

        if current_chunk_tokens + paragraph_tokens > ideal_chunk_tokens and current_chunk:
            # If adding this paragraph exceeds the ideal chunk size and we have content,
            # save the current chunk and start a new one
            chunks.append(''.join(current_chunk))
            current_chunk = []
            current_chunk_tokens = 0

        current_chunk.append(paragraph)
        current_chunk_tokens += paragraph_tokens

        # If we've exceeded the max tokens, force a split
        while current_chunk_tokens > max_tokens:
            # Find a good splitting point within the last paragraph
            split_point = find_split_point(current_chunk[-1], encoder, max_tokens - (current_chunk_tokens - paragraph_tokens))

            # Add the first part of the split to the current chunk and create a new chunk
            first_part = current_chunk[-1][:split_point]
            chunks.append(''.join(current_chunk[:-1] + [first_part]))

            # Start a new chunk with the remainder
            current_chunk = [current_chunk[-1][split_point:]]
            current_chunk_tokens = len(encoder.encode(current_chunk[0]))

    # Add any remaining content as the last chunk
    if current_chunk:
        chunks.append(''.join(current_chunk))

    return chunks

def find_split_point(text, encoder, max_tokens):
    """Find a good point to split the text, preferring sentence boundaries."""
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return len(text)

    # Try to split at a sentence boundary
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current_length = 0
    for i, sentence in enumerate(sentences):
        sentence_tokens = len(encoder.encode(sentence))
        if current_length + sentence_tokens > max_tokens:
            # If adding this sentence would exceed the limit, split at the previous sentence
            return len(''.join(sentences[:i]))
        current_length += sentence_tokens

    # If we can't split at a sentence boundary, split at the token limit
    return len(encoder.decode(tokens[:max_tokens]))

def process_text(text: str, model: str, encoder, max_tokens: int) -> dict[str, str] | None:
    """Process a single text, splitting if necessary."""
    chunks = split_text(text, encoder, max_tokens)
    if len(chunks) == 1:
        return get_quality_evaluation(chunks[0], model)
    else:
        chunk_evaluations = [get_quality_evaluation(chunk, model) for chunk in chunks]
        scores = [int(eval['score']) for eval in chunk_evaluations if eval and eval['score']]
        explanations = [eval['explanation'] for eval in chunk_evaluations if eval and eval['explanation']]
        if scores:
            avg_score = sum(scores) / len(scores)
            combined_explanation = " ".join(explanations)
            return {
                "explanation": f"Combined evaluation of {len(chunks)} chunks: {combined_explanation}",
                "score": str(round(avg_score))
            }
    return None

def process_batch(batch, write_queue: queue.Queue, model: str, encoder, max_tokens: int, file_name: str):
    """Process a batch of texts and put results in the write queue."""
    for item in batch:
        text = item['text']
        text_hash = hash(text)
        evaluation = process_text(text, model, encoder, max_tokens)
        write_queue.put((file_name, text_hash, text, evaluation, item.get('metadata', {})))

def db_writer(db_path, write_queue, total_items):
    """Write evaluation results to the database from a queue."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    with tqdm(total=total_items, desc="Writing to database") as pbar:
        while True:
            item = write_queue.get()
            if item is None:  # None is our signal to stop
                break

            file_name, text_hash, text, evaluation, metadata = item
            if evaluation:
                c.execute("""INSERT OR REPLACE INTO evaluated_texts
                             (file_name, text_hash, text, quality_explanation, quality_score, metadata)
                             VALUES (?, ?, ?, ?, ?, ?)""",
                          (file_name, str(text_hash), text, evaluation['explanation'], evaluation['score'], json.dumps(metadata)))
                conn.commit()

            pbar.update(1)
            write_queue.task_done()

    conn.close()

def get_processed_texts(db_path, file_name):
    """Get the set of processed text hashes for a specific file."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT text_hash FROM evaluated_texts WHERE file_name = ?", (file_name,))
    processed_texts = set(row[0] for row in c.fetchall())
    conn.close()
    return processed_texts

def process_jsonl_gz_file(file_path, db_path, model, batch_size, max_workers):
    """Process a single gzipped JSONL file and store results in SQLite database."""
    encoder = tiktoken.encoding_for_model(model)
    max_tokens = 120000

    file_name = os.path.basename(file_path)
    processed_texts = get_processed_texts(db_path, file_name)

    data = [item for item in load_data_from_jsonl_gz(file_path) if hash(item['text']) not in processed_texts]
    total_items = len(data)

    if total_items == 0:
        print(f"All items in {file_name} have been processed. Skipping.")
        return

    write_queue = queue.Queue()

    # Start the database writer thread
    db_writer_thread = threading.Thread(target=db_writer, args=(db_path, write_queue, total_items))
    db_writer_thread.start()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for batch in chunk_list(data, batch_size):
            futures.append(executor.submit(process_batch, batch, write_queue, model, encoder, max_tokens, file_name))

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {os.path.basename(file_path)}"):
            future.result()

    # Signal the db_writer to stop
    write_queue.put(None)
    db_writer_thread.join()


def main(args):
    # Create SQLite database
    db_path = os.path.join(args.output, "evaluated_texts.db")
    create_database(db_path)

    # Get input files
    input_files = glob.glob(os.path.join(args.input, "*.jsonl.gz"))

    # Process each gzipped JSONL file
    for file in input_files:
        process_jsonl_gz_file(file, db_path, args.model, args.batch_size, args.max_workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate quality evaluations for text data using OpenAI API")
    parser.add_argument("--input", type=str, required=True, help="Directory of input .jsonl.gz files")
    parser.add_argument("--output", type=str, required=True, help="Directory for output SQLite database")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use for evaluation")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of concurrent API calls")

    args = parser.parse_args()
    main(args)
