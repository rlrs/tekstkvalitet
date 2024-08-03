"""
Filter the datasets for the quality classifier using Datatrove.
This is the first step in the training pipeline.
"""
import argparse
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import (
    C4QualityFilter,
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers import JsonlReader, WarcReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.writers.parquet import ParquetWriter
from datatrove.utils.typeshelper import Languages
import numpy as np
import nltk
nltk.download('punkt')

INPUT_BASE_DIR = "/work/dfm-data/pre-training/"
INPUT_DATASETS = ["colossal_oscar_1_0", "nordjylland_news", "lexdk", "eur-lex-sum-da", "dagw-hest"]
LIMITS = 1000*np.array([1, 0.1, 0.2, 0.02, 0.02])
MAIN_OUTPUT_PATH = "/work/RasmusLarsen#5473/quality_filtering"
FILTER_OUTPUT_PATH = f"{MAIN_OUTPUT_PATH}/"

def main():
    for dataset, limit in zip(INPUT_DATASETS, LIMITS):
        folder = f"{INPUT_BASE_DIR}/{dataset}/documents"
        pipeline = [
            JsonlReader(data_folder=folder, limit=limit, doc_progress=True, default_metadata={"url": "Unknown", "harmful_pp": "0"}),
            URLFilter(
                exclusion_writer=JsonlWriter(f"{FILTER_OUTPUT_PATH}/{dataset}/1_url_filter/")
            ),
            LanguageFilter(
                languages=[Languages.danish],
                exclusion_writer=JsonlWriter(
                    f"{FILTER_OUTPUT_PATH}/{dataset}/2_language_filter/",
                    output_filename="${language}/" + "/${rank}.parquet"
                )
            ),
            GopherRepetitionFilter(
                exclusion_writer=JsonlWriter(f"{FILTER_OUTPUT_PATH}/{dataset}/3_repetition_filter/", output_filename="${rank}.parquet"),
                language=Languages.danish
            ),
            GopherQualityFilter(
                exclusion_writer=JsonlWriter(f"{FILTER_OUTPUT_PATH}/{dataset}/4_gopher_quality_filter/", output_filename="${rank}.parquet"),
                language=Languages.danish
            ),
            # FineWebQualityFilter(
            #     exclusion_writer=JsonlWriter(f"{FILTER_OUTPUT_PATH}/{dataset}/5_fine_web_quality_filter/", output_filename="${rank}.jsonl.gz"),
            #     language=Languages.danish
            # ),
            # ParquetWriter(f"{MAIN_OUTPUT_PATH}/output/", output_filename=f"{dataset}_${{rank}}.parquet"), # parquetwriter is broken...
            JsonlWriter(f"{MAIN_OUTPUT_PATH}/output/", output_filename=f"{dataset}_${{rank}}.jsonl.gz", compression="gzip"),
        ]

        print(f"Running pipeline for {dataset} with {limit} docs...")
        executor = LocalPipelineExecutor(pipeline=pipeline, tasks=8, workers=8)
        print(executor.run())


if __name__ == "__main__":
    main()
