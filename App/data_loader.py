"""
Universal Data Loader - Convert Any Dataset to Tokenized JSONL

Handles:
- CSV (with text column)
- JSON (array or nested)
- JSONL (one object per line)
- Parquet
- Plain text (.txt)
- HuggingFace datasets

Outputs tokenized JSONL ready for π-scaling and training.

Usage:
    loader = DataLoader(tokenizer="gpt2")
    loader.process("data.csv", "output.jsonl", text_column="content")
    loader.process("data.parquet", "output.jsonl")
    loader.process("raw_text.txt", "output.jsonl")
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator, Union
from dataclasses import dataclass, field
from enum import Enum
import csv

# Optional imports - graceful fallback if not installed
try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


class DataFormat(str, Enum):
    """Supported input data formats."""
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    PARQUET = "parquet"
    TXT = "txt"
    HUGGINGFACE = "huggingface"
    AUTO = "auto"


@dataclass
class LoaderConfig:
    """Configuration for data loading."""
    tokenizer_name: str = "gpt2"
    text_column: str = "text"           # Column containing text data
    max_length: int = 2048              # Max tokens per sample
    stride: int = 512                   # Overlap for chunking long texts
    batch_size: int = 1000              # Records per batch for progress
    skip_empty: bool = True             # Skip empty/null text entries
    lowercase: bool = False             # Lowercase text before tokenizing
    strip_whitespace: bool = True       # Strip leading/trailing whitespace

    # For HuggingFace datasets
    hf_split: str = "train"
    hf_subset: Optional[str] = None

    # Output options
    include_text: bool = False          # Include original text in output
    chunk_long_texts: bool = True       # Split texts longer than max_length


@dataclass
class LoaderStats:
    """Statistics from data loading."""
    total_records: int = 0
    processed_records: int = 0
    skipped_records: int = 0
    total_tokens: int = 0
    chunks_created: int = 0
    errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "processed_records": self.processed_records,
            "skipped_records": self.skipped_records,
            "total_tokens": self.total_tokens,
            "chunks_created": self.chunks_created,
            "errors": self.errors,
            "avg_tokens_per_record": self.total_tokens / max(1, self.processed_records)
        }


class DataLoader:
    """
    Universal data loader that converts any dataset format to tokenized JSONL.

    Example:
        loader = DataLoader(tokenizer="gpt2")
        stats = loader.process("data.csv", "tokenized.jsonl", text_column="content")
        print(f"Processed {stats.processed_records} records")
    """

    def __init__(
        self,
        tokenizer: str = "gpt2",
        config: Optional[LoaderConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config or LoaderConfig(tokenizer_name=tokenizer)
        self.logger = logger or logging.getLogger(__name__)
        self.tokenizer = None
        self.stats = LoaderStats()

        # Initialize tokenizer
        self._init_tokenizer(tokenizer)

    def _init_tokenizer(self, tokenizer_name: str):
        """Initialize the tokenizer."""
        if not HAS_TRANSFORMERS:
            self.logger.warning(
                "transformers not installed. Install with: pip install transformers\n"
                "Using simple whitespace tokenizer as fallback."
            )
            self.tokenizer = None
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.logger.info(f"Loaded tokenizer: {tokenizer_name}")
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer {tokenizer_name}: {e}")
            self.tokenizer = None

    def _simple_tokenize(self, text: str) -> List[int]:
        """Fallback tokenizer - simple word-based with hash."""
        words = text.split()
        # Use hash to create pseudo-token IDs
        return [hash(w) % 50000 for w in words]

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text, handling long sequences."""
        if self.config.strip_whitespace:
            text = text.strip()
        if self.config.lowercase:
            text = text.lower()

        if not text:
            return []

        if self.tokenizer is None:
            return self._simple_tokenize(text)

        # Tokenize with truncation handling
        if self.config.chunk_long_texts:
            # Use return_overflowing_tokens for proper chunking
            encoded = self.tokenizer(
                text,
                max_length=self.config.max_length,
                truncation=True,
                return_overflowing_tokens=True,
                stride=self.config.stride,
                return_tensors=None
            )

            # Handle both single and multiple chunks
            if isinstance(encoded['input_ids'][0], list):
                return encoded['input_ids']  # Multiple chunks
            else:
                return [encoded['input_ids']]  # Single chunk wrapped in list
        else:
            encoded = self.tokenizer(
                text,
                max_length=self.config.max_length,
                truncation=True,
                return_tensors=None
            )
            return [encoded['input_ids']]

    def detect_format(self, path: Union[str, Path]) -> DataFormat:
        """Auto-detect file format from extension."""
        path = Path(path)
        ext = path.suffix.lower()

        format_map = {
            '.csv': DataFormat.CSV,
            '.json': DataFormat.JSON,
            '.jsonl': DataFormat.JSONL,
            '.parquet': DataFormat.PARQUET,
            '.txt': DataFormat.TXT,
            '.text': DataFormat.TXT,
        }

        return format_map.get(ext, DataFormat.TXT)

    def _iter_csv(self, path: Path) -> Iterator[str]:
        """Iterate over text from CSV file."""
        text_col = self.config.text_column

        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)

            # Check if text column exists
            if reader.fieldnames and text_col not in reader.fieldnames:
                # Try common column names
                common_names = ['text', 'content', 'body', 'message', 'sentence', 'document']
                for name in common_names:
                    if name in reader.fieldnames:
                        text_col = name
                        self.logger.info(f"Using column '{text_col}' for text")
                        break
                else:
                    # Use first column
                    text_col = reader.fieldnames[0]
                    self.logger.warning(f"Column '{self.config.text_column}' not found, using '{text_col}'")

            for row in reader:
                self.stats.total_records += 1
                text = row.get(text_col, '')
                if text or not self.config.skip_empty:
                    yield text
                else:
                    self.stats.skipped_records += 1

    def _iter_json(self, path: Path) -> Iterator[str]:
        """Iterate over text from JSON file."""
        text_col = self.config.text_column

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Try to find array in common keys
            for key in ['data', 'records', 'items', 'rows', 'examples']:
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break
            else:
                items = [data]  # Single record
        else:
            items = []

        for item in items:
            self.stats.total_records += 1
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = item.get(text_col, '')
            else:
                text = str(item)

            if text or not self.config.skip_empty:
                yield text
            else:
                self.stats.skipped_records += 1

    def _iter_jsonl(self, path: Path) -> Iterator[str]:
        """Iterate over text from JSONL file."""
        text_col = self.config.text_column

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.stats.total_records += 1
                line = line.strip()
                if not line:
                    self.stats.skipped_records += 1
                    continue

                try:
                    record = json.loads(line)
                    if isinstance(record, str):
                        text = record
                    elif isinstance(record, dict):
                        text = record.get(text_col, '')
                    else:
                        text = str(record)

                    if text or not self.config.skip_empty:
                        yield text
                    else:
                        self.stats.skipped_records += 1
                except json.JSONDecodeError:
                    self.stats.errors += 1
                    self.logger.warning(f"Invalid JSON line, skipping")

    def _iter_parquet(self, path: Path) -> Iterator[str]:
        """Iterate over text from Parquet file."""
        if not HAS_PARQUET:
            raise ImportError("pyarrow required for Parquet. Install: pip install pyarrow")

        text_col = self.config.text_column
        table = pq.read_table(path)

        # Find text column
        columns = table.column_names
        if text_col not in columns:
            for name in ['text', 'content', 'body', 'message']:
                if name in columns:
                    text_col = name
                    break
            else:
                text_col = columns[0]
                self.logger.warning(f"Using column '{text_col}' for text")

        col_data = table.column(text_col)
        for i in range(len(col_data)):
            self.stats.total_records += 1
            text = str(col_data[i].as_py()) if col_data[i].as_py() else ''

            if text or not self.config.skip_empty:
                yield text
            else:
                self.stats.skipped_records += 1

    def _iter_txt(self, path: Path) -> Iterator[str]:
        """Iterate over text from plain text file."""
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        # Split by double newlines (paragraph) or treat as single doc
        paragraphs = content.split('\n\n')

        for para in paragraphs:
            self.stats.total_records += 1
            text = para.strip()
            if text or not self.config.skip_empty:
                yield text
            else:
                self.stats.skipped_records += 1

    def _iter_huggingface(self, dataset_name: str) -> Iterator[str]:
        """Iterate over text from HuggingFace dataset."""
        if not HAS_DATASETS:
            raise ImportError("datasets required. Install: pip install datasets")

        text_col = self.config.text_column

        ds = load_dataset(
            dataset_name,
            self.config.hf_subset,
            split=self.config.hf_split,
            streaming=True
        )

        for item in ds:
            self.stats.total_records += 1
            text = item.get(text_col, '')

            if text or not self.config.skip_empty:
                yield text
            else:
                self.stats.skipped_records += 1

    def iterate(
        self,
        source: Union[str, Path],
        format: DataFormat = DataFormat.AUTO
    ) -> Iterator[str]:
        """Iterate over text from any source."""

        # Handle HuggingFace datasets
        if format == DataFormat.HUGGINGFACE or (
            isinstance(source, str) and '/' in source and not Path(source).exists()
        ):
            yield from self._iter_huggingface(str(source))
            return

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if format == DataFormat.AUTO:
            format = self.detect_format(path)

        iterators = {
            DataFormat.CSV: self._iter_csv,
            DataFormat.JSON: self._iter_json,
            DataFormat.JSONL: self._iter_jsonl,
            DataFormat.PARQUET: self._iter_parquet,
            DataFormat.TXT: self._iter_txt,
        }

        iterator_fn = iterators.get(format, self._iter_txt)
        yield from iterator_fn(path)

    def process(
        self,
        source: Union[str, Path],
        output: Union[str, Path],
        format: DataFormat = DataFormat.AUTO,
        text_column: Optional[str] = None
    ) -> LoaderStats:
        """
        Process a dataset file and output tokenized JSONL.

        Args:
            source: Input file path or HuggingFace dataset name
            output: Output JSONL file path
            format: Input format (auto-detected if not specified)
            text_column: Override text column name

        Returns:
            LoaderStats with processing statistics
        """
        # Reset stats
        self.stats = LoaderStats()

        if text_column:
            self.config.text_column = text_column

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Processing: {source}")
        self.logger.info(f"Output: {output_path}")
        self.logger.info(f"Tokenizer: {self.config.tokenizer_name}")

        with open(output_path, 'w', encoding='utf-8') as fout:
            batch_count = 0

            for text in self.iterate(source, format):
                try:
                    # Tokenize (may return multiple chunks)
                    token_chunks = self.tokenize(text)

                    for tokens in token_chunks:
                        if not tokens:
                            continue

                        record = {"tokens": tokens}

                        if self.config.include_text and len(token_chunks) == 1:
                            record["text"] = text[:500]  # Truncate for storage

                        fout.write(json.dumps(record) + '\n')

                        self.stats.processed_records += 1
                        self.stats.total_tokens += len(tokens)

                        if len(token_chunks) > 1:
                            self.stats.chunks_created += 1

                    batch_count += 1
                    if batch_count % self.config.batch_size == 0:
                        self.logger.info(f"Processed {batch_count} records...")

                except Exception as e:
                    self.stats.errors += 1
                    self.logger.warning(f"Error processing record: {e}")

        self.logger.info(f"Complete! {self.stats.processed_records} records, "
                        f"{self.stats.total_tokens:,} tokens")

        return self.stats

    def preview(
        self,
        source: Union[str, Path],
        num_samples: int = 5,
        format: DataFormat = DataFormat.AUTO
    ) -> List[Dict[str, Any]]:
        """Preview tokenization of first N samples."""
        samples = []

        for i, text in enumerate(self.iterate(source, format)):
            if i >= num_samples:
                break

            token_chunks = self.tokenize(text)
            tokens = token_chunks[0] if token_chunks else []

            samples.append({
                "text": text[:200] + ("..." if len(text) > 200 else ""),
                "tokens": tokens[:50],
                "token_count": len(tokens),
                "chunks": len(token_chunks)
            })

        return samples


def add_data_loader_routes(app, logger: logging.Logger):
    """Add data loader routes to Flask app."""
    from flask import request, jsonify

    # Shared loader instance
    loader_state = {"loader": None, "tokenizer": "gpt2"}

    @app.route('/api/data-loader/init', methods=['POST'])
    def init_loader():
        """Initialize data loader with specified tokenizer."""
        data = request.get_json() or {}
        tokenizer_name = data.get("tokenizer", "gpt2")

        try:
            config = LoaderConfig(
                tokenizer_name=tokenizer_name,
                text_column=data.get("text_column", "text"),
                max_length=data.get("max_length", 2048),
                include_text=data.get("include_text", False)
            )

            loader_state["loader"] = DataLoader(tokenizer_name, config, logger)
            loader_state["tokenizer"] = tokenizer_name

            return jsonify({
                "status": "initialized",
                "tokenizer": tokenizer_name,
                "has_transformers": HAS_TRANSFORMERS,
                "has_parquet": HAS_PARQUET,
                "has_datasets": HAS_DATASETS
            })
        except Exception as e:
            logger.error(f"Loader init error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/data-loader/formats', methods=['GET'])
    def list_data_loader_formats():
        """List supported data formats."""
        return jsonify({
            "formats": [f.value for f in DataFormat if f != DataFormat.AUTO],
            "supported": {
                "csv": True,
                "json": True,
                "jsonl": True,
                "parquet": HAS_PARQUET,
                "txt": True,
                "huggingface": HAS_DATASETS
            },
            "install_hints": {
                "parquet": "pip install pyarrow",
                "huggingface": "pip install datasets",
                "transformers": "pip install transformers"
            }
        })

    @app.route('/api/data-loader/preview', methods=['POST'])
    def preview_data_loader():
        """Preview tokenization of a dataset."""
        data = request.get_json() or {}
        source = data.get("source", "")
        num_samples = data.get("num_samples", 5)
        text_column = data.get("text_column", "text")

        if not source:
            return jsonify({"error": "Provide 'source' path or dataset name"}), 400

        try:
            loader = loader_state.get("loader")
            if not loader:
                loader = DataLoader("gpt2", logger=logger)
                loader_state["loader"] = loader

            if text_column:
                loader.config.text_column = text_column

            samples = loader.preview(source, num_samples)

            return jsonify({
                "source": source,
                "samples": samples,
                "tokenizer": loader_state.get("tokenizer", "gpt2")
            })
        except Exception as e:
            logger.error(f"Preview error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/data-loader/process', methods=['POST'])
    def process_data():
        """Process and tokenize a dataset."""
        data = request.get_json() or {}
        source = data.get("source", "")
        output = data.get("output", "")
        text_column = data.get("text_column", "text")

        if not source:
            return jsonify({"error": "Provide 'source' path or dataset name"}), 400
        if not output:
            # Generate output name
            source_path = Path(source)
            output = str(source_path.with_suffix('.tokenized.jsonl'))

        try:
            loader = loader_state.get("loader")
            if not loader:
                loader = DataLoader("gpt2", logger=logger)
                loader_state["loader"] = loader

            stats = loader.process(source, output, text_column=text_column)

            return jsonify({
                "status": "complete",
                "source": source,
                "output": output,
                "stats": stats.to_dict()
            })
        except Exception as e:
            logger.error(f"Process error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/data-loader/tokenize', methods=['POST'])
    def tokenize_text():
        """Tokenize raw text directly."""
        data = request.get_json() or {}
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "Provide 'text' to tokenize"}), 400

        try:
            loader = loader_state.get("loader")
            if not loader:
                loader = DataLoader("gpt2", logger=logger)
                loader_state["loader"] = loader

            token_chunks = loader.tokenize(text)
            tokens = token_chunks[0] if token_chunks else []

            return jsonify({
                "text": text[:200],
                "tokens": tokens,
                "token_count": len(tokens),
                "chunks": len(token_chunks)
            })
        except Exception as e:
            logger.error(f"Tokenize error: {e}")
            return jsonify({"error": str(e)}), 500

    logger.info("Data Loader routes registered")


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Universal Data Loader")
    parser.add_argument("source", help="Input file or HuggingFace dataset")
    parser.add_argument("-o", "--output", help="Output JSONL file")
    parser.add_argument("-t", "--tokenizer", default="gpt2", help="Tokenizer name")
    parser.add_argument("-c", "--column", default="text", help="Text column name")
    parser.add_argument("--max-length", type=int, default=2048, help="Max tokens per sample")
    parser.add_argument("--preview", type=int, help="Preview N samples only")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = LoaderConfig(
        tokenizer_name=args.tokenizer,
        text_column=args.column,
        max_length=args.max_length
    )

    loader = DataLoader(args.tokenizer, config)

    if args.preview:
        print(f"\nPreviewing {args.preview} samples from: {args.source}\n")
        samples = loader.preview(args.source, args.preview)

        for i, s in enumerate(samples):
            print(f"--- Sample {i+1} ---")
            print(f"Text: {s['text']}")
            print(f"Tokens ({s['token_count']}): {s['tokens'][:20]}...")
            print()
    else:
        output = args.output or Path(args.source).stem + ".tokenized.jsonl"
        stats = loader.process(args.source, output)

        print(f"\n{'='*50}")
        print("COMPLETE")
        print(f"{'='*50}")
        print(f"Records processed: {stats.processed_records:,}")
        print(f"Total tokens: {stats.total_tokens:,}")
        print(f"Avg tokens/record: {stats.total_tokens / max(1, stats.processed_records):.1f}")
        print(f"Output: {output}")


__all__ = [
    'DataLoader',
    'DataFormat',
    'LoaderConfig',
    'LoaderStats',
    'add_data_loader_routes',
]
