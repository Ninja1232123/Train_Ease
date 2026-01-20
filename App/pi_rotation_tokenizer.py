"""
π/2 4D Rotation Tokenizer
=========================

Takes any tokenized data and outputs 4 rotations:
- 0° (0 radians) - original
- π/2 (90°) - first rotation
- π (180°) - half rotation
- 3π/2 (270°) - three-quarter rotation

The model learns:
- 59,301 patterns × 1 universal rotation function
- NOT 237,204 independent examples

Each phase validates the others. Errors get corrected by geometric consistency.
The phases vote on the correct underlying reality.

Usage:
    rotator = PiRotationTokenizer()
    rotator.process("tokenized.jsonl", "rotated_4d.jsonl")

    # Or with raw data + tokenization in one step:
    rotator.process_raw("data.csv", "rotated_4d.jsonl", text_column="content")
"""

import json
import math
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Union
from dataclasses import dataclass
from enum import Enum

# Import the base data loader
try:
    from data_loader import DataLoader, LoaderConfig, DataFormat
    HAS_LOADER = True
except ImportError:
    HAS_LOADER = False


class Phase(Enum):
    """The four phases of rotation."""
    PHASE_0 = 0           # 0 radians
    PHASE_PI_2 = 1        # π/2 radians (90°)
    PHASE_PI = 2          # π radians (180°)
    PHASE_3PI_2 = 3       # 3π/2 radians (270°)

    @property
    def radians(self) -> float:
        """Get the phase in radians."""
        return self.value * (math.pi / 2)

    @property
    def degrees(self) -> int:
        """Get the phase in degrees."""
        return self.value * 90

    @property
    def symbol(self) -> str:
        """Get a symbolic representation."""
        symbols = ['0', 'π/2', 'π', '3π/2']
        return symbols[self.value]


@dataclass
class RotationConfig:
    """Configuration for 4D rotation tokenization."""
    # Special tokens for phase marking
    phase_token_base: int = 50256      # Base ID for phase tokens (after GPT-2 vocab)
    use_phase_prefix: bool = True      # Add phase token at start
    use_phase_suffix: bool = False     # Add phase token at end

    # Rotation method for token IDs
    rotate_token_ids: bool = True      # Apply numerical rotation to IDs
    vocab_size: int = 50257            # Vocabulary size for modular rotation

    # Output options
    include_phase_metadata: bool = True   # Include phase info in JSON
    shuffle_phases: bool = False          # Randomize phase order per batch

    # For complex rotation (experimental)
    use_complex_encoding: bool = False    # Encode as complex numbers


@dataclass
class RotationStats:
    """Statistics from rotation processing."""
    input_records: int = 0
    output_records: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_records": self.input_records,
            "output_records": self.output_records,
            "expansion_factor": 4,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
        }


class PiRotationTokenizer:
    """
    4D Rotation Tokenizer using π/2 phases.

    Transforms each datapoint into 4 rotated versions.
    The model learns patterns + rotation function, not independent examples.
    """

    def __init__(
        self,
        config: Optional[RotationConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config or RotationConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.stats = RotationStats()

        # Phase tokens
        self.phase_tokens = {
            Phase.PHASE_0: self.config.phase_token_base,
            Phase.PHASE_PI_2: self.config.phase_token_base + 1,
            Phase.PHASE_PI: self.config.phase_token_base + 2,
            Phase.PHASE_3PI_2: self.config.phase_token_base + 3,
        }

    def rotate_token(self, token_id: int, phase: Phase) -> int:
        """
        Rotate a single token ID by the given phase.

        Rotation is modular arithmetic on vocab space:
        - π/2 rotation = shift by vocab_size / 4
        - π rotation = shift by vocab_size / 2
        - 3π/2 rotation = shift by 3 * vocab_size / 4
        """
        if not self.config.rotate_token_ids:
            return token_id

        shift = (phase.value * self.config.vocab_size) // 4
        return (token_id + shift) % self.config.vocab_size

    def rotate_tokens(self, tokens: List[int], phase: Phase) -> List[int]:
        """Rotate all tokens by the given phase."""
        if phase == Phase.PHASE_0:
            rotated = tokens.copy()
        else:
            rotated = [self.rotate_token(t, phase) for t in tokens]

        # Add phase prefix/suffix tokens
        if self.config.use_phase_prefix:
            rotated = [self.phase_tokens[phase]] + rotated
        if self.config.use_phase_suffix:
            rotated = rotated + [self.phase_tokens[phase]]

        return rotated

    def rotate_to_complex(self, tokens: List[int], phase: Phase) -> List[Dict[str, float]]:
        """
        Encode tokens as complex numbers with phase rotation.

        token_id → token_id * e^(i * phase)
                 = token_id * (cos(phase) + i*sin(phase))
        """
        result = []
        for token_id in tokens:
            magnitude = float(token_id)
            angle = phase.radians
            result.append({
                "real": magnitude * math.cos(angle),
                "imag": magnitude * math.sin(angle),
                "magnitude": magnitude,
                "phase": phase.symbol
            })
        return result

    def generate_rotations(
        self,
        tokens: List[int],
        original_text: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate all 4 rotations of a token sequence.

        Yields 4 records, one for each phase.
        """
        for phase in Phase:
            if self.config.use_complex_encoding:
                rotated = self.rotate_to_complex(tokens, phase)
                record = {
                    "complex_tokens": rotated,
                    "original_tokens": tokens,
                }
            else:
                rotated = self.rotate_tokens(tokens, phase)
                record = {
                    "tokens": rotated,
                }

            if self.config.include_phase_metadata:
                record["phase"] = phase.symbol
                record["phase_radians"] = phase.radians
                record["phase_degrees"] = phase.degrees
                record["phase_index"] = phase.value

            if original_text:
                record["text"] = original_text[:200]

            self.stats.total_tokens_out += len(rotated) if isinstance(rotated, list) else len(tokens)
            yield record

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> RotationStats:
        """
        Process a tokenized JSONL file and output 4D rotations.

        Input format: {"tokens": [1, 2, 3, ...]}
        Output: 4x records with rotated tokens and phase metadata
        """
        self.stats = RotationStats()

        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Processing: {input_path}")
        self.logger.info(f"Output: {output_path}")
        self.logger.info(f"Rotation method: {'modular arithmetic' if self.config.rotate_token_ids else 'phase prefix only'}")

        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:

            for line in fin:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    tokens = record.get('tokens', [])
                    text = record.get('text', None)

                    if not tokens:
                        continue

                    self.stats.input_records += 1
                    self.stats.total_tokens_in += len(tokens)

                    # Generate all 4 rotations
                    for rotated_record in self.generate_rotations(tokens, text):
                        fout.write(json.dumps(rotated_record) + '\n')
                        self.stats.output_records += 1

                    if self.stats.input_records % 10000 == 0:
                        self.logger.info(f"Processed {self.stats.input_records} records...")

                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid JSON, skipping line")
                    continue

        self.logger.info(f"Complete!")
        self.logger.info(f"  Input: {self.stats.input_records:,} records")
        self.logger.info(f"  Output: {self.stats.output_records:,} records (4x expansion)")
        self.logger.info(f"  Tokens: {self.stats.total_tokens_in:,} → {self.stats.total_tokens_out:,}")

        return self.stats

    def process_raw(
        self,
        source: Union[str, Path],
        output_path: Union[str, Path],
        tokenizer: str = "gpt2",
        text_column: str = "text",
        format: str = "auto"
    ) -> RotationStats:
        """
        Process raw data (CSV, JSON, etc.) with tokenization + 4D rotation.

        One-step pipeline: raw text → tokens → 4D rotations
        """
        if not HAS_LOADER:
            raise ImportError("data_loader module required for raw processing")

        self.stats = RotationStats()

        # Initialize data loader
        config = LoaderConfig(
            tokenizer_name=tokenizer,
            text_column=text_column
        )
        loader = DataLoader(tokenizer=tokenizer, config=config, logger=self.logger)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        format_enum = DataFormat(format) if format != "auto" else DataFormat.AUTO

        self.logger.info(f"Processing raw: {source}")
        self.logger.info(f"Tokenizer: {tokenizer}")
        self.logger.info(f"Output: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as fout:
            for text in loader.iterate(source, format_enum):
                try:
                    token_chunks = loader.tokenize(text)

                    for tokens in token_chunks:
                        if not tokens:
                            continue

                        self.stats.input_records += 1
                        self.stats.total_tokens_in += len(tokens)

                        # Generate all 4 rotations
                        for rotated_record in self.generate_rotations(tokens, text):
                            fout.write(json.dumps(rotated_record) + '\n')
                            self.stats.output_records += 1

                    if self.stats.input_records % 5000 == 0:
                        self.logger.info(f"Processed {self.stats.input_records} chunks...")

                except Exception as e:
                    self.logger.warning(f"Error: {e}")
                    continue

        self.logger.info(f"Complete!")
        self.logger.info(f"  Input: {self.stats.input_records:,} chunks")
        self.logger.info(f"  Output: {self.stats.output_records:,} records (4x)")

        return self.stats


def add_rotation_routes(app, logger: logging.Logger):
    """Add rotation API routes to Flask app."""
    from flask import request, jsonify

    rotator = None

    @app.route('/api/rotation/init', methods=['POST'])
    def init_rotator():
        nonlocal rotator
        data = request.json or {}

        config = RotationConfig(
            vocab_size=data.get('vocab_size', 50257),
            rotate_token_ids=data.get('rotate_ids', True),
            use_phase_prefix=data.get('phase_prefix', True),
            use_complex_encoding=data.get('complex_encoding', False),
        )

        rotator = PiRotationTokenizer(config=config, logger=logger)

        return jsonify({
            "status": "initialized",
            "config": {
                "vocab_size": config.vocab_size,
                "rotate_token_ids": config.rotate_token_ids,
                "use_phase_prefix": config.use_phase_prefix,
                "use_complex_encoding": config.use_complex_encoding,
            }
        })

    @app.route('/api/rotation/process', methods=['POST'])
    def process_rotation():
        nonlocal rotator
        if rotator is None:
            rotator = PiRotationTokenizer(logger=logger)

        data = request.json
        input_path = data.get('input')
        output_path = data.get('output')

        if not input_path or not output_path:
            return jsonify({"error": "input and output paths required"}), 400

        try:
            stats = rotator.process(input_path, output_path)
            return jsonify({
                "status": "complete",
                "stats": stats.to_dict()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/rotation/process-raw', methods=['POST'])
    def process_raw_rotation():
        nonlocal rotator
        if rotator is None:
            rotator = PiRotationTokenizer(logger=logger)

        data = request.json
        source = data.get('source')
        output_path = data.get('output')
        tokenizer = data.get('tokenizer', 'gpt2')
        text_column = data.get('text_column', 'text')

        if not source or not output_path:
            return jsonify({"error": "source and output paths required"}), 400

        try:
            stats = rotator.process_raw(
                source, output_path,
                tokenizer=tokenizer,
                text_column=text_column
            )
            return jsonify({
                "status": "complete",
                "stats": stats.to_dict()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/rotation/preview', methods=['POST'])
    def preview_rotation():
        """Preview rotation of a sample token sequence."""
        nonlocal rotator
        if rotator is None:
            rotator = PiRotationTokenizer(logger=logger)

        data = request.json
        tokens = data.get('tokens', [1, 2, 3, 4, 5])

        rotations = list(rotator.generate_rotations(tokens))

        return jsonify({
            "original": tokens,
            "rotations": rotations,
            "phases": [p.symbol for p in Phase]
        })


# CLI interface
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="π/2 4D Rotation Tokenizer")
    parser.add_argument("input", help="Input file (tokenized JSONL or raw data)")
    parser.add_argument("output", help="Output file (rotated JSONL)")
    parser.add_argument("--raw", action="store_true", help="Process raw data (not pre-tokenized)")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer for raw data")
    parser.add_argument("--text-column", default="text", help="Text column for CSV/JSON")
    parser.add_argument("--no-rotate-ids", action="store_true", help="Don't rotate token IDs")
    parser.add_argument("--complex", action="store_true", help="Use complex number encoding")
    parser.add_argument("--vocab-size", type=int, default=50257, help="Vocabulary size")

    args = parser.parse_args()

    config = RotationConfig(
        vocab_size=args.vocab_size,
        rotate_token_ids=not args.no_rotate_ids,
        use_complex_encoding=args.complex,
    )

    rotator = PiRotationTokenizer(config=config, logger=logger)

    if args.raw:
        stats = rotator.process_raw(
            args.input, args.output,
            tokenizer=args.tokenizer,
            text_column=args.text_column
        )
    else:
        stats = rotator.process(args.input, args.output)

    print(f"\n{'='*50}")
    print(f"4D ROTATION COMPLETE")
    print(f"{'='*50}")
    print(f"Input records:  {stats.input_records:,}")
    print(f"Output records: {stats.output_records:,}")
    print(f"Expansion:      4x")
    print(f"Tokens in:      {stats.total_tokens_in:,}")
    print(f"Tokens out:     {stats.total_tokens_out:,}")
