#!/usr/bin/env python3
"""
Synthetic ARC Dataset Generation Wrapper

This script provides a robust interface to the re-arc procedural example generator,
enabling controlled generation of synthetic ARC training data for multi-phase
pre-training strategies.

Modes:
    foundational_skills: Generate large datasets from selected primitive tasks
    distributional_alignment: Generate small, diverse samples across all 400 tasks

Author: JARC Reactor Team
Date: 2025-10-08
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for dataset generation."""
    mode: str
    output_dir: Path
    samples_per_task: int
    difficulty_min: float = 0.0
    difficulty_max: float = 1.0
    task_list_path: Optional[Path] = None
    max_retries_per_task: int = 10
    seed: Optional[int] = None


class ReArcInterface:
    """
    Interface to the re-arc submodule with dynamic import handling.
    
    This class manages the import of the re-arc library from the git submodule
    and provides a clean API for generating and verifying examples.
    """
    
    def __init__(self, rearc_path: Path):
        """
        Initialize the re-arc interface.
        
        Args:
            rearc_path: Path to the re-arc submodule directory
        """
        self.rearc_path = rearc_path
        self._ensure_submodule_exists()
        self._import_rearc_modules()
        
    def _ensure_submodule_exists(self):
        """Verify that the re-arc submodule exists and is initialized."""
        if not self.rearc_path.exists():
            raise FileNotFoundError(
                f"re-arc submodule not found at {self.rearc_path}. "
                f"Please run: git submodule update --init"
            )
        
        required_files = ['main.py', 'generators.py', 'verifiers.py', 'dsl.py']
        missing_files = [f for f in required_files if not (self.rearc_path / f).exists()]
        
        if missing_files:
            raise FileNotFoundError(
                f"re-arc submodule incomplete. Missing files: {missing_files}"
            )
    
    def _import_rearc_modules(self):
        """Dynamically import and map re-arc modules."""
        rearc_str = str(self.rearc_path.resolve())
        if rearc_str not in sys.path:
            sys.path.insert(0, rearc_str)

        try:
            import generators
            import verifiers
            from utils import strip_prefix

            gen_prefix = 'generate_'
            self.generators_mapper = {
                strip_prefix(n, gen_prefix): getattr(generators, n) 
                for n in dir(generators) if n.startswith(gen_prefix)
            }

            ver_prefix = 'verify_'
            self.verifiers_mapper = {
                strip_prefix(n, ver_prefix): getattr(verifiers, n) 
                for n in dir(verifiers) if n.startswith(ver_prefix)
            }

            logger.info(f"Successfully imported and mapped {len(self.generators_mapper)} generators "
                        f"and {len(self.verifiers_mapper)} verifiers from {self.rearc_path}")

        except ImportError as e:
            raise ImportError(
                f"Failed to import re-arc modules: {e}. "
                f"Ensure the submodule is properly initialized and all dependencies are met."
            )

    def get_all_task_ids(self) -> List[str]:
        """Get list of all available task IDs from the generators mapper."""
        return sorted(self.generators_mapper.keys())

    def generate_example(
        self, 
        task_id: str, 
        difficulty_range: Tuple[float, float] = (0.0, 1.0)
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a single example for a given task.
        
        Args:
            task_id: The 8-character task ID (e.g., '00d62c1b')
            difficulty_range: Tuple of (min, max) difficulty in [0.0, 1.0]
            
        Returns:
            Dictionary with 'input' and 'output' keys, or None if generation failed
        """
        generator_fn = self.generators_mapper.get(task_id)
        if not generator_fn:
            logger.warning(f"No generator found for task {task_id}")
            return None

        try:
            # The generator expects two positional args: diff_lb, diff_ub
            example = generator_fn(*difficulty_range)
            return example
        except Exception as e:
            logger.debug(f"Generation failed for task {task_id}: {e}")
            return None

    def verify_example(
        self, 
        task_id: str, 
        example: Dict[str, Any]
    ) -> bool:
        """
        Verify that a generated example is correct using the task's verifier.
        
        Args:
            task_id: The task ID
            example: Dictionary with 'input' and 'output' keys
            
        Returns:
            True if example is valid, False otherwise
        """
        verifier_fn = self.verifiers_mapper.get(task_id)
        if not verifier_fn:
            logger.warning(f"No verifier found for task {task_id}")
            return False

        try:
            # The verifier transforms the input and checks if it matches the output
            computed_output = verifier_fn(example['input'])
            return computed_output == example['output']
        except Exception as e:
            logger.debug(f"Verification failed for task {task_id}: {e}")
            return False


class SyntheticDatasetGenerator:
    """
    Main class for orchestrating synthetic dataset generation.
    """
    
    def __init__(self, config: GenerationConfig, rearc: ReArcInterface):
        """
        Initialize the generator.
        
        Args:
            config: Generation configuration
            rearc: Interface to the re-arc library
        """
        self.config = config
        self.rearc = rearc
        
        # Set random seed for reproducibility
        if config.seed is not None:
            random.seed(config.seed)
            logger.info(f"Set random seed to {config.seed}")
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'total_examples_attempted': 0,
            'total_examples_generated': 0,
            'total_examples_verified': 0,
            'failed_tasks': []
        }
    
    def generate_dataset(self):
        """
        Main entry point for dataset generation.
        Dispatches to the appropriate mode-specific method.
        """
        logger.info(f"Starting dataset generation in '{self.config.mode}' mode")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info(f"Samples per task: {self.config.samples_per_task}")
        logger.info(f"Difficulty range: [{self.config.difficulty_min}, {self.config.difficulty_max}]")
        
        if self.config.mode == 'foundational_skills':
            self._generate_foundational_skills()
        elif self.config.mode == 'distributional_alignment':
            self._generate_distributional_alignment()
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")
        
        self._write_statistics()
        self._print_summary()
    
    def _generate_foundational_skills(self):
        """
        Generate large datasets from a curated list of primitive tasks.
        """
        # Load task list
        if self.config.task_list_path is None:
            raise ValueError("foundational_skills mode requires --task-list argument")
        
        task_ids = self._load_task_list(self.config.task_list_path)
        logger.info(f"Loaded {len(task_ids)} tasks from {self.config.task_list_path}")
        
        # Generate examples for each task
        for task_id in task_ids:
            self._generate_task_examples(
                task_id=task_id,
                num_samples=self.config.samples_per_task,
                difficulty_range=(self.config.difficulty_min, self.config.difficulty_max)
            )
    
    def _generate_distributional_alignment(self):
        """
        Generate small, diverse samples across all 400 tasks.
        """
        # Get all available task IDs
        all_task_ids = self.rearc.get_all_task_ids()
        logger.info(f"Generating distributional alignment dataset for {len(all_task_ids)} tasks")
        
        # Generate a small number of examples for each task
        for task_id in all_task_ids:
            self._generate_task_examples(
                task_id=task_id,
                num_samples=self.config.samples_per_task,
                difficulty_range=(self.config.difficulty_min, self.config.difficulty_max)
            )
    
    def _generate_task_examples(
        self, 
        task_id: str, 
        num_samples: int,
        difficulty_range: Tuple[float, float]
    ):
        """
        Generate and verify examples for a single task.
        
        Args:
            task_id: The task ID
            num_samples: Number of verified examples to generate
            difficulty_range: Tuple of (min_difficulty, max_difficulty)
        """
        logger.info(f"Generating {num_samples} examples for task {task_id}")
        self.stats['total_tasks'] += 1
        
        verified_examples = []
        attempts = 0
        max_attempts = num_samples * self.config.max_retries_per_task
        
        while len(verified_examples) < num_samples and attempts < max_attempts:
            attempts += 1
            self.stats['total_examples_attempted'] += 1
            
            # Generate example
            example = self.rearc.generate_example(task_id, difficulty_range)
            if example is None:
                continue
            
            self.stats['total_examples_generated'] += 1
            
            # Verify example
            if self.rearc.verify_example(task_id, example):
                verified_examples.append(example)
                self.stats['total_examples_verified'] += 1
                
                # Log progress every 10 examples
                if len(verified_examples) % 10 == 0:
                    logger.debug(f"  Task {task_id}: {len(verified_examples)}/{num_samples} verified")
        
        # Check if we got enough examples
        if len(verified_examples) < num_samples:
            logger.warning(
                f"Task {task_id}: Only generated {len(verified_examples)}/{num_samples} "
                f"verified examples after {attempts} attempts"
            )
            self.stats['failed_tasks'].append({
                'task_id': task_id,
                'requested': num_samples,
                'generated': len(verified_examples),
                'attempts': attempts
            })
        else:
            self.stats['successful_tasks'] += 1
            logger.info(
                f"Task {task_id}: Successfully generated {len(verified_examples)} "
                f"verified examples in {attempts} attempts"
            )
        
        # Write examples to file in standard ARC format
        if verified_examples:
            self._write_task_file(task_id, verified_examples)
    
    def _write_task_file(self, task_id: str, examples: List[Dict[str, Any]]):
        """
        Write examples to a JSON file in standard ARC format.
        
        Args:
            task_id: The task ID
            examples: List of example dictionaries
        """
        output_file = self.config.output_dir / f"{task_id}.json"
        
        # Format as ARC expects: {"train": [...], "test": [...]}
        # For synthetic data, we only have training examples
        task_data = {
            "train": examples,
            "test": []  # Empty test set for synthetic data
        }
        
        with open(output_file, 'w') as f:
            json.dump(task_data, f, indent=2)
        
        logger.debug(f"Wrote {len(examples)} examples to {output_file}")
    
    def _load_task_list(self, task_list_path: Path) -> List[str]:
        """
        Load task IDs from a text file (one per line).
        
        Args:
            task_list_path: Path to the task list file
            
        Returns:
            List of task ID strings
        """
        with open(task_list_path, 'r') as f:
            task_ids = [line.split('#')[0].strip() for line in f if line.split('#')[0].strip()]
        
        return task_ids
    
    def _write_statistics(self):
        """Write generation statistics to a JSON file."""
        stats_file = self.config.output_dir / 'generation_statistics.json'
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Wrote statistics to {stats_file}")
    
    def _print_summary(self):
        """Print a summary of the generation process."""
        logger.info("=" * 70)
        logger.info("GENERATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Mode: {self.config.mode}")
        logger.info(f"Total tasks processed: {self.stats['total_tasks']}")
        logger.info(f"Successful tasks: {self.stats['successful_tasks']}")
        logger.info(f"Failed tasks: {len(self.stats['failed_tasks'])}")
        logger.info(f"Total examples attempted: {self.stats['total_examples_attempted']}")
        logger.info(f"Total examples generated: {self.stats['total_examples_generated']}")
        logger.info(f"Total examples verified: {self.stats['total_examples_verified']}")
        
        if self.stats['total_examples_attempted'] > 0:
            success_rate = (
                self.stats['total_examples_verified'] / 
                self.stats['total_examples_attempted'] * 100
            )
            logger.info(f"Overall verification rate: {success_rate:.1f}%")
        
        if self.stats['failed_tasks']:
            logger.warning(f"\nFailed tasks details:")
            for failure in self.stats['failed_tasks'][:10]:  # Show first 10
                logger.warning(
                    f"  {failure['task_id']}: "
                    f"{failure['generated']}/{failure['requested']} "
                    f"({failure['attempts']} attempts)"
                )
            if len(self.stats['failed_tasks']) > 10:
                logger.warning(f"  ... and {len(self.stats['failed_tasks']) - 10} more")
        
        logger.info("=" * 70)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic ARC datasets using the re-arc library',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate foundational skills dataset
  %(prog)s --mode foundational_skills \\
      --task-list tasks_for_foundational_skills.txt \\
      --samples-per-task 200 \\
      --output-dir data/synthetic_data/foundational_skills

  # Generate distributional alignment dataset
  %(prog)s --mode distributional_alignment \\
      --samples-per-task 15 \\
      --output-dir data/synthetic_data/distributional_alignment \\
      --difficulty-min 0.2 --difficulty-max 0.8
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['foundational_skills', 'distributional_alignment'],
        help='Generation mode'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for generated datasets'
    )
    
    parser.add_argument(
        '--samples-per-task',
        type=int,
        required=True,
        help='Number of verified examples to generate per task'
    )
    
    # Optional arguments
    parser.add_argument(
        '--task-list',
        type=Path,
        help='Path to task list file (required for foundational_skills mode)'
    )
    
    parser.add_argument(
        '--difficulty-min',
        type=float,
        default=0.0,
        help='Minimum difficulty (0.0 to 1.0, default: 0.0)'
    )
    
    parser.add_argument(
        '--difficulty-max',
        type=float,
        default=1.0,
        help='Maximum difficulty (0.0 to 1.0, default: 1.0)'
    )
    
    parser.add_argument(
        '--max-retries-per-task',
        type=int,
        default=10,
        help='Maximum retry multiplier per task (default: 10)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    
    # Default: Look for re-arc submodule in reproduction package external/re-arc
    # Fallback: Try parent repo structure (for local dev)
    default_rearc = Path(__file__).parent.parent / 'external' / 're-arc'
    if not default_rearc.exists():
        # Try parent repo structure: reproduction/../../../external/re-arc
        alt_rearc = Path(__file__).parent.parent.parent.parent.parent / 'external' / 're-arc'
        if alt_rearc.exists():
            default_rearc = alt_rearc
    
    parser.add_argument(
        '--rearc-path',
        type=Path,
        default=default_rearc,
        help='Path to re-arc submodule (default: auto-detected or ./external/re-arc)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose debug logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate difficulty range
    if not (0.0 <= args.difficulty_min <= 1.0):
        parser.error("difficulty-min must be between 0.0 and 1.0")
    if not (0.0 <= args.difficulty_max <= 1.0):
        parser.error("difficulty-max must be between 0.0 and 1.0")
    if args.difficulty_min > args.difficulty_max:
        parser.error("difficulty-min must be <= difficulty-max")
    
    # Create configuration
    config = GenerationConfig(
        mode=args.mode,
        output_dir=args.output_dir,
        samples_per_task=args.samples_per_task,
        difficulty_min=args.difficulty_min,
        difficulty_max=args.difficulty_max,
        task_list_path=args.task_list,
        max_retries_per_task=args.max_retries_per_task,
        seed=args.seed
    )
    
    try:
        # Initialize re-arc interface
        logger.info(f"Initializing re-arc interface from {args.rearc_path}")
        rearc = ReArcInterface(args.rearc_path)
        
        # Create generator and run
        generator = SyntheticDatasetGenerator(config, rearc)
        generator.generate_dataset()
        
        logger.info("Dataset generation complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
