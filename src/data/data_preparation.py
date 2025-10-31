# jarc_reactor/data/data_preparation.py
import os
import random
from jarc_reactor.decoding.augmentations import enumerate_augmentations
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import hydra
import orjson
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from jarc_reactor.utils.logging_config import setup_logging
from jarc_reactor.data.context_data import ContextPair
from jarc_reactor.utils.padding_utils import pad_to_fixed_size
from .data_loading_utils import inspect_data_structure

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class RawData:
    """A container for the lists of raw data loaded from files."""
    train_inputs: List[torch.Tensor] = field(default_factory=list)
    train_outputs: List[torch.Tensor] = field(default_factory=list)
    train_task_ids: List[str] = field(default_factory=list)
    train_context_pairs: List[ContextPair] = field(default_factory=list)
    test_inputs: List[torch.Tensor] = field(default_factory=list)
    test_outputs: List[torch.Tensor] = field(default_factory=list)
    test_task_ids: List[str] = field(default_factory=list)
    test_context_pairs: List[ContextPair] = field(default_factory=list)

    def append_train(self, input_tensor, output_tensor, task_id, context_pair):
        self.train_inputs.append(input_tensor)
        self.train_outputs.append(output_tensor)
        self.train_task_ids.append(task_id)
        self.train_context_pairs.append(context_pair)

    def append_test(self, input_tensor, output_tensor, task_id, context_pair):
        self.test_inputs.append(input_tensor)
        self.test_outputs.append(output_tensor)
        self.test_task_ids.append(task_id)
        self.test_context_pairs.append(context_pair)

    def to_dict(self):
        return {
            "train_inputs": self.train_inputs, "train_outputs": self.train_outputs, "train_task_ids": self.train_task_ids,
            "train_context_pairs": self.train_context_pairs,
            "test_inputs": self.test_inputs, "test_outputs": self.test_outputs, "test_task_ids": self.test_task_ids,
            "test_context_pairs": self.test_context_pairs
        }


def load_context_pair(filepath: str, task_id: str, num_to_select: int, cfg: DictConfig):
    """
    Load a selected set of context input/output pairs for a task and return them as a ContextPair padded to the configured model size.
    
    Parameters:
        filepath (str): Path to the task JSON file containing 'train' context pairs.
        task_id (str): Identifier for the task; returned alongside the ContextPair.
        num_to_select (int): Desired number of context pairs to select. When the model config enables dynamic_pairs, the final number may be capped by cfg.data.max_context_pairs or the number available.
        cfg (DictConfig): Configuration containing model and data settings (expects cfg.model.max_h, cfg.model.max_w, and optionally cfg.model.context_encoder.dynamic_pairs and cfg.data.max_context_pairs).
    
    Returns:
        tuple: (task_id, ContextPair) where ContextPair.context_input and ContextPair.context_output are long tensors shaped (k, H, W) with H=cfg.model.max_h and W=cfg.model.max_w, and k is the number of selected pairs.
        Returns (None, None) if the file is not found or if no context pairs are selected due to dynamic configuration.
    
    Raises:
        ValueError: If dynamic_pairs is False and the file contains fewer context pairs than num_to_select.
    """
    try:
        with open(filepath, 'rb') as f:
            task_data = orjson.loads(f.read())
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None, None

    context_pairs = task_data.get('train', [])
    num_available_pairs = len(context_pairs)

    # Honour dynamic_pairs flag from model.context_encoder
    dynamic_pairs = getattr(cfg.model.context_encoder, "dynamic_pairs", False)

    if dynamic_pairs:
        # Use dynamic selection with an optional cap to avoid excessive memory usage
        cap = int(getattr(cfg.data, "max_context_pairs", 6) or 6)
        if num_to_select is None or num_to_select <= 0:
            k = min(num_available_pairs, cap)
        else:
            k = min(num_available_pairs, num_to_select, cap)
        if k <= 0:
            logger.warning(f"Task {task_id} has zero context pairs; skipping.")
            return None, None
        # Randomly sample k pairs to promote diversity and avoid ordering bias
        selected_indices = random.sample(range(num_available_pairs), k)
    else:
        if num_available_pairs < num_to_select:
            raise ValueError(
                f"Task {task_id} at {filepath} has only {num_available_pairs} context pairs, but {num_to_select} are required."
            )
        if num_available_pairs > num_to_select:
            selected_indices = random.sample(range(num_available_pairs), num_to_select)
        else:
            selected_indices = range(num_to_select)

    # Per-task selection summary (debug-level)
    try:
        logger.debug(
            f"Task {task_id}: selected {len(selected_indices)} of {num_available_pairs} context pairs "
            f"(dynamic_pairs={dynamic_pairs}, requested={num_to_select})."
        )
    except Exception:
        pass

    target_shape = (cfg.model.max_h, cfg.model.max_w)

    if not selected_indices:
        empty_shape = (0, target_shape[0], target_shape[1])
        empty_input = torch.empty(empty_shape, dtype=torch.long)
        empty_output = torch.empty(empty_shape, dtype=torch.long)
        return task_id, ContextPair(
            context_input=empty_input,
            context_output=empty_output,
            expected_hw=target_shape,
        )

    context_inputs = []
    context_outputs = []

    for i in selected_indices:
        pair = context_pairs[i]
        input_grid = pair['input']
        output_grid = pair['output']

        input_tensor = torch.tensor(input_grid, dtype=torch.long)
        output_tensor = torch.tensor(output_grid, dtype=torch.long)

        padded_input = pad_to_fixed_size(input_tensor, target_shape=target_shape)
        padded_output = pad_to_fixed_size(output_tensor, target_shape=target_shape)

        context_inputs.append(padded_input)
        context_outputs.append(padded_output)

    final_context_input = torch.stack(context_inputs)
    final_context_output = torch.stack(context_outputs)

    expected_shape = (len(selected_indices), cfg.model.max_h, cfg.model.max_w)

    return task_id, ContextPair(
        context_input=final_context_input,
        context_output=final_context_output,
        expected_hw=(cfg.model.max_h, cfg.model.max_w)
    )


def load_context_pairs(directory: str, cfg: DictConfig) -> Dict[str, ContextPair]:
    logger.info(f"Loading context pairs from '{directory}'...")
    num_to_select = cfg.data.num_context_pairs
    dynamic_pairs = getattr(cfg.model.context_encoder, "dynamic_pairs", False)
    if dynamic_pairs and (num_to_select is None or num_to_select <= 0):
        cap = int(getattr(cfg.data, "max_context_pairs", 6) or 6)
        logger.info(f"Will load up to {cap} context pair(s) per task (dynamic_pairs=true).")
    else:
        logger.info(f"Will load {num_to_select} context pair(s) per task.")
    context_map = {}
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(load_context_pair, os.path.join(directory, filename), os.path.splitext(filename)[0], num_to_select, cfg): filename
            for filename in json_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading context pairs"):
            try:
                result = future.result()
                if result:
                    task_id, context_pair = result
                    context_map[task_id] = context_pair
            except ValueError as e:
                logger.warning(f"Skipping task due to context pair issue: {e}")
    return context_map


def _process_data_split(data_split: List[Dict[str, Any]], task_id: str, context_pair: ContextPair, cfg: DictConfig) -> tuple:
    """
    Filter and prepare a list of sample dictionaries into per-sample tensors and matching context pairs.
    
    Processes each item in `data_split` by padding its 'input' and 'output' to the configured model shape, collecting those tensors and the corresponding `task_id`. For each sample, filters the provided `context_pair` to remove context entries whose inputs are identical to the current sample input (to avoid label leakage). If the context encoder is configured as static (dynamic pairs disabled), the filtered context list is padded by repeating the first retained entry to preserve the original context count. If the context shapes are unexpected or an error occurs during filtering, the original `context_pair` is preserved for that sample.
    
    Parameters:
        data_split (List[Dict[str, Any]]): Sequence of sample dicts each containing 'input' and 'output' grids.
        task_id (str): Identifier to associate with each returned sample.
        context_pair (ContextPair): Context inputs/outputs to filter per sample.
        cfg (DictConfig): Configuration providing model.max_h and model.max_w and optional context_encoder.dynamic_pairs.
    
    Returns:
        tuple: A 4-tuple (inputs, outputs, task_ids, context_pairs) where
            - inputs (List[torch.Tensor]): Padded input tensors for each sample.
            - outputs (List[torch.Tensor]): Padded output tensors for each sample.
            - task_ids (List[str]): The `task_id` repeated for each sample.
            - context_pairs (List[ContextPair]): Per-sample ContextPair objects after filtering/padding as described.
    """
    if not data_split:
        return [], [], [], []

    inputs, outputs, task_ids, context_pairs = [], [], [], []
    target_shape = (cfg.model.max_h, cfg.model.max_w)
    try:
        context_encoder_cfg = getattr(cfg.model, "context_encoder", None)
    except Exception:
        context_encoder_cfg = None
    dynamic_pairs_enabled = bool(getattr(context_encoder_cfg, "dynamic_pairs", False))

    for item in data_split:
        # Pad current sample
        cur_input = pad_to_fixed_size(torch.tensor(item['input'], dtype=torch.long), target_shape)
        cur_output = pad_to_fixed_size(torch.tensor(item['output'], dtype=torch.long), target_shape)
        inputs.append(cur_input)
        outputs.append(cur_output)
        task_ids.append(task_id)
        # --- Context integrity: exclude identical inputs to prevent label leakage ---
        try:
            ctx_in = context_pair.context_input  # [K, H, W]
            ctx_out = context_pair.context_output  # [K, H, W]
            if ctx_in.dim() == 3 and ctx_in.size(1) == target_shape[0] and ctx_in.size(2) == target_shape[1]:
                original_count = ctx_in.size(0)
                # Build mask for non-identical context inputs vs current sample input
                diffs = (ctx_in != cur_input.unsqueeze(0))
                keep_mask = diffs.view(diffs.size(0), -1).any(dim=1)
                # Ensure at least one context remains; if all identical, keep the first as fallback
                if not keep_mask.any():
                    keep_mask = torch.zeros_like(keep_mask)
                    keep_mask[0] = True
                filtered_in = ctx_in[keep_mask]
                filtered_out = ctx_out[keep_mask]

                # If dynamic pairs are explicitly disabled in config, preserve original count by
                # padding with the padding token (10) instead of duplicating contexts.
                # Context encoders are designed to mask out padding tokens, so this is transparent.
                strict_static_pairs = context_encoder_cfg is not None and not dynamic_pairs_enabled
                if strict_static_pairs and filtered_in.size(0) < original_count:
                    pad_count = original_count - filtered_in.size(0)
                    H, W = filtered_in.size(1), filtered_in.size(2)
                    # Use padding token (10) for null context grids - context encoders will mask these out
                    pad_token_id = int(getattr(cfg.model, "pad_token_id", 10))
                    pad_in = torch.full((pad_count, H, W), pad_token_id, dtype=filtered_in.dtype, device=filtered_in.device)
                    pad_out = torch.full((pad_count, H, W), pad_token_id, dtype=filtered_out.dtype, device=filtered_out.device)
                    filtered_in = torch.cat([filtered_in, pad_in], dim=0)
                    filtered_out = torch.cat([filtered_out, pad_out], dim=0)

                filtered_pair = ContextPair(
                    context_input=filtered_in,
                    context_output=filtered_out,
                    expected_hw=context_pair.expected_hw
                )
                context_pairs.append(filtered_pair)
            else:
                # When shapes unexpected, preserve the original pair list
                context_pairs.append(context_pair)
        except Exception:
            # Conservative fallback: do not filter if any error occurs
            context_pairs.append(context_pair)
    return inputs, outputs, task_ids, context_pairs


@dataclass
class FileDataProcessor:
    """Encapsulates the state and logic for processing data files."""
    context_map: Dict[str, ContextPair]
    is_synthetic: bool
    cfg: DictConfig
    mode: str
    limit_train_samples: Optional[int] = None
    _inspected_files_count: int = 0
    _successful_inspections: int = 0
    _inspection_log_limit: int = 2

    def _split_synthetic_data(self, data: List[Dict]) -> tuple[List[Dict], List[Dict]]:
        """Splits a list of synthetic examples into train and test sets."""
        potential_examples = data[1:]  # Assuming the first element is metadata
        if len(potential_examples) >= 2:
            # Use a fixed random_state for reproducibility
            return train_test_split(potential_examples, test_size=0.2, shuffle=True, random_state=42)
        return potential_examples, []

    def _inspect_file_if_needed(self, filepath: str):
        """Inspects the data structure of a file if the inspection limit has not been reached."""
        if self._inspected_files_count < self._inspection_log_limit:
            self._inspected_files_count += 1
            filename = os.path.basename(filepath)
            directory = os.path.dirname(filepath)
            if inspect_data_structure(self.cfg, filename, directory):
                self._successful_inspections += 1

    def process_file(self, filepath: str) -> tuple[List, List]:
        """Loads and processes a single data file based on the mode."""
        task_id = Path(filepath).stem
        self._inspect_file_if_needed(filepath)
        
        try:
            with open(filepath, 'rb') as f:
                data = orjson.loads(f.read())

            context_pair = self.context_map.get(task_id)
            if context_pair is None:
                logger.warning(f"Context missing for task {task_id}, skipping all samples in this file.")
                return [], []

            if self.is_synthetic:
                train_data, test_data = self._split_synthetic_data(data)
            else:
                train_data = data.get('train', [])
                test_data = data.get('test', [])

            if self.mode == 'eval':
                # In eval mode, the 'test' data is the primary data source.
                # The 'train' data from the file is used for the validation/context set.
                # In eval mode, the 'test' data is the primary data source.
                # The 'train' data from the file is used for the validation/context set.
                main_results = _process_data_split(test_data, task_id, context_pair, self.cfg)
                val_results = _process_data_split(train_data, task_id, context_pair, self.cfg)
                return main_results, val_results
            
            # This is train mode logic
            train_results = _process_data_split(train_data, task_id, context_pair, self.cfg)
            test_results = _process_data_split(test_data, task_id, context_pair, self.cfg)

            return train_results, test_results
        except Exception as e:
            logger.error(f"Error processing file '{filepath}': {str(e)}")
            return [], []


def load_main_data_concurrently(directory: str, processor: FileDataProcessor, limit_samples: Optional[int] = None):
    """Load main dataset from the specified directory concurrently using a FileDataProcessor."""
    raw_data = RawData()
    filepaths = [str(p) for p in Path(directory).glob('*.json')]
    # NOTE: The `limit_samples` parameter now correctly limits the number of samples, not files.

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(processor.process_file, fp) for fp in filepaths]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing data files"):
            main_results, validation_results = future.result()

            if processor.mode == 'eval':
                # In eval mode, main_results are test data, validation_results are train data
                if main_results:
                    raw_data.test_inputs.extend(main_results[0])
                    raw_data.test_outputs.extend(main_results[1])
                    raw_data.test_task_ids.extend(main_results[2])
                    raw_data.test_context_pairs.extend(main_results[3])
                if validation_results:
                    raw_data.train_inputs.extend(validation_results[0])
                    raw_data.train_outputs.extend(validation_results[1])
                    raw_data.train_task_ids.extend(validation_results[2])
                    raw_data.train_context_pairs.extend(validation_results[3])
            else: # This is 'train' mode
                if main_results and main_results[0]:
                    raw_data.train_inputs.extend(main_results[0])
                    raw_data.train_outputs.extend(main_results[1])
                    raw_data.train_task_ids.extend(main_results[2])
                    raw_data.train_context_pairs.extend(main_results[3])
                if validation_results and validation_results[0]:
                    raw_data.test_inputs.extend(validation_results[0])
                    raw_data.test_outputs.extend(validation_results[1])
                    raw_data.test_task_ids.extend(validation_results[2])
                    raw_data.test_context_pairs.extend(validation_results[3])

    # Apply sample limit after collecting all data
    if processor.mode == 'train' and limit_samples is not None and raw_data.train_inputs:
        logger.info(f"Applying sample limit: {limit_samples} samples.")
        raw_data.train_inputs = raw_data.train_inputs[:limit_samples]
        raw_data.train_outputs = raw_data.train_outputs[:limit_samples]
        raw_data.train_task_ids = raw_data.train_task_ids[:limit_samples]
        raw_data.train_context_pairs = raw_data.train_context_pairs[:limit_samples]

    return raw_data.to_dict()


def _validate_path(directory: str) -> Path:
    """Validates that the given path exists and is a directory."""
    absolute_directory_path = hydra.utils.to_absolute_path(directory)
    data_path = Path(absolute_directory_path)
    if not data_path.exists():
        msg = f"Absolute data directory does not exist: {absolute_directory_path} (original input: {directory})"
        logger.error(msg)
        raise FileNotFoundError(msg)
    if not data_path.is_dir():
        msg = f"Provided absolute path is not a directory: {absolute_directory_path} (original input: {directory})"
        logger.error(msg)
        raise NotADirectoryError(msg)
    return data_path


def _load_raw_data(cfg: DictConfig, data_path: Path, mode: str, is_synthetic: bool = False) -> dict:
    """Loads context pairs and main data concurrently."""
    context_map = load_context_pairs(str(data_path), cfg)
    logger.info("Loading main dataset...")
    limit_samples = cfg.data.get('limit_train_samples') if mode == 'train' else None
    processor = FileDataProcessor(
        context_map=context_map,
        is_synthetic=is_synthetic,
        cfg=cfg,
        mode=mode,
        limit_train_samples=limit_samples
    )
    raw_data = load_main_data_concurrently(
        directory=str(data_path),
        processor=processor,
        limit_samples=limit_samples
    )
    return raw_data

def _process_and_create_tensors(raw_data: dict, cfg: DictConfig) -> dict:
    """Converts data lists to tensors and creates task ID mappings."""
    unique_task_ids = sorted(set(raw_data["train_task_ids"] + raw_data["test_task_ids"]))
    task_id_map = {task_id: idx for idx, task_id in enumerate(unique_task_ids)}
    logger.info(f"Total unique task_ids: {len(unique_task_ids)}")

    # Handle cases where lists might be empty to avoid torch.stack errors
    target_shape = (cfg.model.max_h, cfg.model.max_w)
    train_inputs = torch.stack(raw_data["train_inputs"]) if raw_data["train_inputs"] else torch.empty((0, *target_shape), dtype=torch.long)
    train_outputs = torch.stack(raw_data["train_outputs"]) if raw_data["train_outputs"] else torch.empty((0, *target_shape), dtype=torch.long)
    test_inputs = torch.stack(raw_data["test_inputs"]) if raw_data["test_inputs"] else torch.empty((0, *target_shape), dtype=torch.long)
    test_outputs = torch.stack(raw_data["test_outputs"]) if raw_data["test_outputs"] else torch.empty((0, *target_shape), dtype=torch.long)

    train_task_ids_tensor = torch.tensor([task_id_map[tid] for tid in raw_data["train_task_ids"]], dtype=torch.long)
    test_task_ids_tensor = torch.tensor([task_id_map[tid] for tid in raw_data["test_task_ids"]], dtype=torch.long)

    # --- Variable-length context pairs (no padding, no stacking across samples) ---
    # Keep per-sample context tensors as lists to support dynamic number of pairs (2-6)
    train_ctx_inputs_list = [pair.context_input for pair in raw_data["train_context_pairs"]]
    train_ctx_outputs_list = [pair.context_output for pair in raw_data["train_context_pairs"]]
    test_ctx_inputs_list = [pair.context_input for pair in raw_data["test_context_pairs"]]
    test_ctx_outputs_list = [pair.context_output for pair in raw_data["test_context_pairs"]]

    # Basic length assertions (per-sample lists)
    assert train_inputs.size(0) == train_outputs.size(0) == train_task_ids_tensor.size(0) == len(train_ctx_inputs_list), "Mismatch in training data sizes."
    assert test_inputs.size(0) == test_outputs.size(0) == test_task_ids_tensor.size(0) == len(test_ctx_inputs_list), "Mismatch in testing data sizes."

    # --- V_colours Calculation ---
    # Calculate the maximum color value from all tensors to determine the vocabulary size.
    max_val = 0
    all_tensors = [train_inputs, train_outputs, test_inputs, test_outputs]
    # Extend with variable-length context tensors without stacking
    for t in (train_ctx_inputs_list + train_ctx_outputs_list + test_ctx_inputs_list + test_ctx_outputs_list):
        all_tensors.append(t)
    for t in all_tensors:
        if t.numel() > 0:
            max_val = max(max_val, t.max().item())
    
    # V_colours should be max_val + 1 to account for 0-based indexing.
    cfg.model.V_colours = max_val + 1
    cfg.model.vocab_size = cfg.model.V_colours  # Add this line
    logger.info(f"Set V_colours and vocab_size to: {cfg.model.V_colours}")

    # --- S_shapes Default --- 
    # Set a default value for the number of shapes. This is a placeholder
    # until a dynamic calculation from data is implemented.
    cfg.model.S_shapes = 11 # A common default for ARC-like problems
    logger.info(f"Set default S_shapes: {cfg.model.S_shapes}")

    return {
        "train_inputs": train_inputs, "train_outputs": train_outputs, "train_task_ids": train_task_ids_tensor,
        "train_ctx_inputs_list": train_ctx_inputs_list, "train_ctx_outputs_list": train_ctx_outputs_list,
        "test_inputs": test_inputs, "test_outputs": test_outputs, "test_task_ids": test_task_ids_tensor,
        "test_ctx_inputs_list": test_ctx_inputs_list, "test_ctx_outputs_list": test_ctx_outputs_list,
        "task_id_map": task_id_map
    }


def _generate_yardstick_datasets(cfg: DictConfig):
    """
    Generate synthetic multi-skill "yardstick" datasets and return the training dataset, evaluation dataset, a skill-to-index map, and the (possibly updated) configuration.
    
    The function builds small 3x3 grid tasks for a selectable subset of skills (A/B/C/D), creates per-sample single-pair contexts, adjusts cfg.model grid dimensions and vocabulary size where possible, and packages the data into DynamicContextDataset instances.
    
    Parameters:
        cfg (DictConfig): Configuration that may contain yardstick settings under `data`, including:
            - yardstick_skills (str): Skill letters to use, e.g. "ABCD" (default "ABCD").
            - yardstick_N (int): Number of training samples (default 256).
            - yardstick_M (int): Number of evaluation samples (default 64).
            - yardstick_exact_balanced (bool): If true, balance samples per skill (default True).
            - yardstick_randomize_ctx (bool): If true, randomize the context skill per sample (default False).
        The function may also update cfg.model.max_h, cfg.model.max_w, cfg.model.V_colours and cfg.model.vocab_size.
    
    Returns:
        tuple: (train_ds, eval_ds, task_id_map, cfg)
            - train_ds (DynamicContextDataset): Dataset for training samples.
            - eval_ds (DynamicContextDataset): Dataset for evaluation samples.
            - task_id_map (dict): Mapping from skill letter (e.g., "A") to its assigned integer index.
            - cfg (DictConfig): The (possibly modified) configuration object.
    """
    from jarc_reactor.data.datasets import DynamicContextDataset

    # ------------ Configuration (with safe fallbacks) ------------
    from omegaconf import OmegaConf
    skills_str = str(OmegaConf.select(cfg, 'data.yardstick_skills', default='ABCD'))
    skills = [c for c in skills_str if c in ['A', 'B', 'C', 'D']] or ['A', 'B', 'C', 'D']
    N = int(OmegaConf.select(cfg, 'data.yardstick_N', default=256))
    M = int(OmegaConf.select(cfg, 'data.yardstick_M', default=64))
    exbal = bool(OmegaConf.select(cfg, 'data.yardstick_exact_balanced', default=True))
    randctx = bool(OmegaConf.select(cfg, 'data.yardstick_randomize_ctx', default=False))

    # ------------ Colour IDs and helpers ------------
    BLACK, RED, BLUE, GREEN, YELLOW, CYAN, MAGENTA, ORANGE, GRAY = 0, 1, 2, 3, 4, 5, 6, 7, 8
    def one(v):
        return torch.ones(1, 3, 3, dtype=torch.long) * int(v)

    def make_ctx(skill: str):
        if skill == 'A':
            return one(RED), one(BLUE)
        if skill == 'B':
            return one(GREEN), one(YELLOW)
        if skill == 'C':
            return one(CYAN), one(MAGENTA)
        if skill == 'D':
            return one(ORANGE), one(GRAY)
        raise ValueError("Unknown skill")

    def rand_bin_grid():
        return torch.randint(low=0, high=2, size=(1, 3, 3), dtype=torch.long)

    def apply_rule(x: torch.Tensor, skill: str) -> torch.Tensor:
        if skill == 'A':
            return 1 - x
        if skill == 'B':
            return torch.rot90(x, k=1, dims=(1, 2))
        if skill == 'C':
            y = x.clone()
            y[:, 1:3, 1:3] = x[:, 0:2, 0:2]
            return y
        if skill == 'D':
            y = x.clone()
            y[y > 0] = BLUE
            return y
        raise ValueError("Unknown skill")

    # ------------ Build training samples ------------
    train_inputs_list = []
    train_outputs_list = []
    train_ctx_in_list = []
    train_ctx_out_list = []
    train_skill_ids = []
    # Map selected skills to indices (0, 1, 2, ...) based on actual skills being used
    skill_to_idx = {s: i for i, s in enumerate(skills)}

    if exbal:
        per_skill = max(1, N // max(1, len(skills)))
        quotas = [per_skill for _ in skills]
        rem = N - per_skill * len(skills)
        for i in range(rem):
            quotas[i % len(skills)] += 1
        for idx, s in enumerate(skills):
            for _ in range(quotas[idx]):
                x = rand_bin_grid()
                y = apply_rule(x, s)
                s_ctx = random.choice(skills) if randctx else s
                ci, co = make_ctx(s_ctx)
                train_inputs_list.append(x)
                train_outputs_list.append(y)
                train_ctx_in_list.append(ci)
                train_ctx_out_list.append(co)
                train_skill_ids.append(skill_to_idx[s])
    else:
        for _ in range(N):
            s = random.choice(skills)
            x = rand_bin_grid()
            y = apply_rule(x, s)
            s_ctx = random.choice(skills) if randctx else s
            ci, co = make_ctx(s_ctx)
            train_inputs_list.append(x)
            train_outputs_list.append(y)
            train_ctx_in_list.append(ci)
            train_ctx_out_list.append(co)
            train_skill_ids.append(skill_to_idx[s])

    # ------------ Build evaluation samples ------------
    eval_inputs_list = []
    eval_outputs_list = []
    eval_ctx_in_list = []
    eval_ctx_out_list = []
    eval_skill_ids = []
    for i in range(M):
        s = skills[i % len(skills)]
        x = rand_bin_grid()
        y = apply_rule(x, s)
        ci, co = make_ctx(s)
        eval_inputs_list.append(x)
        eval_outputs_list.append(y)
        eval_ctx_in_list.append(ci)
        eval_ctx_out_list.append(co)
        eval_skill_ids.append(skill_to_idx[s])

    # ------------ Stack to tensors (keep C=1 context) ------------
    train_inputs = torch.cat(train_inputs_list, dim=0)
    train_outputs = torch.cat(train_outputs_list, dim=0)
    train_ctx_inputs_list = [t for t in train_ctx_in_list]   # [N, 1, H, W] per item
    train_ctx_outputs_list = [t for t in train_ctx_out_list]
    train_task_ids = torch.tensor(train_skill_ids, dtype=torch.long)

    eval_inputs = torch.cat(eval_inputs_list, dim=0)
    eval_outputs = torch.cat(eval_outputs_list, dim=0)
    eval_ctx_inputs_list = [t for t in eval_ctx_in_list]
    eval_ctx_outputs_list = [t for t in eval_ctx_out_list]
    eval_task_ids = torch.tensor(eval_skill_ids, dtype=torch.long)

    # ------------ Adjust model vocab/grid dims safely ------------
    try:
        cfg.model.max_h = 3
        cfg.model.max_w = 3
    except Exception:
        pass

    # Compute max token over all tensors incl. contexts; ensure >= pad_token_id (10)
    max_val = 0
    for t in [train_inputs, train_outputs, eval_inputs, eval_outputs]:
        if t.numel() > 0:
            max_val = max(max_val, int(t.max().item()))
    for t in (train_ctx_inputs_list + train_ctx_outputs_list + eval_ctx_inputs_list + eval_ctx_outputs_list):
        if t.numel() > 0:
            max_val = max(max_val, int(t.max().item()))
    vocab = max(10, max_val) + 1
    try:
        cfg.model.V_colours = vocab
        cfg.model.vocab_size = vocab
    except Exception:
        pass

    # ------------ Create DynamicContextDataset ------------
    train_ds = DynamicContextDataset(
        train_inputs, train_outputs,
        train_ctx_inputs_list, train_ctx_outputs_list,
        train_task_ids,
    )
    eval_ds = DynamicContextDataset(
        eval_inputs, eval_outputs,
        eval_ctx_inputs_list, eval_ctx_outputs_list,
        eval_task_ids,
    )

    # Provide mapping for logging/parity with JSON path
    task_id_map = {s: skill_to_idx[s] for s in skills}
    return train_ds, eval_ds, task_id_map, cfg


def prepare_data(
    cfg: DictConfig,
    mode: str,
    return_datasets: bool = False,
    return_task_id_map: bool = False,
):
    """
    Prepare and return training, validation, or evaluation data according to configuration.
    
    Parameters:
        cfg (DictConfig): Configuration object containing data, dataloader, and model settings.
        mode (str): Operation mode, either 'train' or 'eval', which controls directory selection and return ordering.
        return_datasets (bool): If True, return dataset objects and the (possibly updated) config instead of DataLoaders.
        return_task_id_map (bool): If True, include the saved task_id_map as the last element of the returned payload.
    
    Returns:
        If `return_datasets` is True:
            A tuple containing (train_dataset, test_dataset, cfg) when mode is 'train', or (test_dataset, train_dataset, cfg) when mode is 'eval'.
            If `return_task_id_map` is True, the `task_id_map` dict is appended as the final element.
        If `return_datasets` is False:
            A tuple containing (train_loader, val_loader) when mode is 'train', or (val_loader, train_loader) when mode is 'eval'.
    
    Raises:
        ValueError: If `mode` is not 'train' or 'eval', or if no data is produced from the specified directory.
    """
    logger.info(f"prepare_data called in '{mode}' mode.")

    if mode == 'train':
        map_filename = 'task_id_map.json'
        base_dir = cfg.data.training_data_dir
    elif mode == 'eval':
        map_filename = 'eval_id_map.json'
        base_dir = cfg.data.testing_data_dir
    else:
        raise ValueError(f"Invalid mode: '{mode}'. Must be 'train' or 'eval'.")

    # ---------------- Yardstick Mode (synthetic benchmark) ----------------
    from omegaconf import OmegaConf
    if bool(OmegaConf.select(cfg, 'data.use_synthetic_yardstick', default=False)):
        logger.info("YARDSTICK MODE: Generating synthetic multi-skill dataset (no JSON files).")
        train_ds, eval_ds, task_id_map, cfg = _generate_yardstick_datasets(cfg)

        # Save a simple task map for auditability
        try:
            with open(map_filename, 'w') as f:
                json.dump(task_id_map, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save {map_filename}: {str(e)}")

        if return_datasets:
            logger.info("Returning TensorDatasets and the updated config (yardstick mode).")
            payload = (eval_ds, train_ds, cfg) if mode == 'eval' else (train_ds, eval_ds, cfg)
            if return_task_id_map:
                payload = payload + (task_id_map,)
            return payload

        batch_size = cfg.dataloader.batch_size
        val_batch_size = cfg.dataloader.get('val_batch_size', 1) if mode == 'train' else cfg.evaluation.get('batch_size', 1)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(eval_ds, batch_size=val_batch_size, shuffle=False, num_workers=0)
        logger.info(f"Prepared yardstick data for mode '{mode}'. Train size: {len(train_ds)}, Val/Test size: {len(eval_ds)}")
        if mode == 'eval':
            return val_loader, train_loader
        return train_loader, val_loader

    # Synthetic data handling (remains the same, assuming it's configured elsewhere)
    # This part of the logic seems independent of the main data directories.
    mode_cfg = cfg.get(mode, {})
    include_synthetic = mode_cfg.get(f'include_synthetic_{mode}_data', False)
    synthetic_dir = mode_cfg.get('synthetic_data_dir')

    is_synthetic = include_synthetic and synthetic_dir
    effective_data_dir = synthetic_dir if is_synthetic else base_dir
    logger.info(f"Mode '{mode}': Using {'synthetic' if is_synthetic else 'standard'} data from {effective_data_dir}")

    data_path = _validate_path(effective_data_dir)
    # Inspection is now handled within the loading process.

    raw_data = _load_raw_data(cfg, data_path, mode=mode, is_synthetic=is_synthetic)
    
    # Check if any data was produced before proceeding
    if not raw_data["train_inputs"] and not raw_data["test_inputs"]:
        raise ValueError("No data produced from the specified directory.")

    processed_data = _process_and_create_tensors(raw_data, cfg)

    from jarc_reactor.data.datasets import DynamicContextDataset
    train_dataset = DynamicContextDataset(
        processed_data["train_inputs"], processed_data["train_outputs"],
        processed_data["train_ctx_inputs_list"], processed_data["train_ctx_outputs_list"],
        processed_data["train_task_ids"]
    )
    test_dataset = DynamicContextDataset(
        processed_data["test_inputs"], processed_data["test_outputs"],
        processed_data["test_ctx_inputs_list"], processed_data["test_ctx_outputs_list"],
        processed_data["test_task_ids"]
    )

    task_id_map = processed_data["task_id_map"]
    logger.info(f"Saving task map to {map_filename} with {len(task_id_map)} mappings.")
    try:
        with open(map_filename, 'w') as f:
            json.dump(task_id_map, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save {map_filename}: {str(e)}")

    if return_datasets:
        logger.info("Returning TensorDatasets and the updated config.")
        payload = (test_dataset, train_dataset, cfg) if mode == 'eval' else (train_dataset, test_dataset, cfg)
        if return_task_id_map:
            payload = payload + (task_id_map,)
        return payload
    
    batch_size = cfg.dataloader.batch_size
    val_batch_size = cfg.dataloader.get('val_batch_size', 1) if mode == 'train' else cfg.evaluation.get('batch_size', 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=0)

    logger.info(f"Prepared data for mode '{mode}'. Train size: {len(train_dataset)}, Val/Test size: {len(test_dataset)}")
    # STRATEGIC LOGGING: Debug high loss mystery
    logger.critical(f"[DATA_DEBUG] mode={mode}, train_batch_size={batch_size}, "
                   f"val_batch_size={val_batch_size}, "
                   f"train_samples={len(train_dataset)}, val_samples={len(test_dataset)}")
    
    if mode == 'eval':
        # In eval mode, the test loader is the primary one, and the train loader is for context/validation
        logger.info("Eval mode: returning (test_loader, validation_loader)")
        return val_loader, train_loader
        
    return train_loader, val_loader