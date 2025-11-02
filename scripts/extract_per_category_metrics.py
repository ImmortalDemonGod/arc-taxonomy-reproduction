#!/usr/bin/env python3
"""Extract per-category classification metrics from Optuna study."""

import optuna
import pandas as pd
import json
from pathlib import Path

STORAGE_URL = "***REMOVED***"
STUDY_NAME = "visual_classifier_cnn_vs_context_v2_expanded"

def main():
    print(f"Loading study: {STUDY_NAME}")
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)
    
    print(f"Total trials: {len(study.trials)}")
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Completed trials: {len(completed_trials)}")
    
    if not completed_trials:
        print("No completed trials found!")
        return
    
    # Get best trial
    best_trial = study.best_trial
    print(f"\nBest trial: #{best_trial.number}")
    print(f"Best validation accuracy: {best_trial.value:.4f}")
    print(f"\nBest trial params:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    print(f"\nBest trial user attributes (per-category metrics):")
    if best_trial.user_attrs:
        for key, value in best_trial.user_attrs.items():
            if 'category' in key.lower() or 'per_' in key.lower():
                print(f"  {key}: {value}")
    else:
        print("  No user attributes found")
    
    # Extract all trials data
    trials_data = []
    for trial in completed_trials:
        row = {
            'trial_number': trial.number,
            'val_acc': trial.value,
            **trial.params
        }
        # Add user attributes
        if trial.user_attrs:
            for key, value in trial.user_attrs.items():
                row[key] = value
        trials_data.append(row)
    
    df = pd.DataFrame(trials_data)
    
    # Save to CSV
    output_file = Path("outputs") / "visual_classifier_trials_full.csv"
    output_file.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_file, index=False)
    print(f"\nSaved all trials data to: {output_file}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for per-category columns
    category_cols = [col for col in df.columns if 'category' in col.lower() or col in ['A1', 'A2', 'C1', 'C2', 'S1', 'S2', 'S3', 'K1', 'L1']]
    if category_cols:
        print(f"\nPer-category metric columns found: {category_cols}")
        print(f"\nBest trial per-category performance:")
        best_row = df[df['trial_number'] == best_trial.number].iloc[0]
        for col in category_cols:
            if col in best_row:
                print(f"  {col}: {best_row[col]}")
    else:
        print("\nNo per-category metrics found in user_attrs")
        print("Available columns:", list(df.columns))

if __name__ == "__main__":
    main()
