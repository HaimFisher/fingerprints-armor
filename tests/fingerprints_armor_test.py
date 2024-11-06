"""
This script tests the `SingleClassFingerprintsArmor` for adversarial defense on various classes by
running analysis and saving results.

Usage:
    Run the command below to execute the test:
    ```
    python3 -m unittest tests.fingerprints_armor_test
    ```

Expected Directory Structure:
- `ACTIVATIONS_DIR` in `settings` module should contain the original and attacked data in the following format:
    - Each class folder should have `orig` (original data) and `attack` (attacked data) subfolders.
"""

import os
from pathlib import Path
import unittest
import shutil
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.fingerprints_armor.fingerprints import SingleClassFingerprintsArmor
from src.fingerprints_armor.fingerprints_io import get_all_data_for_class
from src.logger_config import setup_logger, get_logger
import src.settings as settings

setup_logger()
logger = get_logger(__name__)
logger.info("Starting SingleClassFingerprintsArmor testing script")

# Set up output directory for results
OUTPUT_DIR = Path(settings.OUTPUT_DIR).resolve()
OUTPUT_DIR.mkdir(exist_ok=True)

# Load all available classes from ACTIVATIONS_DIR
all_classes = [cls for cls in os.listdir(settings.ACTIVATIONS_DIR) if not cls.startswith('.')]
logger.info(f"Classes found for processing: {all_classes}")

def analyze_single_class(class_str: str) -> None:
    """
    Performs adversarial analysis on a single class and saves the results to a CSV file.

    Args:
        class_str (str): The class identifier to analyze.
    """
    logger.info(f"Processing class: {class_str}")
    df_orig, _, attacked_to = get_all_data_for_class(
        class_str, get_attacked_from=False, limit_attacked_to=750
    )
    logger.info(f"Original data shape: {df_orig.shape}, Attacked data shape: {attacked_to.shape}")

    orig_train, orig_test = df_orig.iloc[:400], df_orig.iloc[400:]
    attack_train, attack_test = attacked_to.iloc[:400], attacked_to.iloc[400:]
    logger.info(f"Train/Test Split - Original Train: {orig_train.shape}, Original Test: {orig_test.shape}, "
          f"Attack Train: {attack_train.shape}, Attack Test: {attack_test.shape}")

    # Initialize and fit the Fingerprints Armor model
    fpt = SingleClassFingerprintsArmor(activations_per_fingerprint=50)
    fpt.fit(
        original_data=orig_train,
        adversarial_data=attack_train,
        num_samples=settings.N_FINGERPRINTS_SAMPLE,
        apply_significance_filter=True,
        p_value_threshold=0.001,
        min_effect_size=1.5,
        apply_top_filter=True,
        top_count=settings.K_TOP_TO_USE
    )
    logger.info(f"Number of fingerprints generated: {len(fpt._fingerprints)}")

    # Run predictions and save results
    y1 = fpt.vote(attack_test)
    y2 = fpt.vote(orig_test) 
    y_hat_vote = np.hstack((y1, y2))
    
    y1 = fpt.likelihood(attack_test)
    y2 = fpt.likelihood(orig_test) 
    y_hat_anomaly = np.hstack((y1, y2))
    
    y1 = fpt.likelihood_ratio(attack_test)
    y2 = fpt.likelihood_ratio(orig_test) 
    y_hat_ll_ratio = np.hstack((y1, y2))
    
    y = np.hstack((np.zeros_like(y1), np.ones_like(y2)))
    
    out = pd.DataFrame([y, y_hat_vote, y_hat_anomaly, y_hat_ll_ratio]).T
    out.columns = "y vote anomaly ll_ratio".split()
    out["cls"] = class_str
    out.to_csv(os.path.join(OUTPUT_DIR, f"{class_str}.csv"))
    logger.info(f"Results saved for class {class_str} to {OUTPUT_DIR}")

def aggregate_results(output_dir: Path) -> pd.DataFrame:
    """
    Aggregates results from individual class analysis and returns a DataFrame.

    Args:
        output_dir (Path): Path to the directory where results are stored.

    Returns:
        pd.DataFrame: Aggregated results DataFrame.
    """
    data_files = [pd.read_csv(f) for f in output_dir.glob("*.csv") if f.is_file()]
    return pd.concat(data_files, ignore_index=True)


def calculate_confusion_matrix(data: pd.DataFrame, threshold_column: str) -> pd.DataFrame:
    """
    Calculates the confusion matrix for a specified threshold column.

    Args:
        data (pd.DataFrame): Data containing ground truth and predictions.
        threshold_column (str): Column name used for thresholding predictions.

    Returns:
        pd.DataFrame: Normalized confusion matrix.
    """
    y_hat = data[threshold_column] > data[data.y == 1][threshold_column].quantile(0.01)
    return pd.DataFrame(confusion_matrix(y_true=data.y, y_pred=y_hat, labels=[0, 1], normalize="true"))

class TestFingerprintsMethode(unittest.TestCase):
    def test_fingerprints(self):
        for cls in all_classes:
            analyze_single_class(cls)

        # Aggregate and analyze the final results
        aggregated_data = aggregate_results(OUTPUT_DIR)
        
        for metric in ["vote", "anomaly", "ll_ratio"]:
            conf_matrix = calculate_confusion_matrix(aggregated_data, metric)
            logger.info(f"{metric}_conf_matrix:\n{conf_matrix}")

        shutil.rmtree(OUTPUT_DIR)

if __name__ == "__main__":
    unittest.main()