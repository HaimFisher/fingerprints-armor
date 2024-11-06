"""
This module handles the input and output operations for processing the activations database.

The expected structure of the input database, which should be created by the `db_creator` module, is as follows:

database/
├── category_1/
│   ├── orig/
│   │   ├── image_1.pkl
│   │   └── ...
│   └── attack/
│       ├── image_1_0_to_1_ifgsm_3_01.pkl
│       └── ...
└── ...

### Input Database Format:
- **category_n/**: A folder for each class category.
- **orig/**: Contains original images activations in `.pkl` format.
- **attack/**: Contains attacked images from current class to other class activations following a specific naming format.

### Naming Format for Attack Files:
Each attack file is named according to the following convention:
```python
file_name = (
    f"{image_name}_{folder_idx}_to_{top1_catid[0]}_ifgsm_{attack_iter}_{epsilon}_"
    f"{top1_prob[0].item():.2f}.pkl"
)

 - image_name: The name of the original image.
 - folder_idx: An index representing the specific folder or category.
 - top1_catid[0]: The ID of the top predicted category.
 - attack_iter: The number of iterations for the attack.
 - epsilon: The perturbation value used in the attack.
 - top1_prob[0].item():.2f: The probability associated with the top predicted category, formatted to two decimal places.

This format ensures consistency in data handling and facilitates easy access to the relevant information needed for further processing.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle

import pandas as pd

import src.settings as settings


def generate_class_string_index_map() -> pd.Series:
    """
    Generates a mapping between class strings and their corresponding indices.

    Scans the activations directory and creates a mapping between class string codes
    and their numerical indices extracted from filenames.

    Returns:
        pd.Series: A pandas Series where index is the class string code and value
            is the corresponding numerical index extracted from the filename.
    """
    return pd.Series({
        class_label: re.search(r'_(\d+)_to_', os.path.splitext(os.listdir(os.path.join(settings.ACTIVATIONS_DIR, class_label, "attack"))[0])[0]).group(1)
        for class_label in os.listdir(settings.ACTIVATIONS_DIR) if not class_label.startswith(".")
    })


def get_list_attacked_to(target_class: str) -> List[str]:
    """
    Retrieves a list of files that were attacked to become the target class.

    Args:
        target_class (str): The target class string identifier to search for.

    Returns:
        List[str]: A list of full file paths for images that were attacked to 
            become the target class.
    """
    target_index = generate_class_string_index_map()[target_class]

    return [
        str(file_path) 
        for class_dir in os.listdir(settings.ACTIVATIONS_DIR) if not class_dir.startswith(".")
        for file_path in Path(settings.ACTIVATIONS_DIR, class_dir, "attack").glob(f"*to_{target_index}*")
    ]


def get_data_by_file_name_list(files_paths: List[str]) -> pd.DataFrame:
    """
    Loads pickle data from a list of files into a DataFrame.

    Args:
        files_paths (List[str]): List of file paths to load data from.

    Returns:
        pd.DataFrame: DataFrame containing the loaded pickle data, with file paths
            as index.
    """
    data = []
    paths = []

    for filepath in files_paths:
        if Path(filepath).stat().st_size > 0:
            with open(filepath, 'rb') as f:
                data.append(pickle.load(f))
                paths.append(filepath)
    
    return pd.DataFrame(data, index=files_paths)


def get_all_data_for_class(
    class_str: str, 
    get_attacked_from: bool = True,
    limit_attacked_to: Optional[int] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """
    Retrieves all relevant data for a specific class.

    This function loads three types of data:
    1. Activations data of the class original images
    2. Activations data of attacks FROM this class to other classes (optional)
    3. Activations data of attacks TO this class from other classes

    Args:
        class_str (str): The class string identifier to get data for.
        get_attacked_from (bool, optional): Whether to load data for attacks 
            originating from this class. Defaults to True.
        limit_attacked_to (int, optional): Limit the number of 'attacked to' samples
            to load. Defaults to None (no limit).

    Returns:
        Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]: A tuple containing:
            - DataFrame with original class data
            - DataFrame with 'attacked from' data (None if get_attacked_from=False)
            - DataFrame with 'attacked to' data
    """
    # Load original data
    orig_path = Path(settings.ACTIVATIONS_DIR, class_str, "orig")
    orig_files = [str(f) for f in orig_path.glob("[!.]*")]
    df_orig = get_data_by_file_name_list(orig_files)

    # Load attacked FROM data
    attacked_from = None
    if get_attacked_from:
        attack_path = Path(settings.ACTIVATIONS_DIR, class_str, "attack")
        attack_files = [str(f) for f in attack_path.glob("[!.]*")]
        attacked_from = get_data_by_file_name_list(attack_files)
    
    # Load attacked TO data
    attacked_to_files = get_list_attacked_to(class_str)
    if limit_attacked_to:
        attacked_to_files = attacked_to_files[:limit_attacked_to]
    attacked_to = get_data_by_file_name_list(attacked_to_files)

    return df_orig, attacked_from, attacked_to
