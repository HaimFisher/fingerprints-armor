import os

SOURCE_DIRECTORY: str = 'tests/test_db/in'
TARGET_DIRECTORY: str = 'tests/test_db/out'

INPUT_DB_CLASSES_COUNT: int = 2
"""
Number of classes in the input database.
(default) number of classes in test database.
"""

ACTIVATIONS_DIR: str = TARGET_DIRECTORY
""" 
Directory where the activations database is found.
"""

OUTPUT_DIR: str = os.path.abspath("./output_files")
""" 
Directory to store output files of fingerprints methode generated during testing.
"""

N_FINGERPRINTS_SAMPLE: int = 25
"""
Number of fingerprints to sample for each class during evaluation.
"""

K_TOP_TO_USE: int = 20
"""
Number of top fingerprint to use after apply filters on the fingerprints.
"""
