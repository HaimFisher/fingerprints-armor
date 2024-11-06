"""
run cmd: python3 -m unittest tests.db_creator_test
"""

import os
import shutil
import unittest
from src.db_creator.db_creator import main
import src.settings as settings


class TestCreateDB(unittest.TestCase):
    def test_create_db(self):
        if os.path.exists(settings.TARGET_DIRECTORY):
            shutil.rmtree(settings.TARGET_DIRECTORY)

        main()
        
if __name__ == "__main__":
    unittest.main()