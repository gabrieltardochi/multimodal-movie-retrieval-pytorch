import os

from dotenv import load_dotenv

load_dotenv(".env")  # take environment variables from .env

TINY_MMIMDB_DATASET_PATH = os.environ["TINY_MMIMDB_DATASET_PATH"]
TRAINED_MODEL_STATE_DICT_PATH = os.getenv("TRAINED_MODEL_STATE_DICT_PATH", None)
