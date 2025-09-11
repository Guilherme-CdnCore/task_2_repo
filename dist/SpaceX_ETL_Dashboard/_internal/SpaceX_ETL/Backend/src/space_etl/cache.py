import os
import json 
from typing import Any 
import logging
from .exceptions_file import DataLoadError

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_cache(cache_path: str) -> Any:
    try: 
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading cache from {cache_path}: {e}")
        raise DataLoadError(f"Error loading cache from {cache_path}: {e}")
    finally:
        logger.info("Cache loading process completed.")


def save_cache(cache_path: str, data: Any) -> None:
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving cache to {cache_path}: {e}")
        raise DataLoadError(f"Error saving cache to {cache_path}: {e}")
    finally:
        logger.info("Cache saving process completed.")






def is_unchanged(content_hash:str, cache: dict, key: str) -> bool:
    return cache.get(key) == content_hash





def update_cache(content_hash: str, cache: dict, key: str) -> None:
    cache[key] = content_hash
    logger.debug(f"Cache updated for key: {key}")
    return cache
