import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import snapshot_download
from loguru import logger


def load_dataset_from_hf(repo_id, local_dir):
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        repo_type="dataset",
        tqdm_class=None,
        etag_timeout=30,
        token=os.environ["HF_TOKEN"],
    )
    return load_dataset(repo_id)


class CacheDB:
    """Handles database operations for storing and retrieving cache entries."""

    def __init__(self, db_path: Path):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.lock = threading.Lock()

        # Initialize the database
        try:
            self.initialize_db()
        except Exception as e:
            logger.exception(f"Failed to initialize database: {e}")
            logger.warning(f"Please provide a different filepath or remove the file at {self.db_path}")
            raise

    def initialize_db(self) -> None:
        """Initialize SQLite database with the required table."""
        # Check if database file already exists
        if self.db_path.exists():
            self._verify_existing_db()
        else:
            self._create_new_db()

    def _verify_existing_db(self) -> None:
        """Verify and repair an existing database if needed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                self._ensure_table_exists(cursor)
                self._verify_schema(cursor)
                self._ensure_index_exists(cursor)
                conn.commit()
            logger.info(f"Using existing SQLite database at {self.db_path}")
        except Exception as e:
            logger.exception(f"Database corruption detected: {e}")
            raise ValueError(f"Corrupted database at {self.db_path}: {str(e)}")

    def _create_new_db(self) -> None:
        """Create a new database with the required schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                self._create_table(cursor)
                self._ensure_index_exists(cursor)
                conn.commit()
                logger.info(f"Initialized new SQLite database at {self.db_path}")
        except Exception as e:
            logger.exception(f"Failed to initialize SQLite database: {e}")
            raise

    def _ensure_table_exists(self, cursor) -> None:
        """Check if the llm_cache table exists and create it if not."""
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='llm_cache'")
        if not cursor.fetchone():
            self._create_table(cursor)
            logger.info("Created missing llm_cache table")

    def _create_table(self, cursor) -> None:
        """Create the llm_cache table with the required schema."""
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            key TEXT PRIMARY KEY,
            request_json TEXT,
            response_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

    def _verify_schema(self, cursor) -> None:
        """Verify that the table schema has all required columns."""
        cursor.execute("PRAGMA table_info(llm_cache)")
        columns = {row[1] for row in cursor.fetchall()}
        required_columns = {"key", "request_json", "response_json", "created_at"}

        if not required_columns.issubset(columns):
            missing = required_columns - columns
            raise ValueError(f"Database schema is corrupted. Missing columns: {missing}")

    def _ensure_index_exists(self, cursor) -> None:
        """Create an index on the key column for faster lookups."""
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_llm_cache_key ON llm_cache (key)")

    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Get cached entry by key.

        Args:
            key: Cache key to look up

        Returns:
            Dict containing the request and response or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT request_json, response_json FROM llm_cache WHERE key = ?", (key,))
                result = cursor.fetchone()

                if result:
                    logger.debug(f"Cache hit for key: {key}. Response: {result['response_json']}")
                    return {
                        "request": result["request_json"],
                        "response": result["response_json"],
                    }

                logger.debug(f"Cache miss for key: {key}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None

    def set(self, key: str, request_json: str, response_json: str) -> bool:
        """Set entry in cache.

        Args:
            key: Cache key
            request_json: JSON string of request parameters
            response_json: JSON string of response

        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT OR REPLACE INTO llm_cache (key, request_json, response_json) VALUES (?, ?, ?)",
                        (key, request_json, response_json),
                    )
                    conn.commit()
                    logger.debug(f"Saved response to cache with key: {key}, response: {response_json}")
                    return True
            except Exception as e:
                logger.error(f"Failed to save to SQLite cache: {e}")
                return False

    def get_all_entries(self) -> dict[str, dict[str, Any]]:
        """Get all cache entries from the database."""
        cache = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT key, request_json, response_json FROM llm_cache ORDER BY created_at")

                for row in cursor.fetchall():
                    cache[row["key"]] = {
                        "request": row["request_json"],
                        "response": row["response_json"],
                    }

                logger.debug(f"Retrieved {len(cache)} entries from cache database")
                return cache
        except Exception as e:
            logger.error(f"Error retrieving all cache entries: {e}")
            return {}

    def clear(self) -> bool:
        """Clear all cache entries.

        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM llm_cache")
                    conn.commit()
                    logger.info("Cache cleared")
                    return True
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
                return False

    def get_existing_keys(self) -> set:
        """Get all existing keys in the database.

        Returns:
            Set of keys
        """
        existing_keys = set()
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT key FROM llm_cache")
                for row in cursor.fetchall():
                    existing_keys.add(row[0])
                return existing_keys
        except Exception as e:
            logger.error(f"Error retrieving existing keys: {e}")
            return set()

    def bulk_insert(self, items: list, update: bool = False) -> int:
        """Insert multiple items into the cache.

        Args:
            items: List of (key, request_json, response_json) tuples
            update: Whether to update existing entries

        Returns:
            Number of items inserted
        """
        count = 0
        UPDATE_OR_IGNORE = "INSERT OR REPLACE" if update else "INSERT OR IGNORE"
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.executemany(
                        f"{UPDATE_OR_IGNORE} INTO llm_cache (key, request_json, response_json) VALUES (?, ?, ?)",
                        items,
                    )
                    count = cursor.rowcount
                    conn.commit()
                return count
            except Exception as e:
                logger.error(f"Error during bulk insert: {e}")
                return 0


class LLMCache:
    def __init__(
        self, cache_dir: str = ".", hf_repo: str | None = None, cache_sync_interval: int = 3600, reset: bool = False
    ):
        self.cache_dir = Path(cache_dir)
        self.db_path = self.cache_dir / "llm_cache.db"
        self.hf_repo_id = hf_repo
        self.cache_sync_interval = cache_sync_interval
        self.last_sync_time = time.time()

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Initialize CacheDB
        self.db = CacheDB(self.db_path)
        if reset:
            self.db.clear()

        # Try to load from HF dataset if available
        try:
            self._load_cache_from_hf()
        except Exception as e:
            logger.warning(f"Failed to load cache from HF dataset: {e}")

    def response_format_to_dict(self, response_format: Any) -> dict[str, Any]:
        """Convert a response format to a dict."""
        # If it's a Pydantic model, use its schema
        if hasattr(response_format, "model_json_schema"):
            response_format = response_format.model_json_schema()

        # If it's a Pydantic model, use its dump
        elif hasattr(response_format, "model_dump"):
            response_format = response_format.model_dump()

        if not isinstance(response_format, dict):
            response_format = {"value": str(response_format)}

        return response_format

    def _generate_key(
        self, model: str, system: str, prompt: str, response_format: Any, temperature: float | None = None
    ) -> str:
        """Generate a unique key for caching based on inputs."""
        response_format_dict = self.response_format_to_dict(response_format)
        response_format_str = json.dumps(response_format_dict, sort_keys=True)
        # Include temperature in the key
        key_content = f"{model}:{system}:{prompt}:{response_format_str}"
        if temperature is not None:
            key_content += f":{temperature:.2f}"
        return hashlib.md5(key_content.encode()).hexdigest()

    def _create_request_json(
        self, model: str, system: str, prompt: str, response_format: Any, temperature: float | None
    ) -> str:
        """Create JSON string from request parameters."""
        request_data = {
            "model": model,
            "system": system,
            "prompt": prompt,
            "response_format": self.response_format_to_dict(response_format),
            "temperature": temperature,
        }
        return json.dumps(request_data)

    def _check_request_match(
        self,
        cached_request: dict[str, Any],
        model: str,
        system: str,
        prompt: str,
        response_format: Any,
        temperature: float | None,
    ) -> bool:
        """Check if the cached request matches the new request."""
        # Check each field and log any mismatches
        if cached_request["model"] != model:
            logger.debug(f"Cache mismatch: model - cached: {cached_request['model']}, new: {model}")
            return False
        if cached_request["system"] != system:
            logger.debug(f"Cache mismatch: system - cached: {cached_request['system']}, new: {system}")
            return False
        if cached_request["prompt"] != prompt:
            logger.debug(f"Cache mismatch: prompt - cached: {cached_request['prompt']}, new: {prompt}")
            return False
        response_format_dict = self.response_format_to_dict(response_format)
        if cached_request["response_format"] != response_format_dict:
            logger.debug(
                f"Cache mismatch: response_format - cached: {cached_request['response_format']}, new: {response_format_dict}"
            )
            return False
        if cached_request["temperature"] != temperature:
            logger.debug(f"Cache mismatch: temperature - cached: {cached_request['temperature']}, new: {temperature}")
            return False

        return True

    def get(
        self, model: str, system: str, prompt: str, response_format: dict[str, Any], temperature: float | None = None
    ) -> Optional[dict[str, Any]]:
        """Get cached response if it exists."""
        key = self._generate_key(model, system, prompt, response_format, temperature)
        result = self.db.get(key)

        if not result:
            return None
        request_dict = json.loads(result["request"])
        if not self._check_request_match(request_dict, model, system, prompt, response_format, temperature):
            logger.warning(f"Cached request does not match new request for key: {key}")
            return None

        return json.loads(result["response"])

    def set(
        self,
        model: str,
        system: str,
        prompt: str,
        response_format: dict[str, Any],
        temperature: float | None,
        response: dict[str, Any],
    ) -> None:
        """Set response in cache and sync if needed."""
        key = self._generate_key(model, system, prompt, response_format, temperature)
        request_json = self._create_request_json(model, system, prompt, response_format, temperature)
        response_json = json.dumps(response)

        success = self.db.set(key, request_json, response_json)

        # Check if we should sync to HF
        if success and self.hf_repo_id and (time.time() - self.last_sync_time > self.cache_sync_interval):
            try:
                self.sync_to_hf()
                self.last_sync_time = time.time()
            except Exception as e:
                logger.error(f"Failed to sync cache to HF dataset: {e}")

    def _load_cache_from_hf(self) -> None:
        """Load cache from HF dataset if it exists and merge with local cache."""
        if not self.hf_repo_id:
            return

        try:
            # Check for new commits before loading the dataset
            ds_path = (self.cache_dir / "hf_cache").as_posix()
            dataset = load_dataset_from_hf(self.hf_repo_id, ds_path)["train"]
            if not dataset:
                logger.info("No new items to merge from HF dataset")
                return

            existing_keys = self.db.get_existing_keys()

            logger.info(f"Found {len(dataset)} items in HF dataset. Existing keys: {len(existing_keys)}")

            # Prepare batch items for insertion
            items_to_insert = []
            for item in dataset:
                key = item["key"]
                # Only update if not in local cache to prioritize local changes
                if key in existing_keys:
                    continue
                # Create request JSON
                request_data = {
                    "model": item["model"],
                    "system": item["system"],
                    "prompt": item["prompt"],
                    "temperature": item["temperature"],
                    "response_format": None,  # We can't fully reconstruct this
                }

                items_to_insert.append(
                    (
                        key,
                        json.dumps(request_data),
                        item["response"],  # This is already a JSON string
                    )
                )

            # Bulk insert new items
            if items_to_insert:
                inserted_count = self.db.bulk_insert(items_to_insert)
                logger.info(f"Merged {inserted_count} items from HF dataset into SQLite cache")
            else:
                logger.info("No new items to merge from HF dataset")
        except Exception as e:
            logger.warning(f"Could not load cache from HF dataset: {e}")

    def get_all_entries(self) -> dict[str, dict[str, Any]]:
        """Get all cache entries from the database."""
        cache = self.db.get_all_entries()
        entries = {}
        for key, entry in cache.items():
            request = json.loads(entry["request"])
            response = json.loads(entry["response"])
            entries[key] = {"request": request, "response": response}
        return entries

    def sync_to_hf(self) -> None:
        """Sync cache to HF dataset."""
        if not self.hf_repo_id:
            return

        self._load_cache_from_hf()

        # Get all entries from the database
        cache = self.db.get_all_entries()

        # Convert cache to dataset format
        entries = []
        for key, entry in cache.items():
            request = json.loads(entry["request"])
            response_str = entry["response"]
            entries.append(
                {
                    "key": key,
                    "model": request["model"],
                    "system": request["system"],
                    "prompt": request["prompt"],
                    "response_format": request["response_format"],
                    "temperature": request["temperature"],
                    "response": response_str,
                }
            )

        # Create and push dataset
        dataset = Dataset.from_list(entries)
        dataset.push_to_hub(self.hf_repo_id, private=True)
        logger.info(f"Synced {len(cache)} cached items to HF dataset {self.hf_repo_id}")

    def clear(self) -> None:
        """Clear all cache entries."""
        self.db.clear()
