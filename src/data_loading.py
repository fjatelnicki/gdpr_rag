import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_article_summaries(articles_summaries_path):
    with open(articles_summaries_path, "r") as f:
        return json.load(f)


def load_summary_embeddings(embeddings_path: Path) -> Dict[str, List[float]]:
    return np.load(embeddings_path, allow_pickle=True).item()
