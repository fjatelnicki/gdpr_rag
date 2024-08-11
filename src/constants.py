# Regex patterns
import re

PAGINATION_PATTERN = re.compile(r"\bpage\s+(\d+)\s*/\s*(\d+)\b", re.IGNORECASE)
ARTICLE_NUMBER_PATTERN = re.compile(r"^Article\s+(\d+)")
LEGAL_NUMBER_PATTERN = re.compile(r"(?:[A-Z]-\d+/\d+|\d{4}/\d+|WP\d+(?:\s?rev\.\d+)?)")

LINK1 = "GDPR training, consulting and DPO outsourcing\nwww.data-privacy-\noffice.eu\nwww.gdpr-text.com\ninfo@data-privacy-\noffice.eu"
LINK2 = "www.gdpr-text.com/en"

custom_stopwords = [
    "gdpr",
    "training",
    "consulting",
    "dpo",
    "outsourcing",
    "article",
    "articles",
    "information",
]

EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"

# File paths
PDF_PATH = "data/gdpr_articles.pdf"
SUMMARY_EMBEDDINGS_PATH = "data/summary_embeddings.npy"
SUMMARIES_PATH = "data/articles_summaries.json"
VECTORSTORE_DIR = "data/vectorstore"

# Constants
EXPECTED_ARTICLE_COUNT = 21
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
NUMBER_OF_PARTS_TO_RETRIEVE = 60

# Scores
KEYWORD_MATCH_SCORE = 0.3
SUMMARY_SIMILARITY_SCORE = 0.5
ARTICLE_MENTION_SCORE = 0.3


ascii_art = """
  ____ ____  ____  ____    ____    _    ____ 
 / ___|  _ \|  _ \|  _ \  |  _ \  / \  / ___|
| |  _| | | | |_) | |_) | | |_) |/ _ \| |  _ 
| |_| | |_| |  __/|  _ <  |  _ </ ___ \ |_| |
 \____|____/|_|   |_| \_\ |_| \_\_/  \_\____|

Welcome to the GDPR Articles RAG System!
"""

prompt_template = """You are an AI assistant answering questions about the GDPR. Each chunk of text is preceded by a LOCATION field in the format [Article X | Page Y]. This LOCATION is the official source of the information. ALWAYS use the LOCATION field to cite sources, not any mentions within the text content.

Context:
{context}

Question: {question}

Answer: Provide answer here"""
