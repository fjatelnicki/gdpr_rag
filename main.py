import sys
from pathlib import Path
import logging
import os
import warnings
import argparse
from src.data_preparation import DataPreparation
from src.rag_model import RAGModel
from src.console_interface import run_console_interface
from src.vectorization import SegregatedVectorStore

src_dir = Path(__file__).resolve().parent / "src"
sys.path.append(str(src_dir))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.tokenization_utils_base"
)
for logger_name in ["httpx", "sentence_transformers", "chromadb"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def main():
    parser = argparse.ArgumentParser(description="GDPR Articles RAG System")
    parser.add_argument("--hide-documents", action="store_true", help="Hide relevant documents, show only answers")
    args = parser.parse_args()

    data_preparation = DataPreparation()
    documents = data_preparation.prepare_documents()
    segregated_vector_store = SegregatedVectorStore(documents)
    rag_model = RAGModel(segregated_vector_store)

    run_console_interface(rag_model, documents, hide_documents=args.hide_documents)


if __name__ == "__main__":
    main()