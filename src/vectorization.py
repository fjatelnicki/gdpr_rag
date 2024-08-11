from langchain_huggingface import HuggingFaceEmbeddings
from . import constants
from langchain_community.vectorstores import Chroma


class SegregatedVectorStore:
    def __init__(self, documents):
        # Initialize the embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name=constants.EMBEDDINGS_MODEL)
        self.article_stores = {}

        # Group documents by article number
        grouped_docs = {}
        for doc in documents:
            article_number = str(
                doc.metadata.get("article_number")
            )  # Convert to string
            if article_number not in grouped_docs:
                grouped_docs[article_number] = []
            grouped_docs[article_number].append(doc)

        # Create a Chroma vector store for each article
        from tqdm import tqdm

        for article_number, docs in tqdm(
            grouped_docs.items(), desc="Creating vector stores"
        ):
            try:
                self.article_stores[article_number] = Chroma.from_documents(
                    docs, self.embeddings, collection_name=f"article_{article_number}"
                )
            except Exception:
                raise Exception(
                    f"Failed to create vector store for article {article_number}"
                )

    def search(self, query, article_numbers=None, top_k=3):
        # If no specific articles are provided, search all articles
        if article_numbers is None:
            article_numbers = list(self.article_stores.keys())

        results = []
        # Search each specified article's vector store
        for article_number in article_numbers:
            if article_number in self.article_stores:
                article_results = self.article_stores[
                    article_number
                ].similarity_search_with_score(query, k=top_k)
                results.extend(article_results)

        # Sort results by score (descending) and return top k
        results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
        return results
