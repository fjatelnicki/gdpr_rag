from typing import List, Tuple, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

from src import constants
from src.vectorization import SegregatedVectorStore


class Retriever:
    def __init__(self, segregated_vector_store: SegregatedVectorStore):
        self.segregated_vector_store = segregated_vector_store
        self.embeddings = HuggingFaceEmbeddings(model_name=constants.EMBEDDINGS_MODEL)

    def get_relevant_documents(
        self, query: str, documents: List[Document], top_k: int = 3
    ) -> Tuple[List[Tuple[Document, float]], List[Dict[str, Any]]]:
        query_embedding = self.embeddings.embed_query(query)

        relevant_articles, article_scores = self._select_relevant_articles(
            query, query_embedding, documents, top_k
        )

        relevant_docs = self._select_relevant_parts(
            query, relevant_articles, top_k, article_scores
        )

        if not relevant_docs:
            return [], article_scores

        return relevant_docs, article_scores

    def _select_relevant_articles(
        self,
        query: str,
        query_embedding: np.ndarray,
        documents: List[Document],
        top_k: int,
    ) -> Tuple[List[List[Document]], List[Dict[str, Any]]]:

        scored_articles = self._calculate_article_scores(
            query, query_embedding, documents
        )
        top_articles = self._get_top_k_items(
            scored_articles, top_k, key=lambda x: x["score"]
        )

        selected_article_numbers = [
            str(article["article_number"]) for article in top_articles
        ]

        relevant_articles = [
            [
                doc
                for doc in documents
                if str(doc.metadata["article_number"]) == article_number
            ]
            for article_number in selected_article_numbers
        ]

        return relevant_articles, top_articles

    def _calculate_article_scores(
        self, query: str, query_embedding: np.ndarray, documents: List[Document]
    ) -> List[Dict[str, Any]]:
        scored_articles = []

        unique_articles = {
            (
                doc.metadata["article_number"],
                doc.metadata["article_summary"],
                doc.metadata.get("keywords", ""),
            )
            for doc in documents
        }

        for article_number, summary, keywords in unique_articles:
            keywords = [k.strip().lower() for k in keywords.split(",") if k.strip()]

            summary_embedding = self.embeddings.embed_query(summary)
            summary_similarity = cosine_similarity(
                [query_embedding], [summary_embedding]
            )[0][0]

            score = summary_similarity * constants.SUMMARY_SIMILARITY_SCORE
            reasons = [f"Query-Summary similarity: {summary_similarity:.2f}"]

            if f"article {article_number}" in query.lower():
                score += constants.ARTICLE_MENTION_SCORE
                reasons.append(f"Exact article {article_number} mention")

            keyword_matches = []
            query_words = query.lower().split()
            for word in query_words:
                if word in keywords:
                    score += constants.KEYWORD_MATCH_SCORE
                    keyword_matches.append(word)
            if keyword_matches:
                reasons.append(f"Keyword matches: {', '.join(keyword_matches)}")

            scored_articles.append(
                {
                    "article_number": article_number,
                    "score": score,
                    "reasons": reasons,
                    "summary_similarity": summary_similarity,
                    "keyword_matches": keyword_matches,
                }
            )

        return scored_articles

    def _select_relevant_parts(
        self,
        query: str,
        relevant_articles: List[List[Document]],
        top_k: int,
        article_scores: List[Dict[str, Any]],
    ) -> List[Tuple[Document, float]]:
        article_numbers = [
            str(article_list[0].metadata["article_number"])
            for article_list in relevant_articles
        ]

        results = self.segregated_vector_store.search(
            query,
            article_numbers=article_numbers,
            top_k=constants.NUMBER_OF_PARTS_TO_RETRIEVE,
        )

        scored_parts = []

        for doc, similarity in results:
            article_number = str(doc.metadata.get("article_number"))
            article_score = next(
                (
                    a["score"]
                    for a in article_scores
                    if str(a["article_number"]) == article_number
                ),
                0,
            )

            combined_score = similarity + article_score
            scored_parts.append((doc, combined_score))

        selected_parts = self._get_top_k_items(scored_parts, top_k, key=lambda x: x[1])
        return selected_parts

    @staticmethod
    def _get_top_k_items(items: List[Any], k: int, key: callable) -> List[Any]:
        return sorted(items, key=key, reverse=True)[:k]
