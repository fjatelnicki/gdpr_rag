from .retriever import Retriever
from .generator import Generator


class RAGModel:
    def __init__(self, segregated_vector_store):
        self.retriever = Retriever(segregated_vector_store)
        self.generator = Generator()

    def answer_query(self, query, documents):
        relevant_docs, article_scores = self.retriever.get_relevant_documents(
            query, documents
        )
        answer = self.generator.generate_answer(query, relevant_docs)
        return answer, relevant_docs, article_scores
