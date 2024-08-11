import unittest
from src.retriever import Retriever
from src.vectorization import SegregatedVectorStore
from src.data_preparation import DataPreparation

class TestRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_preparation = DataPreparation()
        cls.documents = data_preparation.prepare_documents()
        vector_store = SegregatedVectorStore(cls.documents)
        cls.retriever = Retriever(vector_store)

    def test_get_relevant_documents(self):
        query = "What are the rights of data subjects?"
        relevant_docs, article_scores = self.retriever.get_relevant_documents(query, self.documents)
        self.assertIsInstance(relevant_docs, list)
        self.assertIsInstance(article_scores, list)
        self.assertTrue(len(relevant_docs) > 0)
        self.assertTrue(len(article_scores) > 0)

    def test_select_relevant_articles(self):
        query = "How to obtain valid consent?"
        query_embedding = self.retriever.embeddings.embed_query(query)
        relevant_articles, article_scores = self.retriever._select_relevant_articles(query, query_embedding, self.documents, top_k=3)
        self.assertIsInstance(relevant_articles, list)
        self.assertIsInstance(article_scores, list)
        self.assertEqual(len(relevant_articles), 3)
        self.assertEqual(len(article_scores), 3)

    def test_calculate_article_scores(self):
        query = "What is data minimization?"
        query_embedding = self.retriever.embeddings.embed_query(query)
        scored_articles = self.retriever._calculate_article_scores(query, query_embedding, self.documents)
        self.assertIsInstance(scored_articles, list)
        self.assertTrue(len(scored_articles) > 0)
        for article in scored_articles:
            self.assertIn('article_number', article)
            self.assertIn('score', article)
            self.assertIn('reasons', article)

if __name__ == '__main__':
    unittest.main()