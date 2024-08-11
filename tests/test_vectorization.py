import unittest
from src.vectorization import SegregatedVectorStore
from src.data_preparation import DataPreparation

class TestSegregatedVectorStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_preparation = DataPreparation()
        cls.documents = data_preparation.prepare_documents()
        cls.vector_store = SegregatedVectorStore(cls.documents)

    def test_initialization(self):
        self.assertIsInstance(self.vector_store, SegregatedVectorStore)
        self.assertEqual(len(self.vector_store.article_stores), 21)

    def test_search(self):
        query = "What is personal data?"
        results = self.vector_store.search(query, top_k=3)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        for doc, score in results:
            self.assertIn('article_number', doc.metadata)
            self.assertIsInstance(score, float)

    def test_search_specific_articles(self):
        query = "What are the principles of data processing?"
        article_numbers = ['5', '6']
        results = self.vector_store.search(query, article_numbers=article_numbers, top_k=2)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        for doc, score in results:
            self.assertIn(str(doc.metadata['article_number']), article_numbers)  # Convert to string

if __name__ == '__main__':
    unittest.main()