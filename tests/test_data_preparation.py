import unittest
from src.data_preparation import DataPreparation
from src import constants

class TestDataPreparation(unittest.TestCase):
    def setUp(self):
        self.data_preparation = DataPreparation()

    def test_extract_pdf_text(self):
        pages = self.data_preparation.extract_pdf_text(constants.PDF_PATH)
        self.assertIsInstance(pages, list)
        self.assertTrue(len(pages) > 0)
        self.assertIsInstance(pages[0], tuple)
        self.assertEqual(len(pages[0]), 2)

    def test_parse_articles(self):
        pages = self.data_preparation.extract_pdf_text(constants.PDF_PATH)
        articles = self.data_preparation.parse_articles(pages)
        self.assertIsInstance(articles, dict)
        self.assertEqual(len(articles), constants.EXPECTED_ARTICLE_COUNT)

    def test_clean_articles(self):
        pages = self.data_preparation.extract_pdf_text(constants.PDF_PATH)
        articles = self.data_preparation.parse_articles(pages)
        cleaned_articles = self.data_preparation.clean_articles(articles)
        self.assertIsInstance(cleaned_articles, dict)
        self.assertEqual(len(cleaned_articles), len(articles))
        for article_number, pages in cleaned_articles.items():
            for _, content in pages:
                self.assertNotIn(constants.LINK1, content)
                self.assertNotIn(constants.LINK2, content)

    def test_prepare_documents(self):
        documents = self.data_preparation.prepare_documents()
        self.assertIsInstance(documents, list)
        self.assertTrue(len(documents) > 0)
        for doc in documents:
            self.assertIn('article_number', doc.metadata)
            self.assertIn('article_summary', doc.metadata)
            self.assertIn('page', doc.metadata)
            self.assertIn('keywords', doc.metadata)

if __name__ == '__main__':
    unittest.main()