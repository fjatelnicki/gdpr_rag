import fitz
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import logging
from src.data_loading import load_article_summaries
from src.keywords_extraction import extract_keywords
from . import constants
from tqdm import tqdm


class DataPreparation:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Compile regex patterns for pagination and article numbers
        self.pagination_pattern = constants.PAGINATION_PATTERN
        self.article_number_pattern = constants.ARTICLE_NUMBER_PATTERN

    def extract_pdf_text(self, pdf_path):
        # Extract text from PDF, returning list of (page_number, text) tuples
        with fitz.open(pdf_path) as doc:
            return [(page.number + 1, page.get_text()) for page in doc]

    def parse_articles(self, pages):
        # Parse extracted text into separate articles
        articles = {}
        current_article = None

        for page_num, page_text in pages:
            if not self.pagination_pattern.search(page_text):
                lines = page_text.split("\n")
                if len(lines) > 1:
                    match = self.article_number_pattern.match(lines[1])
                    if match:
                        current_article = int(match.group(1))
                        articles[current_article] = []

            if current_article is not None:
                articles[current_article].append((page_num, page_text))

        # Log warning if unexpected number of articles found
        if len(articles) != constants.EXPECTED_ARTICLE_COUNT:
            self.logger.warning(
                f"Expected {constants.EXPECTED_ARTICLE_COUNT} articles, but found {len(articles)}"
            )

        return articles

    def clean_articles(self, articles):
        # Clean article text by removing links, newlines, and pagination
        cleaned_articles = {}
        for article_number, pages in articles.items():
            cleaned_articles[article_number] = []
            for page_num, content in pages:
                cleaned_content = content.replace(constants.LINK1, "").strip()
                cleaned_content = cleaned_content.replace(constants.LINK2, "").strip()
                cleaned_content = cleaned_content.replace("\n", " ").strip()
                cleaned_content = re.sub(
                    self.pagination_pattern, "", cleaned_content
                ).strip()
                cleaned_articles[article_number].append((page_num, cleaned_content))
        return cleaned_articles

    def prepare_documents(
        self,
        pdf_path=constants.PDF_PATH,
        summaries_path=constants.SUMMARIES_PATH,
        chunk_size=constants.DEFAULT_CHUNK_SIZE,
        chunk_overlap=constants.DEFAULT_CHUNK_OVERLAP,
    ):
        # Main method to prepare documents for processing

        with tqdm(total=5, desc="Document Preparation") as pbar:
            pages = self.extract_pdf_text(pdf_path)
            pbar.update(1)
            pbar.set_description("Parsing articles")
            parsed_articles = self.parse_articles(pages)
            pbar.update(1)
            pbar.set_description("Cleaning articles")
            cleaned_articles = self.clean_articles(parsed_articles)
            pbar.update(1)
            pbar.set_description("Loading summaries")
            article_summaries = load_article_summaries(summaries_path)
            pbar.update(1)

            # Initialize text splitter for chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
            )

            # Prepare all article contents for keyword extraction
            all_article_contents = [
                " ".join(content for _, content in pages)
                for pages in cleaned_articles.values()
            ]

            documents = []
            pbar.set_description("Processing articles")
            for article_number, pages in tqdm(
                cleaned_articles.items(), desc="Articles", leave=False
            ):
                article_content = " ".join(content for _, content in pages)
                chunks = text_splitter.split_text(article_content)
                summary = article_summaries.get(
                    str(article_number), "Summary not available"
                )
                keywords = extract_keywords(article_content, all_article_contents)

                # Calculate page start positions for chunk mapping
                cumulative_length = 0
                page_start_positions = [0]
                for _, content in pages:
                    cumulative_length += len(content) + 1
                    page_start_positions.append(cumulative_length)

                for chunk in chunks:
                    # Map chunk to its corresponding page
                    chunk_start = article_content.index(chunk)
                    page_num = 0
                    for i, pos in enumerate(page_start_positions):
                        if pos > chunk_start:
                            page_num = i - 1
                            break
                    exact_page = pages[page_num][0]

                    # Create metadata for the chunk
                    metadata = {
                        "article_number": article_number,
                        "article_summary": summary,
                        "page": exact_page,
                        "keywords": ", ".join(keywords),
                    }
                    documents.append(Document(page_content=chunk, metadata=metadata))
            pbar.update(1)

        return documents
