# GDPR Articles RAG System

This project implements a Retrieval-Augmented Generation (RAG) system for querying and retrieving information from GDPR Articles 1 to 21. The system parses a PDF document, understands the context of the articles, vectorizes the content, and provides accurate responses to user queries through a console-based interface.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Key Components](#key-components)
3. [Data Preparation](#data-preparation)
4. [Vectorization](#vectorization)
5. [Retrieval](#retrieval)
6. [Generation](#generation)
7. [Console Interface](#console-interface)
8. [Constants and Configuration](#constants-and-configuration)
9. [Testing](#testing)
10. [Installation and Usage](#installation-and-usage)

## Project Structure

The project is organized into several Python modules:
```
src/
├── __init__.py
├── constants.py
├── data_loading.py
├── data_preparation.py
├── generator.py
├── keywords_extraction.py
├── rag_model.py
├── retriever.py
├── vectorization.py
└── console_interface.py
tests/
├── ...
data/
├── gdpr_articles.pdf
└── articles_summaries.json
main.py
requirements.txt
```

## Key Components

1. **DataPreparation**: Extracts and processes text from the PDF document.
2. **SegregatedVectorStore**: Manages vector representations of document chunks.
3. **Retriever**: Selects relevant documents based on user queries.
4. **Generator**: Generates answers using retrieved documents.
5. **RAGModel**: Combines retrieval and generation components.
6. **Console Interface**: Handles user interaction through the command line.

## Data Preparation

The `DataPreparation` class (in `data_preparation.py`) is responsible for extracting text from the PDF document, parsing the text into separate articles, cleaning the extracted text, chunking the text into smaller segments, and adding metadata to each chunk.

## Vectorization

The `SegregatedVectorStore` class (in `vectorization.py`) handles grouping documents by article number, creating a separate Chroma vector store for each article, and providing a search method to find relevant documents across all or specific articles.

## Retrieval

The `Retriever` class (in `retriever.py`) is responsible for selecting relevant articles based on the query, calculating article scores using summary similarity, keyword matches, and article mentions, and selecting relevant document parts from the chosen articles.

## Generation

The `Generator` class (in `generator.py`) handles preparing the context from relevant documents and generating answers using the OpenAI language model.

## Console Interface

The console interface (in `console_interface.py`) manages user input handling and displaying results, including the generated answer and retrieved documents.

## Constants and Configuration

The `constants.py` file contains various configuration parameters and constants used throughout the project, including regex patterns for text processing, file paths for input data, and model parameters and scoring weights.

## Keyword Extraction and Legal Document Matching

### Keyword Extraction using c-TF-IDF

The system employs a class-based Term Frequency-Inverse Document Frequency (c-TF-IDF) approach for keyword extraction. This method is particularly effective for identifying important terms within each GDPR article:

1. Each article is treated as a separate "class" or document.
2. TF-IDF scores are calculated for each term within each article.
3. The terms with the highest c-TF-IDF scores are selected as keywords for each article.

This approach helps in identifying terms that are particularly important or unique to each article, improving the relevance of retrieved documents.

### Legal Document Matching using Regex

To enhance the accuracy of document retrieval, especially for legal texts like GDPR articles, the system uses regular expressions (regex) for pattern matching:

1. Specific regex patterns are defined to identify legal references, article numbers, and key phrases commonly used in legal documents.
2. These patterns are applied during the text processing stage to extract and highlight relevant information.
3. The extracted information is used to enrich the document metadata and improve the retrieval process.

This regex-based approach ensures that important legal references and structures within the GDPR articles are properly recognized and utilized in the retrieval process.


## Intermediary Model for Article Selection

In addition to the core RAG system, we have implemented an intermediary model for intelligent Article selection based on user queries. This model analyzes the user's question to determine the most relevant GDPR Articles for the query, improving the relevance and accuracy of the responses.

The article selection process is implemented in the `Retriever` class and uses a combination of techniques:

1. Summary similarity: Compares the query embedding with article summary embeddings.
2. Keyword matching: Checks for keyword matches between the query and article summaries.
3. Article number mentions: Detects explicit mentions of article numbers in the query.

The Article Score is calculated using the following formula:

$$\text{Article\_Score} = \frac{w_1 \cdot \text{Summary\_Similarity} + w_2 \cdot \text{Keyword\_Match\_Score} + w_3 \cdot \text{Article\_Mention\_Score}}{w_1 + w_2 + w_3}$$

Where:
- $w_1$, $w_2$, and $w_3$ are weights for each component
- $\text{Summary\_Similarity}$ is the similarity score between the article and summary
- $\text{Keyword\_Match\_Score}$ is the score based on keyword matches
- $\text{Article\_Mention\_Score}$ is the score based on article mentions


This approach allows the system to focus on the most relevant articles for each query, enhancing the overall performance of the RAG system.

// ... rest of the existing content ...

## Testing

The project includes several test files to ensure the correct functioning of key components:

1. `test_data_preparation.py`: Tests for the DataPreparation class.
2. `test_vectorization.py`: Tests for the SegregatedVectorStore class.
3. `test_retriever.py`: Tests for the Retriever class.
4. `test_minimal.py`: A minimal test to ensure the testing setup works.

## Installation and Usage

1. Clone the repository
2. Install the required dependencies:

```
pip install -r requirements.txt
```

3. Run the main script to start the console interface:

```
python main.py
```


4. Enter your questions about GDPR articles when prompted. Type 'quit' to exit the program.

The system will process your query, retrieve relevant documents, and generate an answer based on the GDPR articles.

## Usage Examples

Here are two examples of how to use the GDPR Articles RAG System:

### Example 1: Querying about a specific article
```
Enter your question (or 'quit' to exit): What is Article 1 about?

Searching for relevant documents...

Answer:
Article 1 is about the cooperation and exchange of personal data between Member States in order to protect personal data in the face of rapid technological developments and globalization. It also addresses the need for the free flow of personal data within the Union and to third countries and international organizations, while ensuring a high level of protection for personal data. Additionally, it mentions the increase in cross-border flows of personal data due to the functioning of the internal market and the role of national authorities in training, consulting, and outsourcing DPOs.
Retrieved Documents:
Article 1:
Reasons:
Query-Summary similarity: 0.18
Exact article 1 mention
Retrieved Document Parts:
...
```

### Example 2: Querying about a specific concept/keyword
```
Enter your question (or 'quit' to exit): Explain what portability is
Searching for relevant documents...
Answer:
Portability, as defined in [Article 20 | Page 139] of the GDPR, is the right of a data subject to receive their personal data in a structured, commonly used and machine-readable format and to transmit those data to another controller without hindrance from the original controller. This means that individuals have the right to move, copy or transfer their personal data from one organization to another, in a secure and safe manner. This allows individuals to have more control over their personal data and to easily switch between service providers. However, this right does not apply if the processing of personal data is necessary for compliance with a legal obligation or for the performance of a task carried out in the public interest.
Retrieved Documents:
Article 20:
Reasons:
Query-Summary similarity: 0.26
Keyword matches: portability
Retrieved Document Parts:
...
```