from . import constants
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def extract_keywords(article_content, all_articles):
    # Extract legal numbers from the article content
    legal_numbers = find_legislation_numbers(article_content)

    # Create a list of stop words including custom and English stop words
    stop_words = list(constants.custom_stopwords)
    stop_words.extend(TfidfVectorizer(stop_words="english").get_stop_words())

    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(all_articles)
    feature_names = vectorizer.get_feature_names_out()

    # Get TF-IDF vector for the current article
    article_index = all_articles.index(article_content)
    article_vector = tfidf_matrix[article_index].toarray()[0]

    # Calculate c-TF-IDF scores
    average_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)
    c_tfidf_scores = article_vector - average_tfidf

    # Extract top keywords based on c-TF-IDF scores
    top_keywords = sorted(
        zip(feature_names, c_tfidf_scores), key=lambda x: x[1], reverse=True
    )[:10]
    keywords = [word for word, score in top_keywords if score > 0]

    # Combine legal numbers and keywords, removing duplicates
    return list(dict.fromkeys(legal_numbers + keywords))


def find_legislation_numbers(article_content):
    # Find all legislation numbers in the article content using regex
    return constants.LEGAL_NUMBER_PATTERN.findall(article_content)
