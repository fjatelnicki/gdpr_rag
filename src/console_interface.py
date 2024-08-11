from src import constants


def print_retrieved_documents(final_docs, article_scores):
    print("\nRetrieved Documents:")
    if not final_docs:
        print("No documents retrieved.")
        return

    for article in article_scores:
        # Find corresponding document parts for this article
        relevant_parts = [
            doc
            for doc, score in final_docs
            if doc.metadata.get("article_number") == article["article_number"]
        ]

        if relevant_parts:
            print(f"\nArticle {article['article_number']}:")
            print("  Reasons:")
            for reason in article["reasons"]:
                print(f"    - {reason}")
            print("  Retrieved Document Parts:")
            for doc in relevant_parts:
                page = doc.metadata.get("page", "N/A")
                score = next((s for d, s in final_docs if d == doc), None)
                print(f"    Page {page}")
                if score is not None:
                    print(f"    Relevance Score: {score:.4f}")
                print(f"    Content preview: {doc.page_content[:100]}...")
            print()


def run_console_interface(rag_model, documents, hide_documents=False):
    # Display welcome message
    print(constants.ascii_art)

    while True:
        # Get user input
        query = input("\nEnter your question (or 'quit' to exit): ").strip()
        if query.lower() == "quit":
            break

        print("\nSearching for relevant documents...")
        # Process query and get answer
        answer, final_docs, article_scores = rag_model.answer_query(query, documents)

        # Display results
        print(f"\nAnswer: {answer}")
        if not hide_documents:
            print_retrieved_documents(final_docs, article_scores)
        print("=" * 50)
