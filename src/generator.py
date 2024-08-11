import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from . import constants
from langchain_openai import OpenAI

load_dotenv()


class Generator:
    def __init__(self):
        self.llm = OpenAI(temperature=0.0, openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.prompt_template = PromptTemplate.from_template(constants.prompt_template)

    def generate_answer(self, query, relevant_docs):
        context = self._prepare_context(relevant_docs)
        prompt = self.prompt_template.format(context=context, question=query)
        return self.llm.invoke(prompt)

    def _prepare_context(self, relevant_docs):
        context = ""
        for doc, score in relevant_docs:
            article_number = doc.metadata.get("article_number", "Unknown")
            page_number = doc.metadata.get("page", "Unknown")
            context += f"[Article {article_number} | Page {page_number}]\n{doc.page_content}\n\n"
        return context
