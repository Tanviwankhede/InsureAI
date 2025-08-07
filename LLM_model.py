import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import *

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


CUSTOM_PROMPT_TEMPLATE = """
You are an intelligent insurance assistant. Answer the following question based only on the context provided.
If the answer is not in the context, reply with "Not found in the document."

Context:
{context}

Question:
{question}

Answer:
"""


# Create reusable LLM and prompt
# Load Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

# llm = ChatOpenAI(
#     model="gpt-4",  
#     temperature=0.7,
#     openai_api_key= OPENAI_API_KEY
# )

prompt = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

output_parser = StrOutputParser()

def run_qa_on_pdf(pdf_url: str, questions: list[str]) -> list[str]:
    try:
        # Step 1: Download and parse
        pdf_path = download_pdf(pdf_url)
        text = extract_text_from_pdf(pdf_path)

        # Step 2: Create retriever from vectorstore
        retriever = create_vectorstore(text)

        # Step 3: Define the retrieval-augmented pipeline
        rag_chain = (
    {
        "context": lambda x: retriever.invoke(x["question"]),  # Correctly invokes retriever
        "question": lambda x: x["question"]
    }
    | prompt
    | llm
    | output_parser
)


        # Step 4: Execute pipeline
        answers = []
        for question in questions:
            result = rag_chain.invoke({"question": question})
            answers.append(result)

        return answers

    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)