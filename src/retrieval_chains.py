from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from dotenv import load_dotenv
from utils import vectorize
from llms.py import llm
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain.load import dumps, loads
from langchain_core.prompts import ChatPromptTemplate
from prompts import rag_prompt

load_dotenv()

prompt = hub.pull('rlm/rag-prompt')


retriever = vectorize('shreyankresume.pdf').as_retriever()

def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)


rag_chain = (
    {"context":retriever |format_docs , 'question':RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


prompt = ChatPromptTemplate.from_template(rag_prompt)

generate_queries = (
    prompt
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)


def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]


retrieval_chain = generate_queries | retriever.map() | get_unique_union




template = """Answer the following question based on this context:

{context}

Question: {question}
"""



prompt = ChatPromptTemplate.from_template(template)



final_rag_chain = (
    {"context": retrieval_chain, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

