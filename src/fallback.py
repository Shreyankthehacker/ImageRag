from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from state import State
from prompts import fallback_prompt,fallback_response
from llms import llm

fall_back_prompt = ChatPromptTemplate.from_template(fallback_prompt)

def fallback_node(state: State) -> dict:
    chat_summary = state.chat_summary
    question = state.question
    answer = state.answer

    # Combine into a runnable chain
    fallback_chain = fall_back_prompt | llm | StrOutputParser()

    # Provide input variables to the chain
    decision = fallback_chain.invoke({
        "chat_summary": chat_summary,
        "question": question,
        "answer": answer
    })

    decision = decision.strip().lower()

    if decision == "continue":
        return {"answer": answer}
    else:
        fallback_response = fallback_response
        return {"answer": fallback_response}
