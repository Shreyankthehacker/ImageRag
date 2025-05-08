import dotenv 
dotenv.load_dotenv()

from state import State
from llms import llm
from prompts import summarizer_prompt

def summarizer_node(state:State):
    
    question = state.question
    answer = state.answer
    previous_summary = state.chat_summary 
    
    
    prompt = summarizer_prompt.format(
        previous_summary=previous_summary,
        question=question,
        answer=answer
    )

    
    updated_summary = llm.invoke(prompt)

    
    return {"updated_summary": updated_summary}