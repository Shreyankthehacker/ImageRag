from state import AnswerGrader,State
from langchain_core.prompts import ChatPromptTemplate
from prompts import answer_grader_system_prompt_template
from llms import llm
from hallucination_grader import hallucination_grader


answer_grader_prompt = ChatPromptTemplate.from_template(answer_grader_system_prompt_template)
answer_grader_chain = answer_grader_prompt | llm.with_structured_output(
        AnswerGrader, method="json_mode"
    )


def answer_grader(state:State):
    answer = state.answer
    question = state.question
    grade = answer_grader_chain.invoke({'question':question,'answer':answer})
    return grade
    

