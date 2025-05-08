from langchain_core.prompts import ChatPromptTemplate
from prompts import hallucination_grader_system_prompt_template
from state import HallucinationGrader,State
from llms import llm

hallucination_grader_prompt = ChatPromptTemplate.from_template(hallucination_grader_system_prompt_template)


def hallucination_grader(state: State):
    hallucination_grader_chain = (
        hallucination_grader_prompt
        | llm.with_structured_output(HallucinationGrader, method="json_mode")
    )

    
    graded_response = hallucination_grader_chain.invoke({
        'question': state.question,
        'answer': state.answer
    })
    
    return graded_response