from pydantic import BaseModel,Field
from typing import Literal 
from langchain_core.runnables import RunnableParallel


class State(BaseModel):
    question : str = Field(description= "Question given be the user")
    answer : str = Field(description="Answer given by the Application")
    chat_summary : str = Field(description="Chat history maintained by the applciation")
    


class HallucinationGrader(BaseModel):
    "Binary score for hallucination check in llm's response"

    grade: Literal["yes", "no"] = Field(
        ..., description="'yes' if the llm's reponse is hallucinated otherwise 'no'"
    )


class AnswerGrader(BaseModel):
    '''Binary score for an answer check based on a query.'''

    grade: Literal["yes", "no"] = Field(
        description="'yes' if the provided answer is an actual answer to the query otherwise 'no',"
    )
