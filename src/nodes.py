from state import State
from retrieval_chains import final_rag_chain
from answer_grader import answer_grader
from hallucination_grader import hallucination_grader


def rag_answer(state:State):
    query = state.question
    answer = final_rag_chain.invoke({'question':query})
    return {'answer':answer}


def hallucination_and_answer_relevance_check(state:State):
    

    hallucination_grade = hallucination_grader(state)
    print(f"hallucinatio grade is {hallucination_grade}")
    if hallucination_grade.grade == "no":
        print("---Hallucination check passed---")
        answer_relevance_grade = answer_grader(state)
        print(answer_relevance_grade)
        if answer_relevance_grade.grade == "yes":
            print("---Answer is relevant to question---\n")
            return "useful"
        else:
            print("---Answer is not relevant to question---")
            return "not_useful"
    print("---Hallucination check failed---")
    return "generate"