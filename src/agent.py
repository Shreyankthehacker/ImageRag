from langgraph.graph import StateGraph, START, END
from state import State
from nodes import rag_answer,hallucination_and_answer_relevance_check
from fallback import fallback_node
from summarizer import summarizer_node
from hallucination_grader import hallucination_grader
from answer_grader import answer_grader


builder = StateGraph(State)


builder.add_node("rag_answer", rag_answer)
builder.add_node("summarizer_node", summarizer_node)
builder.add_node("fallback_node", fallback_node)
builder.add_node(hallucination_grader)
builder.add_node(answer_grader)



builder.add_edge(START, "rag_answer")
builder.add_edge("rag_answer", "summarizer_node")
builder.add_edge('summarizer_node','hallucination_grader')
builder.add_edge('hallucination_grader','answer_grader')


builder.add_conditional_edges(
    "answer_grader",
    hallucination_and_answer_relevance_check,  
   {
        "useful": END,  
        "generate": "rag_answer", 
        "not_useful": "fallback_node"
    },
)
builder.add_edge("fallback_node", END)

graph = builder.compile()

state = State(question = "What job role is perfect for the candidiate",answer = '',chat_summary = '')

print(graph.invoke(state).answer)