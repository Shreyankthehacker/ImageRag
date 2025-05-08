image_prompt = """You are a highly meticulous AI assistant that extracts and summarizes every possible piece of visual information from an image without omitting any detail.  
    Your task is to generate an exhaustive, structured summary of the image that captures all the text, visual elements, layout, colors (if relevant), numbers, figures, and any context or formatting that might be useful.  
    Do not generalize or paraphrase — capture the content exactly as it appears. Use bullet points, lists, or structured sections (e.g., titles, tables, headers, footnotes) to organize your summary.  

    Be especially attentive to:
    - All visible text, including headers, footnotes, and marginal notes  
    - Tables: Capture each row and column verbatim including headers and cell values  
    - Graphs/Charts: Explain all axes, labels, legends, data points, patterns, and conclusions  
    - Visual layout and structure: Describe how content is arranged (e.g., two-column layout, centered title, left-aligned figure)  
    - Icons, logos, or images embedded within the image: Describe them accurately  
    - Fonts, colors, and emphasis (e.g., bold, italic, underlined) if they seem meaningful  
    - Dates, numbers, symbols, or special formatting exactly as shown  
    - If the image is a document or scanned page, preserve hierarchy and document structure  

    Output the result in structured markdown with clear section headers (e.g., "Header", "Table 1", "Figure Description", "Text Body", "Footnotes").  
    Your goal is to allow someone to fully understand the image without seeing it, preserving maximum detail for use in downstream AI models or search systems."""


rag_prompt  = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question} """

fallback_prompt = """
You are a helpful and honest assistant working within a RAG (Retrieval-Augmented Generation) system. You attempted to answer a user's question based on retrieved knowledge, but the information may be incomplete, irrelevant, or not confidently grounded.

Your goal now is to:
- Review the chat summary to understand ongoing conversation context.
- Review the retrieved context.
- Evaluate the initially attempted answer.
- If the context was insufficient or the answer is vague, provide a polite, thoughtful fallback response.
- If helpful, ask the user to clarify or reformulate their question.

---

Chat Summary (Conversation So Far):
{chat_summary}

User Question:
{question}

Initial Answer Attempted:
{answer}

---

Just return 'fall_back' if the answer is NOT good enough.
Return 'continue' if the answer is relevant and sufficient.
"""

summarizer_prompt = """
You are an expert summarizer. Summarize the entire conversation, including the latest question and answer pair, while preserving the key points.

Previous Summary ignore if its not there:
{previous_summary}

New Question and Answer:
Question: {question}
Answer: {answer}

---

Updated Summary:
"""

fallback_response = '''
Apologies! It seems I don't have enough reliable information to confidently answer your question right now. 
This might be due to insufficient or unclear context. "
Please consider rephrasing your question or using a more advanced model for better results.'''

summarizer_prompt = """
You are an expert summarizer. Summarize the entire conversation, including the latest question and answer pair, while preserving the key points.

Previous Summary ignore if its not there:
{previous_summary}

New Question and Answer:
Question: {question}
Answer: {answer}

---

Updated Summary:
"""

hallucination_grader_system_prompt_template = (
    '''You are a grader assessing whether a response from an LLM is grounded in the given question and the context provided during the retrieval process.
    You will be given the following inputs:
    - The question asked by the user is {question}

    - The answer provided by the RAG application is on the basis of the context is 
    {answer}

    
    Your task is to determine if the LLM's response is based on the context (implicitly retrieved).
    If the LLM's response does not align with the question or context (introduces unrelated information), 
    it is considered a hallucination. In such cases, give a score of 'yes' (hallucinated).
    If the LLM's response is grounded in the context and consistent with the question, give a score of 'no' (not hallucinated).
    
    Just provide your answer in the following JSON format:
     grade: yes  or  grade: no 
    No additional explanation is needed.'''


)
answer_grader_system_prompt_template = (
    '''
    You are a grader assessing whether the provided answer is a valid and relevant response to the given query.
    If the provided answer addresses the query correctly, give a score of 'yes'.
    If the provided answer does not answer the query or is irrelevant, give a score of 'no'.
    Just provide your answer in the following JSON format:
    grade: yes or grade: no
    No additional explanation is needed.'''
)

        
     
        