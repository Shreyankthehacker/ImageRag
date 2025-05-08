import filetype
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from pdf2image import convert_from_path
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import base64
from langchain_core.messages import HumanMessage
from io import BytesIO
from prompts import image_prompt
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader



def file_router(file):
    kind = filetype.guess(file)
    if kind is None:
        return "Unknown"
    file_type =  kind.mime
    if file_type.startswith("image/"):
        return 'imagesingle'

    # or else this is pdf and if there is images kind of pdf then return 'image' or text based then return pdf

    loader = PyPDFLoader(file)

    docs = loader.load()

    if not len(docs[0].page_content):
        return 'imagepdf'
    
    else :
        return 'pdf'
    




model = llm = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')

def encode_image(image) -> str:
    """Encode a PIL image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def image_summarize(model, base64_image: str, prompt: str) -> str:
    """Make image summary"""
    msg = model.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ]
            )
        ]
    )
    return msg.content

def image_handler(image):

    base64_img = encode_image(image)
    summary = image_summarize(model, base64_img, prompt=image_prompt)
    with open('example.txt','w') as f:
        f.write(summary)
    return summary



def image_handler_append(image):
    base64_img = encode_image(image)
    summary = image_summarize(model, base64_img, prompt=image_prompt)
    
    # Append instead of overwrite
    with open('example.txt', 'a') as f:
        f.write(summary + '\n')  # Add newline for separation
    
    return summary



def vectorize_text(text:str):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 600,chunk_overlap = 50)

    docs = splitter.split_text(text)

    vectorstore = Chroma.from_texts(docs,embedding= HuggingFaceEmbeddings())

    return vectorstore


def vectorize_single_image(image):
    summary = image_handler(image)
    return vectorize_text(summary)


def vectorize_multiple_images(image):

    images = convert_from_path(image)
    summary = ''
    for i, image in enumerate(images):
        filename = f"page_{i + 1}.png"
        print(filename)
        image.save(filename, "PNG")
        if filename == 'page_1.png':
            summary = image_handler(image)
        else:
            summary += image_handler_append(image)
    
    return vectorize_text(summary)

def vectorize_docs(filepath):
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size= 600,chunk_overlap= 80)
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(chunks,HuggingFaceEmbeddings())
    return vectorstore


def vectorize(filepath):
    type_of_file = file_router(filepath)
    print(type_of_file)
    if type_of_file == 'imagesingle':
        return vectorize_single_image(filepath)
    elif type_of_file == 'imagepdf':
        return vectorize_multiple_images(filepath)
    else :
        return vectorize_docs(filepath)
        
