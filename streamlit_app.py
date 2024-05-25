import streamlit as st
from langchain_community.vectorstores import FAISS
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
import base64
import time

# Set up Google API key
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCEdziXn1Lm_B6H7WluHG74j14LWbZlXSY" 

# Initialize Google GenAI model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", convert_system_message_to_human=True)

# Load local embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def ocr_image(image_bytes):
    """Performs OCR on the image using the Google GenAI model."""
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "What is the text in this image? if you find a graph or an equation, clear about it too",
            },
            {
                "type": "image_url",
                "image_url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}",
            },
        ]
    )
    with st.spinner("OCR is running..."):
        res = model.invoke([
            SystemMessage(content="You are an expert at reading text from images."),
            message
        ])
    return res.content

def generate_response(reference_text, user_input, image_url=None):
    """Generates a response from the LLM."""
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": user_input,
            },
        ]
    )
    if image_url:
        message.content.append({"type": "image_url", "image_url": image_url})

    start_time = time.time()
    with st.spinner("Thinking..."):
        res = model.invoke([
            SystemMessage(content=f"""you are a student helper, helping answering and studying digital communication, response with elaboration like a professor, alwayes try to get the answer from the documents, and if you do not find an answer, say what you know about the question, never say The provided document snippets don't, I do not know ... or any other meaning. :references :{reference_text}"""),
            message
        ])
    end_time = time.time()
    elapsed_time = end_time - start_time
    return res.content, elapsed_time

def rag_with_text(user_ask_text, vectorstore):
    docs = vectorstore.similarity_search(user_ask_text, k=35)
    return docs

def main():
    st.title("Digital Communication Assistant")

    # Load or create FAISS index (using local embeddings)
    vectorstore = FAISS.load_local("faiss_index.bin", embeddings, allow_dangerous_deserialization=True)  # Load existing index

    # Upload image (optional)
    uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])
    image_url = None
    ocr_text = None
    if uploaded_file is not None:
        # Convert image to base64 for display
        image_bytes = uploaded_file.read()
        encoded_image = base64.b64encode(image_bytes).decode()
        st.image(f"data:image/png;base64,{encoded_image}", caption="Uploaded Image")

        # Store image URL in a variable to pass to the model
        image_url = f"data:image/png;base64,{encoded_image}"

        # Perform OCR on the image
        ocr_text = ocr_image(image_bytes)
        # st.write(f"**OCR Text:**\n{ocr_text}")  # You can uncomment this line to display OCR text 

    # Text input
    user_input = st.text_area("Ask me anything about digital communication!", height=200)
    # Generate response and display
    if st.button("Ask"):
        if user_input and ocr_text:
            # Get relevant documents from the vector store
            docs = rag_with_text(user_input, vectorstore)
            docs += rag_with_text(ocr_text, vectorstore)

            # Combine retrieved documents into a single string
            combined_reference_text = "\n".join([doc.page_content for doc in docs])

            # Generate response with combined reference text
            response, elapsed_time = generate_response(combined_reference_text, user_input, image_url)
            st.write(f"**Response:**\n{response}")
            st.write(f"**Elapsed Time:** {elapsed_time:.2f} seconds")
            
        if user_input :
            # Get relevant documents from the vector store
            docs = rag_with_text(user_input, vectorstore)
            # Combine retrieved documents into a single string
            combined_reference_text = "\n".join([doc.page_content for doc in docs])
            # Generate response with combined reference text
            response, elapsed_time = generate_response(combined_reference_text, user_input, image_url)
            st.write(f"**Response:**\n{response}")
            st.write(f"**Elapsed Time:** {elapsed_time:.2f} seconds")
            
        if ocr_text :
            # Get relevant documents from the vector store
            docs = rag_with_text(ocr_text, vectorstore)
            # Combine retrieved documents into a single string
            combined_reference_text = "\n".join([doc.page_content for doc in docs])
            # Generate response with combined reference text
            response, elapsed_time = generate_response(combined_reference_text, user_input, image_url)
            st.write(f"**Response:**\n{response}")
            st.write(f"**Elapsed Time:** {elapsed_time:.2f} seconds")
        

if __name__ == "__main__":
    main()
