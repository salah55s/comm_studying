import streamlit as st
from langchain_community.vectorstores import FAISS
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
import base64
import io
from PIL import Image
import time

# Set up Google API key
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDdePysSFcpnwI_2HbPdM07pndliZ-rnb4" 

# Initialize Google GenAI model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", convert_system_message_to_human=True)

# Load local embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Function to perform OCR
def ocr_image(image_bytes):
    """Performs OCR on the image using the Google GenAI model."""
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "What is the text in this image? If you find a graph or an equation, clear it, and if it's a circuit, write it's name, type and any information about it",
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

# Function to generate response from the LLM
def generate_response(reference_text, user_input, image_url=None,image_urls=None, subject="Electronics"):
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

    if subject == "Electronics":

        system_message = f"""You are a student helper AI, specializing in answering and studying questions related to advanced electronics circuits. Respond with detailed explanations as a professor would, using examples and information from the provided documents without explicitly mentioning them. If you cannot find an answer, provide what you know about the question without referencing the provided text, regardless of whether it contains the required information or not.

When you write an equation, it must be displayed directly. Additionally, you must solve any questions provided to you step by step, whether they are in the form of an image or text.

You must use only circuit theorems and laws to provide a step-by-step solution. The references may contain equations and information about the circuit. Focus on every detail in the circuit image, never missing any information in the circuit images. Pay close attention to the directions and values of all nodes in the images. Your way of solving is to simplify the circuit first, if needed, then you must elaborate about each step and calculate it.

References: {reference_text}

you will have images as References, which is diffrent that the user ask."""
        
        for imgs in image_urls:
            system_message.append({"type": "image_url", "image_url": imgs})
    else:
        system_message = f"""You are a student helper AI, called ECE HELPER, helping answer and study questions about the chosen subject. Respond with elaboration like a professor, and try to use the information from the documents provided, but never say that you used it. 
        If you do not find an answer, say what you know about the question, and never mention "The provided text defines", use the information and never say about the provided text, whether it has the required information or even if it does not. When you write an equation, it must be displayed directly. Additionally, you must solve any questions provided to you step by step, whether they are in the form of an image or text.        :references :{reference_text}"""
    
    
    start_time = time.time()
    with st.spinner("Thinking..."):
        
        res = model.invoke([
            SystemMessage(content=system_message),
            message
        ])
    end_time = time.time()
    elapsed_time = end_time - start_time
    return res.content, elapsed_time

# Function to perform RAG 
def rag_with_text(user_ask_text, vectorstore):
    """Performs RAG using FAISS index."""
    # For Electronics, search for images based on OCR text
    if "Electronics" in user_ask_text.lower():
        faiss_index = FAISS.load_local("faiss_index3.bin", embeddings)
        docs = faiss_index.similarity_search(user_ask_text, k=2)
    else:
        docs = vectorstore.similarity_search(user_ask_text, k=10) 
    return docs

# Function to provide a downloadable file link
def download_file(file_path):
    """Generates a link allowing the user to download a given file."""
    with open(file_path, "rb") as f:
        bytes_data = f.read()
    b64 = base64.b64encode(bytes_data).decode()
    href = f"<a href='data:application/octet-stream;base64,{b64}' download='{os.path.basename(file_path)}'>Download {os.path.basename(file_path)}</a>"
    return href

# Main Streamlit app
def main():
    st.title("Study with me - ECE AI Helper")

    # Subject Selection
    subject = st.selectbox("Choose a subject:", ["Electronics","Control Systems","Digital Communication"])

    # Load the appropriate FAISS index based on subject selection
    if subject == "Electronics":
        vectorstore = FAISS.load_local("faiss_index3.bin", embeddings)  # Assuming this is for electronics text
    elif subject == "Digital Communication":
        vectorstore = FAISS.load_local("faiss_index.bin", embeddings)
    else:  # Control Systems
        vectorstore = FAISS.load_local("faiss_index2.bin", embeddings)
        vectorstore2 = FAISS.load_local("faiss_index22.bin", embeddings)
        vectorstore.merge_from(vectorstore2)

    # Upload image (optional)
    uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])
    image_url = None
    ocr_text = None
    image_urls=[]
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        encoded_image = base64.b64encode(image_bytes).decode()
        st.image(f"data:image/png;base64,{encoded_image}", caption="Uploaded Image")
        image_url = f"data:image/png;base64,{encoded_image}"
        ocr_text = ocr_image(image_bytes)

    # Text Input
    user_input = st.text_area(f"Ask me anything about {subject}!", height=200)

    # Generate Response
    if st.button("Ask"):
        if user_input or ocr_text:
            docs = []
            if user_input:
                docs += rag_with_text(user_input, vectorstore)
            if ocr_text:
                docs += rag_with_text(ocr_text, vectorstore)  

            # Combine page content from retrieved documents
            combined_reference_text = "\n".join([doc.page_content for doc in docs])
            
            for doc in docs:
                image_data = doc.metadata.get("image_data") # Assuming binary data field is 'image_data'
                if image_data:
                    encoded_image = (doc.metadata.get("image_base64") for doc in docs if doc.metadata.get("image_base64"))
                    image = f"data:image/png;base64,{encoded_image}"
                    image_urls.append(image)
            response, elapsed_time = generate_response(combined_reference_text, user_input, image_url,image_urls, subject)
            st.write(f"**Response:**\n{response}")
            st.write(f"**Elapsed Time:** {elapsed_time:.2f} seconds")

    # Display developer credits based on subject
    st.markdown("---") 
    if subject == "Digital Communication":
        st.write("Developed by: **Salah Eldin**") 
    elif subject == "Control Systems":
        st.write("Developed by: **Salah Eldin, Karim Hesham, Abdulrahman Sleem, Mohsen Mohamed**")
        st.markdown("**Download Additional Resources:** the rest is coming")
        file_paths = [
            "Part_3.pdf",
            "Part_4.pdf", 
            "lec 4.pdf",  
        ]
        for file_path in file_paths:
            if os.path.exists(file_path):
                st.markdown(download_file(file_path), unsafe_allow_html=True)
            else:
                st.warning(f"File not found: {file_path}")

    elif subject == "Electronics":
        st.write("Developed by: **Salah Eldin**")  

# Run the app
if __name__ == "__main__":
    main()
