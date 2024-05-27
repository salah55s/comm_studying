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
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDdePysSFcpnwI_2HbPdM07pndliZ-rnb4"  # Replace with your actual API key

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
                "text": "What is the text in this image? If you find a graph or an equation, be clear about it too",
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
            SystemMessage(content=f"""You are a student helper, helping answer and study questions about the chosen subject. Respond with elaboration like a professor, and try to use the information from the documents provided. 
            If you do not find an answer, say what you know about the question, never say "The provided document snippets don't...", "I do not know ..." or any other similar meaning. if you will write an equation, you must write it to display, not LaTeX, must solve any question provided to you through image or text  
            :references :{reference_text}"""),
            message
        ])
    end_time = time.time()
    elapsed_time = end_time - start_time
    return res.content, elapsed_time

# Function to perform RAG 
def rag_with_text(user_ask_text, vectorstore):
    docs = vectorstore.similarity_search(user_ask_text, k=35)
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
    subject = st.selectbox("Choose a subject:", ["Control Systems","Digital Communication"])

    # Load the appropriate FAISS index based on subject selection
    if subject == "Digital Communication":
        vectorstore = FAISS.load_local("faiss_index.bin", embeddings, allow_dangerous_deserialization=True)
    else:  # Control Systems
        vectorstore = FAISS.load_local("faiss_index2.bin", embeddings, allow_dangerous_deserialization=True)
        vectorstore2 = FAISS.load_local("faiss_index22.bin", embeddings, allow_dangerous_deserialization=True)
        vectorstore.merge_from(vectorstore2)

    # Upload image (optional)
    uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])
    image_url = None
    ocr_text = None

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

            combined_reference_text = "\n".join([doc.page_content for doc in docs])
            response, elapsed_time = generate_response(combined_reference_text, user_input, image_url)
            st.write(f"**Response:**\n{response}")
            st.write(f"**Elapsed Time:** {elapsed_time:.2f} seconds")
       # Display developer credits based on subject
    if subject == "Control Systems":
        st.markdown("---")  # Separator
        st.markdown("**Download Additional Resources:** the rest is coming")
        # Replace with actual file paths:
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

    if subject == "Digital Communication":
        st.write("Developed by: **Salah Eldin**") 
    elif subject == "Control Systems":
        st.write("Developed by: **Salah Eldin ,Karem Hisham, Abdulrahman Sleem, Mohsen Mohamed**") 
    

# Run the app
if __name__ == "__main__":
    main()

