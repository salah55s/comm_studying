import streamlit as st
from langchain_community.vectorstores import FAISS
import os
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
    
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
import base64
import io
from PIL import Image
import time
from streamlit_paste_button import paste_image_button

safety_settings=[
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    }
]

api_key = None

def get_api_key_from_user():
    """Guides the user on how to obtain a Google AI Platform API key."""
    st.sidebar.header("How to Get Your Google API Key")
    st.sidebar.markdown("""
    1. **Go to Google AI studio:** [https://ai.google.dev/aistudio/](https://ai.google.dev/aistudio/)
    2. **get a gemini api key** If you don't have one already.
    3. **Create an API Key:**
       - Go with the flow.
       - When you get an API key, save it for later usage. 
    4. **Paste Your Key Below:** 
    """)
    api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")
    return api_key

if api_key is None: 
    api_key_option = st.sidebar.radio(
        "How would you like to set your API key?",
        ("Use my own API key", "Use the default API key")
    ) 

    if api_key_option == "Use my own API key":
        api_key = get_api_key_from_user()
    else:
        api_key = "AIzaSyDdePysSFcpnwI_2HbPdM07pndliZ-rnb4"  # Default API key
        st.info("Using the default API key. This may be subject to rate limits and speed.")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key 
    # Initialize Google GenAI model
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", convert_system_message_to_human=True,safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                                                                                                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                                                                                                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                                                                                                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE},)

    # Load local embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Function to perform OCR
    def ocr_image(image_bytes):
        """Performs OCR on the image using the Google GenAI model."""
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "What is the text in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}",
                },
            ]
        )
        with st.spinner("OCR is running..."):
            res = model.invoke([
                SystemMessage(content="You are an expert at reading text from images. If you find a graph or an equation, clear it, and if it's a circuit, write it's name, type and any information about it"),
                message
            ])
        return res.content

    # Function to generate response from the LLM
    def generate_response(reference_text, user_input, image_url=None, subject="Logic design"):
        """Generates a response from the LLM."""
        message_content = [
            {
                "type": "text",
                "text": user_input,
            },
        ]
        if image_url:
            message_content.append({"type": "image_url", "image_url": image_url})
        message = HumanMessage(content=message_content)
        

        if subject == "Logic design":
            system_message = [
                SystemMessage(content=f"""You are a student helper AI, called ECE HELPER, helping answer and study questions about Logic design, you must answer if the user asks a question or upload a problem. Respond with elaboration like a professor with a step-by-step answer, and try to use the information from the documents provided, but never say that you used it.
            You must write the full truth table focusing on all points, equations, and all the steps needed to solve the questions. If you do not find an answer, say what you know about the question, and never mention "The provided text defines", use the information and never say about the provided text, whether it has the required information or even if it does not. When you write an equation, it must be displayed directly. Additionally, you must solve any questions step by step, whether they are in the form of an image or text./n/n:references :/n{reference_text}""")]
        else:
            system_message = [
                SystemMessage(content=f"""You are a student helper AI, called ECE HELPER, helping answer and study questions about the chosen subject, you must answer if the user asks a question or upload a problem. Respond with elaboration like a professor with a step-by-step answer, and try to use the information from the documents provided, but never say that you used it. 
            If you do not find an answer, say what you know about the question, and never mention "The provided text defines", use the information and never say about the provided text, whether it has the required information or even if it does not. When you write an equation, it must be displayed directly. Additionally, you must solve any questions step by step, whether they are in the form of an image or text./n/n:references :/n{reference_text}""")
            ]
        
        start_time = time.time()
        with st.spinner("Thinking..."):
            res = model.invoke(system_message + [message])
        end_time = time.time()
        elapsed_time = end_time - start_time
        return res.content, elapsed_time

    # Function to perform RAG 
    def rag_with_text(user_ask_text, vectorstore):
        """Performs RAG using FAISS index."""
        # For Electronics, search for images based on OCR text
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
        subject = st.selectbox("Choose a subject:", ["Logic design", "Control Systems", "Digital Communication"])

        # Load the appropriate FAISS index based on subject selection
        if subject == "Logic design":
            vectorstore = FAISS.load_local("faiss_index4.bin", embeddings,allow_dangerous_deserialization=True) 
        elif subject == "Digital Communication":
            vectorstore = FAISS.load_local("faiss_index.bin", embeddings,allow_dangerous_deserialization=True)
        else:  # Control Systems
            vectorstore = FAISS.load_local("faiss_index2.bin", embeddings,allow_dangerous_deserialization=True)
            vectorstore2 = FAISS.load_local("faiss_index22.bin", embeddings,allow_dangerous_deserialization=True)
            vectorstore.merge_from(vectorstore2)

        # Image Input Options
        image_input_method = st.radio("How would you like to input your image?", 
                                    ("Upload an image", "Paste an image"))

        image_url = None
        ocr_text = None

        if image_input_method == "Upload an image":
            uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image_bytes = uploaded_file.read()
                encoded_image = base64.b64encode(image_bytes).decode()
                st.image(f"data:image/png;base64,{encoded_image}", caption="Uploaded Image")
                image_url = f"data:image/png;base64,{encoded_image}"
                ocr_text = ocr_image(image_bytes)

        elif image_input_method == "Paste an image":
                paste_result = paste_image_button("Click to Paste Image") 
                if paste_result:
                    if paste_result.image_data is not None:
                        st.image(paste_result.image_data, caption="Pasted Image")
                        
                        # Convert PIL Image to bytes for OCR
                        img_bytes = io.BytesIO()
                        paste_result.image_data.save(img_bytes, format='PNG')
                        image_bytes = img_bytes.getvalue()
        
                        # Keep image_url as a string for base64 encoded data
                        image_url = f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"  
                        ocr_text = ocr_image(image_bytes) 
                    else:
                        st.warning("No image pasted.")

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
                
                response, elapsed_time = generate_response(combined_reference_text, user_input, image_url,subject)
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

        elif subject == "Logic design":
            st.write("Developed by: **Salah Eldin**")  

    # Run the app
    if __name__ == "__main__":
        main()
