import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

st.set_page_config(page_title="AI Chatbot", layout="centered")

# Initialize session states
if "api_key_configured" not in st.session_state:
    st.session_state["api_key_configured"] = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "input" not in st.session_state:
    st.session_state.input = ""

# Title and description
st.title("AI Chatbot")

# API Key input section
with st.sidebar:
    st.markdown("## Configuration")
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    if st.button("Save API Key"):
        if api_key.startswith("sk-") and len(api_key) > 50:
            os.environ["OPENAI_API_KEY"] = api_key
            st.session_state["api_key_configured"] = True
            st.success("API Key saved successfully!")
        else:
            st.error("Please enter a valid OpenAI API key.")
    
    st.markdown("---")
    st.markdown("""
    ### How to get an API Key:
    1. Go to [OpenAI API](https://platform.openai.com/api-keys)
    2. Sign up or log in
    3. Create a new API key
    4. Copy and paste it here
    """)

# Main chat interface
if st.session_state["api_key_configured"]:
    try:
        # Initialize OpenAI and LangChain components
        llm = OpenAI()
        
        # Define the prompt template
        template = """You are a helpful AI assistant. Please provide a clear and concise response to the following:

        User's question: {question}

        Response:"""
        
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        # Chat interface
        st.markdown("""
        <style>
        .user-message {
            background-color: #2b313e;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 5px 0;
        }
        .assistant-message {
            background-color: #f0f2f6;
            color: black;
            padding: 15px;
            border-radius: 10px;
            margin: 5px 0;
            border: 1px solid #e0e0e0;
        }
        </style>
        """, unsafe_allow_html=True)

        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

        # Input field and send button
        def submit():
            if st.session_state.input.strip():
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": st.session_state.input})
                
                # Get AI response
                response = llm_chain.run(question=st.session_state.input)
                
                # Add assistant response to chat
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Clear input
                st.session_state.input = ""

        st.text_input("Type your message:", key="input", on_change=submit)
        
    except Exception as e:
        st.error(f"Configuration error: {str(e)}")
else:
    st.warning(" Please configure your OpenAI API key in the sidebar to start chatting.")
