import streamlit as st
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import html

# Set page config for a wider layout
st.set_page_config(page_title="TinyLlama Chatbot", layout="wide")

# Custom CSS for chat styling
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #2b313e;
        color: white;
        margin-left: 1rem;
        border-bottom-right-radius: 0.2rem;
    }
    .assistant-message {
        background-color: #7e57c2;
        color: white;
        margin-right: 1rem;
        border-bottom-left-radius: 0.2rem;
    }
    .message-content {
        margin-top: 0.5rem;
    }
    .stTextInput input {
        width: 100%;
        border-radius: 1rem;
        padding: 0.5rem 1rem;
    }
    .stButton button {
        border-radius: 1rem;
        background-color: #7e57c2;
        color: white;
    }
    .stButton button:hover {
        background-color: #673ab7;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("TinyLlama 1.1B Chatbot")

# Add token input in sidebar
with st.sidebar:
    st.header("Configuration")
    hf_token = st.text_input("Enter Hugging Face Token:", type="password", help="Get your token from https://huggingface.co/settings/tokens")
    st.markdown("**Note:** You need a Hugging Face token to use this app. Get it from [here](https://huggingface.co/settings/tokens)")

SYSTEM_PROMPT = """You are a helpful AI assistant. Your responses should be:
- Direct and relevant to the user's question
- Based only on the current conversation context
- Professional but friendly
- Never share or ask for personal information
- If unsure, admit it and ask for clarification
- Keep responses focused and concise

Current conversation:
"""

@st.cache_resource
def load_model(token):
    try:
        if not token:
            st.warning("Please enter your Hugging Face token in the sidebar.")
            return None
            
        st.info("Loading the model... Please wait.")
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=token, device_map="auto")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        st.success("Model loaded successfully!")
        return llm
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Only load model if token is provided
llm = load_model(hf_token) if hf_token else None

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Main chat interface
if not hf_token:
    st.warning("Please enter your Hugging Face token in the sidebar to start chatting.")
else:
    # Create a container for the chat messages with a fixed height
    chat_container = st.container()
    
    # Create a form for the input
    with st.form(key="message_form", clear_on_submit=True):
        user_input = st.text_input("You:", key="user_input", placeholder="Type your message here...")
        submit_button = st.form_submit_button("Send")

        if submit_button and user_input:
            if llm is None:
                st.error("Cannot generate response because model failed to load.")
            else:
                st.session_state["messages"].append({"role": "user", "content": user_input})
                
                try:
                    with st.spinner("Generating response..."):
                        # Build conversation history
                        conversation = SYSTEM_PROMPT + "\n"
                        for msg in st.session_state["messages"]:
                            role = "User" if msg["role"] == "user" else "Assistant"
                            conversation += f"{role}: {msg['content']}\n"
                        conversation += "Assistant:"
                        
                        ai_response = llm(conversation)
                        # Clean up the response if needed
                        ai_response = ai_response.strip()
                        if ai_response.startswith("Assistant:"):
                            ai_response = ai_response[len("Assistant:"):].strip()
                            
                        st.session_state["messages"].append({"role": "assistant", "content": ai_response})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

    # Display chat history in the container with scrolling
    with chat_container:
        if st.session_state["messages"]:
            for msg in st.session_state["messages"]:
                role = msg["role"]
                content = html.escape(msg["content"])  # Escape the content for safe HTML rendering
                
                # Apply different styling for user and assistant messages
                message_style = "user-message" if role == "user" else "assistant-message"
                with st.container():
                    st.markdown(f"""
                        <div class="chat-message {message_style}">
                            <div><strong>{'You' if role == 'user' else 'Assistant'}</strong></div>
                            <div class="message-content">{content}</div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Send a message to start the conversation!")
