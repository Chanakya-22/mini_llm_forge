import streamlit as st
import requests

# Configure the page
st.set_page_config(page_title="Mini-LLM Assistant", page_icon="🤖")
st.title("Mini-LLM Assistant 🤖")
st.caption("Powered by Mini-LLM-Forge & Local QLoRA")

# Your Docker Backend URL
API_URL = "http://localhost:8000/api/v1/completions"

# Initialize memory so the chat doesn't disappear when you type
if "messages" not in st.session_state:
    st.session_state.messages = []

# Redraw the chat history on every interaction
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# The Chat Input Box
if prompt := st.chat_input("Ask me anything about my projects or skills..."):
    # 1. Show user message
    st.chat_message("human").markdown(prompt)
    st.session_state.messages.append({"role": "human", "content": prompt})

    # 2. Format the payload using the backend's current chat schema
    payload = {
        "messages": st.session_state.messages,
        "max_new_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "do_sample": True,
        "stop_sequences": ["<|im_end|>"],
    }

    # 3. Request the answer from the Docker container
    with st.chat_message("bot"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(API_URL, json=payload)
                print(response.text)
                response.raise_for_status()  # Check for 500 errors
                
                bot_reply = response.json().get("generated_text", "Error: No response generated.")
                st.markdown(bot_reply)
                
                # Save bot reply to history
                st.session_state.messages.append({"role": "bot", "content": bot_reply})
            
            except requests.exceptions.ConnectionError:
                st.error("🚨 Backend is offline! Make sure your Docker container is running on port 8000.")
            except Exception as e:
                st.error(f"An error occurred: {e}")