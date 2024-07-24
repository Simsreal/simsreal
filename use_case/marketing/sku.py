import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage

st.title("Simsreal 洗髮水自動營銷助理")

model = ChatOllama(model="phi3:14b")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
st.subheader("聊天記錄")
for message in st.session_state.chat_history:
    st.text(f"{'你' if message['is_user'] else 'AI'}: {message['content']}")

user_input = st.text_input("Ask a question:")

if st.button("Submit"):
    if user_input:
        st.session_state.chat_history.append({"is_user": True, "content": user_input})
        
        with st.spinner("AI is thinking..."):
            try:
                st.text("Invoking model...")
                response = model.invoke([HumanMessage(content=user_input)])
                st.text("Model response received.")
                st.write("Debug - Raw model response:", response)
                
                ai_message = response.content if hasattr(response, 'content') else str(response)
                st.session_state.chat_history.append({"is_user": False, "content": ai_message})
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.text("Traceback:")
                import traceback
                st.text(traceback.format_exc())
        
        st.rerun()
    else:
        st.warning("Please enter a question.")

st.write("Debug - Current chat history:")
st.write(st.session_state.chat_history)