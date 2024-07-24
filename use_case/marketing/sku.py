import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
import pandas as pd
import time
import os

st.title("Simsreal 洗髮水自動營銷助理")

@st.cache_resource
def load_data():
    try:
        df = pd.read_excel('sku.xlsx')
        csv_file = 'temp.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Failed to create {csv_file}")
        
        loader = CSVLoader(csv_file, encoding='utf-8-sig')
        documents = loader.load()
        
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        os.remove(csv_file)
        return vectorstore
    except Exception as e:
        st.error(f"Error in load_data(): {str(e)}")
        if 'csv_file' in locals() and os.path.exists(csv_file):
            with open(csv_file, 'r', encoding='utf-8-sig') as f:
                st.text(f"Contents of {csv_file}:")
                st.text(f.read())
            os.remove(csv_file)
        raise e
try:
    vectorstore = load_data()
except Exception as e:
    st.error(f"Failed to load data: {str(e)}")
    st.stop()

vectorstore = load_data()
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
                st.text("Retrieving relevant information...")
                docs = vectorstore.similarity_search(user_input, k=3)
                context = "\n".join([doc.page_content for doc in docs])
                
                st.text("Invoking model...")
                prompt = f"Context: {context}\n\nHuman: {user_input}\n\nAI:"
                response = model.invoke([HumanMessage(content=prompt)])
                st.text("Model response received.")
                
                ai_message = response.content if hasattr(response, 'content') else str(response)
                
                # Simulate streaming effect
                message_placeholder = st.empty()
                full_response = ""
                for chunk in ai_message.split():
                    full_response += chunk + " "
                    time.sleep(0.05)  # Adjusted for faster response
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
                st.session_state.chat_history.append({"is_user": False, "content": ai_message})
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.text("Traceback:")
                import traceback
                st.text(traceback.format_exc())
        
        st.rerun()
    else:
        st.warning("Please enter a question.")

# st.write("Debug - Current chat history:")
# st.write(st.session_state.chat_history)