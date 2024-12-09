import streamlit as st
import uuid
from rag.document_processing import processing_documents
from rag.graph import graph_architecture
from config.config import Configuration

def stream_response(response):
    for word in response.split():
        yield word + " "
        st.empty()

def main():
    config = Configuration()
    st.title("Banking Chat Assistant")

    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if 'vector_store' not in st.session_state:
        pdf_file = st.file_uploader("Upload PDF Document", type="pdf")
        if pdf_file is not None:
            with open("temp/document.pdf", "wb") as f:
                f.write(pdf_file.getvalue())
            
            with st.spinner('Processing PDF ...'):
                st.session_state.vector_store = processing_documents(
                    document="temp/document.pdf",
                    chunk_size=config.chunk_strategy.chunk_size,
                    chunk_overlap=config.chunk_strategy.chunk_overlap,
                    embedding_model=config.embedding.model,
                    dimension=config.embedding.dimension
                )
                st.session_state.graph = graph_architecture(
                    vector_store=st.session_state.vector_store,
                    search_type=config.vector_search_strategy.method,
                    k=config.vector_search_strategy.k
                )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Tanyakan sesuatu seputar Finansial, Akuntansi, dan Perbankan"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            config_dict = {
                "configurable": {
                    "thread_id": st.session_state.thread_id
                }
            }
            inputs = {
                "messages": [{
                    "role": "human",
                    "content": prompt,
                }]
            }

            response_placeholder = st.empty()

            with st.spinner("Thinking ..."):
                final_state = st.session_state.graph.invoke(input=inputs, config=config_dict)
                response = final_state["messages"][-1].content
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                streamed_response = ""
                for word in stream_response(response):
                    streamed_response += word
                    response_placeholder.markdown(streamed_response)

                for step in st.session_state.graph.stream(input=inputs, stream_mode="values", config=config_dict):
                    step["messages"][-1].pretty_print()

if __name__ == "__main__":
    main()