from rag import document_processing
from rag.graph import graph_architecture
import streamlit as st
import uuid
import yaml

def main():
   
   st.title("DesiAI: Accounting Knowledge Assistant")
   
   pdf_file = st.file_uploader("Upload PDF Document", type="pdf")
   
   if pdf_file is not None:
      with open("temp/document.pdf", "wb") as f:
         f.write(pdf_file.getvalue())
      
      vector_store = document_processing("data/document.pdf")
      graph = graph_architecture(vector_store)
      
      st.sidebar.header("Chat with DesiAI")
      
      if "messages" not in st.session_state:
         st.session_state.messages = []
      
      for message in st.session_state.messages:
         with st.chat_message(message["role"]):
               st.markdown(message["content"])
      
      if prompt := st.chat_input("Ask a question about Accounting"):
         st.session_state.messages.append({"role": "user", "content": prompt})
         
         with st.chat_message("user"):
               st.markdown(prompt)
         
         with st.chat_message("assistant"):
               config = {
                  "configurable": {
                     "thread_id": str(uuid.uuid4())
                  }
               }
               
               inputs = {
                  "messages": [{
                     "role": "human",
                     "content": prompt,
                  }]
               }
               
               response_placeholder = st.empty()
               full_response = ""
               
               for step in graph.stream(input=inputs, stream_mode="values", config=config):
                  if step and "messages" in step and step["messages"]:
                     full_response = step["messages"][-1].content
                     response_placeholder.markdown(full_response)
               
               st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
   main()