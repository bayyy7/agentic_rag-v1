# Agentic RAG with Langchain

## 🌟 Project Overview

This is the implementation of using Google Gemini 1.5 Pro with langchain as framework. Allowing LLM to give an answer based on the given context (pdf). Also implemented using Langgrapph, a powerful state from Langchain, allowing developer to create custom flow or architecture, deliver with Chat Memory

## ✨ Features

- 📄 PDF Document Upload
- 🤖 AI-Powered Knowledge Retrieval
- 💬 Interactive Chat Interface
- 🔍 Semantic Document Search

## 🛠 Tech Stack

- **Language Model**: Google Gemini 1.5 Pro
- **Framework**: 
  - Streamlit
  - LangChain
- **Embedding**: Sentence Transformers
- **Vector Store**: FAISS
- **Programming Language**: Python 3.8+

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Langchain
- Google Generative AI API Key

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/desi-ai-accounting-assistant.git
cd desi-ai-accounting-assistant
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure environment variables
- Create a `.env` file in the project root
- Add your Google API key:
```
GOOGLE_GENERATIVE_AI=your_google_api_key_here
```
5. Create your system prompt
- open `prompt` folder
- create new python file `system_prompt.py`
```
def system_prompt(tool_messages):
   """
   Generate the system prompt content.
   """
   docs_content = "\n\n".join(doc.content for doc in tool_messages)
   return (
      "[YOUR PROMPT HERE]"
      f"{docs_content}\n\n"
)
```

### Running the Application

```bash
streamlit run app.py
```
### Custom Config
You can directly change the configuration on the `config/config.py`. There are several example you can change by what you want. Also be careful when changes the Embedding Model, you must known the dimension of the Embedding it self. This code below is the helper to know the size of embedding dimension.
- Using the `embed_query` function
```bash
vector = embeddings.embed_query("aiueo")
matrix = numpy.array(vector).astype('float32')
len(matrix)
```
- Using the `embed_document` function
```bash
vector = embeddings.embed_documents(str("aiueo"))
matrix = numpy.array(vector).astype('float32')
matrix.shape[1]
```