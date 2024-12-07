# 📚 DesiAI: Accounting Knowledge Assistant 🧠

## 🌟 Project Overview

DesiAI is an intelligent AI-powered accounting knowledge assistant that leverages advanced language models and document retrieval techniques to provide insightful answers from uploaded PDF documents. Built with Streamlit, LangChain, and Google's Gemini AI, this application transforms how you interact with complex accounting materials.

![Project Banner](https://via.placeholder.com/800x300.png?text=DesiAI+Accounting+Knowledge+Assistant)

## ✨ Features

- 📄 PDF Document Upload
- 🤖 AI-Powered Knowledge Retrieval
- 💬 Interactive Chat Interface
- 🔍 Semantic Document Search
- 🧠 Contextual Response Generation

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

### Running the Application

```bash
streamlit run app.py
```

## 🔍 How It Works

1. Upload an accounting-related PDF
2. Ask questions about the document
3. Receive AI-generated responses with source references

## 📘 Usage Example

```python
# Upload your accounting textbook PDF
# Ask questions like:
# - "What is SAK ETAP?"
# - "Explain the concept of balance sheet"
# - "How are financial statements structured?"
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🏆 Acknowledgments

- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [Google AI](https://ai.google/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

---

**Developed with ❤️ by Your Name**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/yourusername)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-blue)](https://twitter.com/yourusername)