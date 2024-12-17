# PDF_Chatbot_Project
This project is an interactive chatbot application powered by a large language model (LLM) for document-based question answering. The main goal was to create a tool that allows users to upload PDF documents and interactively ask questions about the content, leveraging advanced natural language processing techniques.

## **Features**

- Upload text-based PDFs and scanned PDFs.
- Extract content using **PyPDF2** for text PDFs and **Tesseract OCR** for scanned PDFs.
- Ask questions and receive context-based answers using the **`deepset/roberta-base-squad2` model** from Hugging Face.
- Efficient document retrieval using **FAISS** embeddings for fast search and answer extraction.


## **Technologies Used**

- **Python**: Programming language.
- **Flask**: Web framework to create the chatbot API.
- **Hugging Face Transformers**: For question-answering models.
- **FAISS**: For vector-based search and retrieval.
- **Tesseract OCR**: For extracting text from scanned documents.
- **PyPDF2**: For reading text-based PDFs.
- **LangChain**: For simplifying document processing and retrieval.


## **Work Flow**
-Upload a PDF document.
-Ask questions about its content in natural language.
-Receive precise answers based on the document's context.
