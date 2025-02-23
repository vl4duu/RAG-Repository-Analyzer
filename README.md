This project demonstrates a Retrieval Augmented Generation (RAG) system for answering questions about a GitHub repository. It leverages ChromaDB for efficient vector similarity search and OpenAI's embedding models for representing textual and code data.
## Project Overview
The system processes a GitHub repository, extracts textual and code chunks, generates embeddings for each chunk using OpenAI's `text-embedding-ada-002` and CodeBERT, and stores them in ChromaDB collections. When a user asks a question, the system retrieves the most relevant chunks based on semantic similarity, constructs a prompt incorporating the retrieved context, and uses OpenAI's `gpt-3.5-turbo` to generate an answer.
## System Architecture
1. **GitHub Data Parsing:** The `github_parser.py` script uses the GitHub API to retrieve files from a specified repository. It then splits the content into chunks, differentiating between textual and code files.
2. **Embedding Generation:** The `embedding.py` script generates embeddings for both textual and code chunks. Textual embeddings are created using OpenAI's `text-embedding-ada-002`, while code embeddings are generated using CodeBERT.
3. **ChromaDB Storage:** The `chromaDB_setup.py` script sets up two ChromaDB collections: one for textual embeddings and one for code embeddings. The embeddings and corresponding metadata are added to these collections.
4. **Query Processing:** The `helper.ipynb` notebook contains the core logic for answering user questions. It retrieves the most similar chunks from ChromaDB, constructs a RAG prompt, and uses OpenAI's `gpt-3.5-turbo` to generate the answer.

## Setup
1. **Install Dependencies:**
``` bash
pip install -r requirements.txt
```
1. **Set API Keys:** Create a `.env` file and add your OpenAI and GitHub API keys:
``` 
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
GITHUB_API_KEY="YOUR_GITHUB_API_KEY"
```
1. **Run the Notebook:** Execute the `helper.ipynb` notebook. This will parse the specified GitHub repository, generate embeddings, create ChromaDB collections, and allow you to ask questions.

## Usage
The `helper.ipynb` notebook provides a simple interface for querying the system. You can modify the `query` variable to test different questions. The system will return the AI's response based on the context found in the repository.
## Future Improvements
- More sophisticated chunk splitting strategies.
- Improved prompt engineering for better AI responses.
- Integration with other knowledge sources.
- Enhanced error handling and logging.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.
