import chromadb


def setup_chroma_collections(chunked_docs, embedded_chunks, batch_size=1000):
    """Sets up ChromaDB collections for textual and code chunks.

        Args:
            client: The ChromaDB client instance.
            chunked_docs: Dictionary containing textual and code chunks.
            embedded_chunks: Dictionary containing textual and code embeddings.
            batch_size: The batch size for adding documents (default is 1000).

        Returns:
            A dictionary containing the textual and code collections.
        """
    try:
        client = chromadb.Client()  # Create the client only once

        # Try to delete existing collections
        try:
            client.delete_collection(name="text_collection")
            client.delete_collection(name="code_collection")
        except Exception as e:
            print(f"Warning: Could not delete collections. Error: {e}")

        # Create new collections
        text_collection = client.get_or_create_collection(name="text_collection")
        code_collection = client.get_or_create_collection(name="code_collection")
        
        # Validate input data
        if not chunked_docs or not embedded_chunks:
            raise ValueError("chunked_docs and embedded_chunks cannot be None or empty")
        
        if "textual_chunks" not in chunked_docs or "code_chunks" not in chunked_docs:
            raise ValueError("chunked_docs must contain 'textual_chunks' and 'code_chunks' keys")
            
        if "textual_embeddings" not in embedded_chunks or "code_embeddings" not in embedded_chunks:
            raise ValueError("embedded_chunks must contain 'textual_embeddings' and 'code_embeddings' keys")

        # Add textual chunks to ChromaDB
        textual_chunks = chunked_docs["textual_chunks"]
        textual_embeddings = embedded_chunks["textual_embeddings"]
        
        if len(textual_chunks) != len(textual_embeddings):
            raise ValueError(f"Mismatch: {len(textual_chunks)} textual chunks but {len(textual_embeddings)} embeddings")
            
        for idx, doc in enumerate(textual_chunks):
            try:
                text_collection.add(
                    documents=[doc["content"]],
                    metadatas=[{"file_name": doc["file_name"], "content_type": "text"}],
                    ids=[f"{doc['file_name']}_text_{idx}"],
                    embeddings=textual_embeddings[idx].tolist()
                )
            except Exception as e:
                print(f"Error adding textual document {idx}: {e}")
                raise

        # Add code chunks to ChromaDB
        code_chunks = chunked_docs["code_chunks"]
        code_embeddings = embedded_chunks["code_embeddings"]
        
        if len(code_chunks) != len(code_embeddings):
            raise ValueError(f"Mismatch: {len(code_chunks)} code chunks but {len(code_embeddings)} embeddings")
            
        for idx, doc in enumerate(code_chunks):
            try:
                code_collection.add(
                    documents=[doc["content"]],
                    metadatas=[{"file_name": doc["file_name"], "content_type": "code"}],
                    ids=[f"{doc['file_name']}_code_{idx}"],
                    embeddings=code_embeddings[idx].tolist()
                )
            except Exception as e:
                print(f"Error adding code document {idx}: {e}")
                raise

        return {'textual_collection': text_collection, 'code_collection': code_collection}
        
    except Exception as e:
        print(f"Error in setup_chroma_collections: {e}")
        raise
