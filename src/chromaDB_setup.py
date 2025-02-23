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

    client = chromadb.Client()  # Create the client only once
    text_collection = client.get_or_create_collection(name="text_collection")
    code_collection = client.get_or_create_collection(name="code_collection")
    # Add to ChromaDB (separate collections)
    for idx, doc in enumerate(chunked_docs["textual_chunks"]):
        print()
        text_collection.add(
            documents=[doc["content"]],
            metadatas=[{"file_name": doc["file_name"], "content_type": "text"}],
            ids=[f"{doc['file_name']}_{idx}"],
            embeddings=embedded_chunks["textual_embeddings"][idx].tolist()  # corrected to tolist()
        )

    for idx, doc in enumerate(chunked_docs["code_chunks"]):
        code_collection.add(
            documents=[doc["content"]],
            metadatas=[{"file_name": doc["file_name"], "content_type": "code"}],
            ids=[f"{doc['file_name']}_{idx}"],
            embeddings=embedded_chunks["code_embeddings"][idx].tolist()  # corrected to tolist()
        )
 # Removed extra brackets

    return {'textual_collection': text_collection, 'code_collection': code_collection}
