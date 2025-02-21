import chromadb


def setup_chroma_collections(chunked_docs, embedded_chunks):
    text_dir = "./chromaDB/text/collection/"
    code_dir = "./chromaDB/code/collection/"

    client = chromadb.Client()  # Create the client only once
    text_collection = client.get_or_create_collection(name="text_collection")
    code_collection = client.get_or_create_collection(name="code_collection")

    # Add to ChromaDB (separate collections)
    for idx, doc in enumerate(chunked_docs["textual_chunks"]):
        text_collection.add(
            documents=[doc["content"]],
            metadatas=[{"file_name": doc["file_name"], "content_type": "text"}],
            ids=[f"{doc['file_name']}_{idx}"],
            embeddings=[embedded_chunks["textual_embeddings"][idx].tolist()]  # corrected to tolist()
        )

    for idx, doc in enumerate(chunked_docs["code_chunks"]):
        code_collection.add(
            documents=[doc["content"]],
            metadatas=[{"file_name": doc["file_name"], "content_type": "code"}],
            ids=[f"{doc['file_name']}_{idx}"],
            embeddings=[embedded_chunks["code_embeddings"][idx].tolist()]  # corrected to tolist()
        )
 # Removed extra brackets

    return {'text_collection': text_collection, 'code_collection': code_collection}
