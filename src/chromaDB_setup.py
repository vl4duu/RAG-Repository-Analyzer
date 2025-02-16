def setup_chroma_collection(chunked_docs, collection, embeddings):
    for idx, doc in enumerate(chunked_docs):
        collection.add(
            documents=[doc["content"]],
            metadatas=[{"file_name": doc["file_name"]}],
            ids=[f"{doc['file_name']}_{idx}"],
            embeddings=[embeddings[idx].tolist()]
        )