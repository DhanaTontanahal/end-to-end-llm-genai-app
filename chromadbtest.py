import chromadb
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="my_collection")
collection.add(
    documents=[
        "This document is about Newyork",
        "This document is about Delhi"
    ],
    ids=["id1", "id2"]
)

all_docs = collection.get()

# results = collection.query(
#     query_texts=["This is a query document about Hawaii"],
#     n_results=2
# )

# results=collection.query(
#     query_texts=['Query is about India'],
#     n_results=2
# )

# print(results)

res=collection.delete(ids=all_docs['ids'])
# collection.get()
print(res)

