from langchain_groq import ChatGroq

llm = ChatGroq(
    groq_api_key='',
    model="llama-3.1-70b-versatile",
    temperature=0,
)

response = llm.invoke("The first person to land on sun")
print(response)