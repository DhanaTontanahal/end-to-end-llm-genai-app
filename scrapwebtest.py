from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

llm = ChatGroq(
    groq_api_key='',
    model="llama-3.1-70b-versatile",
    temperature=0,
)

loader = WebBaseLoader("https://jobs.nike.com/job/R-48680")

page_data = loader.load().pop().page_content

# print(page_data)

prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE
        {page_data}
        ### INSTRUCTION
        The scraped text is from the career's page of a website.
        Your job is to extract the job postings and return them in JSON format
        containing keys :`role` , `experience` , `skills` , `description`.
        Only return valid JSON.
        ### VALID JSON (NO PREAMBLE):
        """
)

chain_extract = prompt_extract | llm
res = chain_extract.invoke(input={'page_data':page_data})
# print(res.content)

json_parser = JsonOutputParser()
json_response=json_parser.parse(res.content)

print(json_response)