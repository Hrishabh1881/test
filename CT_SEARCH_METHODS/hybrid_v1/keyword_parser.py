import sys
if '/code/CT_SEARCH_METHODS/' not in sys.path:
    sys.path.append('/code/CT_SEARCH_METHODS/')
print(sys.path)
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai.chat_models import ChatOpenAI
from hybrid_v1.config import *

class GradeDocuments(BaseModel):
    
    keyword_list: list = Field(description="list of keywords for clinical trials search")
    
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=OPEN_API_KEY)
structured_llm_parser = llm.with_structured_output(GradeDocuments)

system = """You are a medicine and oncology keyword extractor. \n 
    You will look at the string and parse or extract all the important key words \n
    Give a list of important keywords from the string. Only give keywords from the string, do not make up adjascent keywords"""
    
    
grade_prompt = ChatPromptTemplate.from_messages(
    [('system', system),
    ('human', "User query: {query}")]
)

keyword_extractor = grade_prompt | structured_llm_parser

