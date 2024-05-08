import threading
import time
import openai
from threading import Lock
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
import json
from langchain_community.vectorstores import Chroma
import pandas as pd
import sys



class filter_by_value():
    _instance = None
    _lock = Lock()
    _query_df = None
    system_template = f'''Give the names of any location such as city, state, country from the provided sentence in the specified format:
---BEGIN FORMAT TEMPLATE---
{{"CITY":"city"
"STATE":"state of the city"
"COUNTRY": "the country the city and state belong to"}}
---END FORMAT TEMPLATE---
Give the output of the format template in json format
'''
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize_vector_db()
                cls._instance._initialize_query_df()
        return cls._instance
    
    
    def _initialize_vector_db(self):
        self.vector_database = Chroma(persist_directory='/code/CT_VDB/VDB_V_01', embedding_function=OpenAIEmbeddings())
        
    
    def _initialize_query_df(self):
        if filter_by_value._query_df is None:
            filter_by_value._query_df = pd.read_csv('/code/CT_SEARCH_METHODS/hybrid_v1/FinalCTTrialsDF_P1_w_ContactInfo_LocList.csv')

    def filter_by_cancer(self,cancer_type):
        return self._query_df[self._query_df["CONDITIONS"].str.contains(cancer_type, case=False)]
    
    def filter_by_age(self,age):
        return self._query_df[self._query_df["AGE"] == age] 
    
  

    def filter_by_stage(self,eligibility):
        output = {"stage" : "stage",
          "gender" : "gender" }
        system_prompt = """You are helpful assistant to give stage of cancer and gender from given senetence and give output in json format as {output} """        
        response = openai.chat.completions.create(
            model='gpt-4-0125-preview', 
            temperature=0,
            messages=[
                {"role":"system", "content":system_prompt},
                {"role":"user", "content":f""" Extract the stage of cancer and gender from given input {eligibility} and give output in {output} format, stricrtly specify gender as male or female """},
            ],
            max_tokens = 1024,
            response_format={ "type": "json_object" }
            
        )

        res = response.choices[0].message.content
        return res
            
        
      