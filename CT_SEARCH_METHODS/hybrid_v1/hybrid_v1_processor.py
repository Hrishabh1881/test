import threading
import time
import openai
from threading import Lock
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
import json
from langchain_community.vectorstores import Chroma
import pandas as pd
import sys
from uszipcode import SearchEngine
from geopy.distance import geodesic
sys.path.append("/Users/suryabhosale/Documents/projects/DORIS/src/POCClinicalTrial")


class ProcessQuery():
    
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
        if ProcessQuery._query_df is None:
            ProcessQuery._query_df = pd.read_csv('/code/CT_SEARCH_METHODS/hybrid_v1/FinalCTTrialsDF_P1_w_ContactInfo.csv')
    
    def get_nct_scores(self, docs:list) -> dict:
        ct_score_dict = {}
        for doc in docs:
            ct_score_dict[doc[0].metadata['nct_number']] = doc[1]  
        return ct_score_dict  
    
    
    def search_vector_db(self, args, result_dict):
        vector_db = self.vector_database
        result = vector_db.similarity_search_with_relevance_scores(args)
        nct_score_dict = self.get_nct_scores(result)
        result_dict['vector_db_scores_dict'] = nct_score_dict
        result_dict['vector_db_nct_numbers'] = list(nct_score_dict.keys())
        
        
    def get_location(self, system_prompt, user_prompt, model='gpt-4-0125-preview', temperature=0, verbose=False):
        response = openai.chat.completions.create(
            model=model, 
            temperature=temperature,
            messages=[
                {"role":"system", "content":system_prompt},
                {"role":"user", "content":str(user_prompt)},
            ],
            max_tokens = 1024,
            response_format={ "type": "json_object" }
            
        )
        res = response.choices[0].message.content
        return res

    def search_dataframe(self, args, result_dict):
        response = self.get_location(system_prompt=self.system_template, user_prompt=args)
        response_dict = json.loads(response)
        result_dict['location'] = response_dict
        

    @classmethod
    def process_query(cls, query:str):
        result_dict = {}
        vectordb_thread = threading.Thread(target=cls().search_vector_db, args=(query,result_dict))
        smart_df_thread = threading.Thread(target=cls().search_dataframe, args=(query,result_dict))
        vectordb_thread.start()
        smart_df_thread.start()
        vectordb_thread.join()
        smart_df_thread.join()
        query_df = cls._query_df 
        print(query_df)
        print(result_dict)
        
        scores_df = pd.DataFrame.from_dict(result_dict['vector_db_scores_dict'], orient='index', columns=['score'])
        scores_df.reset_index(inplace=True)
        scores_df.columns = ['NCT_NUMBER', 'score']
        
        distilled_df = query_df[query_df['NCT_NUMBER'].isin(result_dict['vector_db_nct_numbers'])]
        distilled_df = pd.merge(distilled_df, scores_df, on='NCT_NUMBER')

        distilled_df = distilled_df.sort_values(by='score', ascending=False)
        
        # distilled_df.to_csv('../result_tests/result.csv')
        
        if len(distilled_df[distilled_df['CITY'] == result_dict['location']['CITY']]):
            distilled_df = distilled_df[distilled_df['CITY'] == result_dict['location']['CITY']]
        elif len(distilled_df[distilled_df['STATE'] == result_dict['location']['STATE']]):
            distilled_df = distilled_df[distilled_df['STATE'] == result_dict['location']['STATE']]
        elif len(distilled_df[distilled_df['COUNTRY'] == result_dict['location']['COUNTRY']]):
            distilled_df = distilled_df[distilled_df['COUNTRY'] == result_dict['location']['COUNTRY']]
        # else:
        #     distilled_df = {}
        drop_cols = [col for col in distilled_df.columns if 'Unnamed' in col]
        distilled_df.drop(columns=drop_cols, axis=1, inplace=True)
        
        # _unexp_location=pd.read_csv('/code/ct_csv/CTRecruiting_LocNoExp_w_ContactInfo.csv')
        # check = _unexp_location[_unexp_location['NCT Number'].isin(distilled_df['NCT_NUMBER'])]
        
        return distilled_df
    
    
    def __init__(self):
        self.return_value = None



class ProcessQueryLocationList():
    
    _instance = None
    _lock = Lock()
    _query_df = None
    system_template = f'''Give the names of any location such as city, state, country from the provided sentence in the specified format. If only a state is given, return the city as the capital of the state.:
---BEGIN FORMAT TEMPLATE---
{{"CITY":"city"
"STATE":"state of the city"
"COUNTRY": "the country the city and state belong to"}}
---END FORMAT TEMPLATE---
Give the output of the format template in json format
'''

    drugs_biomarkers_template=f'''
    Please extract all the names of the drugs and biomarkers from the information provided
    ---BEGIN FORMAT TEMPLATE---
{{"DRUGS":"list of name of the drugs"
"BIOMARKERS":"list of name of the biomarker"}}
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
        if ProcessQueryLocationList._query_df is None:
            ProcessQueryLocationList._query_df = pd.read_csv('/code/CT_SEARCH_METHODS/hybrid_v1/FinalCTTrialsDF_P1_w_ContactInfo_LocList.csv')
    
    def get_nct_scores(self, docs:list) -> dict:
        ct_score_dict = {}
        for doc in docs:
            ct_score_dict[doc[0].metadata['nct_number']] = doc[1]  
        return ct_score_dict  
    
    
    def search_vector_db(self, args, result_dict):
        vector_db = self.vector_database
        result = vector_db.similarity_search_with_relevance_scores(args)
        nct_score_dict = self.get_nct_scores(result)
        result_dict['vector_db_scores_dict'] = nct_score_dict
        result_dict['vector_db_nct_numbers'] = list(nct_score_dict.keys())
        
        
    def get_location(self, system_prompt, user_prompt, model='gpt-4-0125-preview', temperature=0, verbose=False):
        response = openai.chat.completions.create(
            model=model, 
            temperature=temperature,
            messages=[
                {"role":"system", "content":system_prompt},
                {"role":"user", "content":str(user_prompt)},
            ],
            max_tokens = 1024,
            response_format={ "type": "json_object" }
            
        )
        res = response.choices[0].message.content
        return res
    
    def get_drugs_biomarkers(self, system_prompt, user_prompt, model='gpt-4-0125-preview', temperature=0, verbose=False):
        response = openai.chat.completions.create(
            model=model, 
            temperature=temperature,
            messages=[
                {"role":"system", "content":system_prompt},
                {"role":"user", "content":str(user_prompt)},
            ],
            max_tokens = 1024,
            response_format={ "type": "json_object" }
            
        )
        res = response.choices[0].message.content
        return res

    def search_dataframe(self, args, result_dict):
        response = self.get_location(system_prompt=self.system_template, user_prompt=args)
        response_dict = json.loads(response)
        result_dict['location'] = response_dict
    

    def filter_by_stage(self,eligibility):
        output = {"stage" : "stage",
          "gender" : "gender" }
        system_prompt = f"""You are helpful assistant to give stage of cancer and gender from given senetence and give output in json format as {output} """        
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
    
        
        
        
    def get_closest_by_city(self, city):
        distances = []
        search_engine = SearchEngine()
        if len(list(search_engine.by_city(city))) > 0:
            city_info = search_engine.by_city(city)[0]
            center_lat, center_lon = city_info.lat, city_info.lng
            nearby_zip_codes = search_engine.by_coordinates(center_lat, center_lon, returns=0)
            for zip_instance in nearby_zip_codes:
                if zip_instance.lat and zip_instance.lng:
                    distance = geodesic((center_lat, center_lon), (zip_instance.lat, zip_instance.lng)).miles
                    distances.append((zip_instance.zipcode, distance))
            closest_zip_codes = [(item[0], round(item[1], 2)) for item in sorted(distances, key=lambda x: x[1])]
            return closest_zip_codes  
    
    
    def custom_geo_sort(location, closest_zips):
        distances = []
        for loc in location:
            zip_code = loc.get('Location Zip')
            if zip_code in closest_zips:
                distance = closest_zips.index(zip_code)
            else:
                distance = len(closest_zips)
            distances.append((loc, distance))
        sorted_locations = [loc for loc, _ in sorted(distances, key=lambda x: x[1])]
        return sorted_locations 


    @classmethod
    def process_query(cls, query:str):
        result_dict = {}
        vectordb_thread = threading.Thread(target=cls().search_vector_db, args=(query,result_dict))
        smart_df_thread = threading.Thread(target=cls().search_dataframe, args=(query,result_dict))
        vectordb_thread.start()
        smart_df_thread.start()
        vectordb_thread.join()
        smart_df_thread.join()
        query_df = cls._query_df 
        
        print(result_dict['location']['CITY'])

        closest_zip_codes_w_distance = cls().get_closest_by_city(city=result_dict['location']['CITY'])
        closest_zip_codes = [item[0] for item in closest_zip_codes_w_distance]
        print(closest_zip_codes)
        
        scores_df = pd.DataFrame.from_dict(result_dict['vector_db_scores_dict'], orient='index', columns=['score'])
        scores_df.reset_index(inplace=True)
        scores_df.columns = ['NCT_NUMBER', 'score']
        
        distilled_df = query_df[query_df['NCT_NUMBER'].isin(result_dict['vector_db_nct_numbers'])]
        distilled_df = pd.merge(distilled_df, scores_df, on='NCT_NUMBER')

        distilled_df = distilled_df.sort_values(by='score', ascending=False)
        
        drop_cols = [col for col in distilled_df.columns if 'Unnamed' in col]
        distilled_df.drop(columns=drop_cols, axis=1, inplace=True)
        
        filtered_df = pd.DataFrame()
        

        for index, row in distilled_df.iterrows():
            for location in eval(row['LOCATIONS']):
                if (result_dict['location']['CITY'] == location.get('Location City')) \
                    or (result_dict['location']['STATE'] == location.get('Location State')) \
                    or (result_dict['location']['COUNTRY'] == location.get('Location Country')):
                    filtered_df = filtered_df.append(row, ignore_index=True)
                    break
        # drugs = cls().get_drugs_biomarkers(user_prompt=filtered_df['STUDY_TITLE'], system_prompt=cls().drugs_biomarkers_template)
        # print(drugs)
        dnb_result = filtered_df['STUDY_TITLE'].apply(lambda row: cls().get_drugs_biomarkers(user_prompt=row, system_prompt=cls().drugs_biomarkers_template))
        filtered_df['DRUGS_AND_BIOMARKERS'] = dnb_result; filtered_df['DRUGS_AND_BIOMARKERS'].apply(lambda content: eval(content))
        sorted_df_for_location_distance = filtered_df.copy()    
        sorted_df_for_location_distance['LOCATIONS'] = sorted_df_for_location_distance['LOCATIONS'].apply(lambda x: cls.custom_geo_sort(eval(x), closest_zip_codes))
        
        
        
        return sorted_df_for_location_distance
    
    
    def __init__(self):
        self.return_value = None
        
        
        
class ProcessQueryZipLocator():
    
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
        if ProcessQueryZipLocator._query_df is None:
            ProcessQueryZipLocator._query_df = pd.read_csv('/code/CT_SEARCH_METHODS/hybrid_v1/FinalCTTrialsDF_P1_w_ContactInfo_LocList.csv')
            
    
    def get_closest_zip_codes(self, zip_code:int, num_closest:int=500, radius=100):
        distances = []
        search_engine = SearchEngine()
        zip_info = search_engine.by_zipcode(zip_code)
        center_lat, center_lon = zip_info.lat, zip_info.lng
        nearby_zip_codes = search_engine.by_coordinates(center_lat, center_lon, radius=radius, returns=0)
        for zip_instance in nearby_zip_codes:
            if zip_instance.lat and zip_instance.lng:
                distance = geodesic((center_lat, center_lon), (zip_instance.lat, zip_instance.lng)).miles
                distances.append((zip_instance.zipcode, distance))
        closest_zip_codes = [(item[0], round(item[1], 2)) for item in sorted(distances, key=lambda x: x[1])]
        return closest_zip_codes
    
    
    def get_closest_by_city(self, city):
        distances = []
        search_engine = SearchEngine()
        city_info = search_engine.by_city(city)[0]
        center_lat, center_lon = city_info.lat, city_info.lng
        nearby_zip_codes = search_engine.by_coordinates(center_lat, center_lon, returns=0)
        for zip_instance in nearby_zip_codes:
            if zip_instance.lat and zip_instance.lng:
                distance = geodesic((center_lat, center_lon), (zip_instance.lat, zip_instance.lng)).miles
                distances.append((zip_instance.zipcode, distance))
        closest_zip_codes = [(item[0], round(item[1], 2)) for item in sorted(distances, key=lambda x: x[1])]
        return closest_zip_codes  
    
    
    
    def get_nct_scores(self, docs:list) -> dict:
        ct_score_dict = {}
        for doc in docs:
            ct_score_dict[doc[0].metadata['nct_number']] = doc[1]  
        return ct_score_dict  
    
    
    def search_vector_db(self, args, result_dict):
        vector_db = self.vector_database
        result = vector_db.similarity_search_with_relevance_scores(args)
        nct_score_dict = self.get_nct_scores(result)
        result_dict['vector_db_scores_dict'] = nct_score_dict
        result_dict['vector_db_nct_numbers'] = list(nct_score_dict.keys())
        
        
    def get_location(self, system_prompt, user_prompt, model='gpt-4-0125-preview', temperature=0, verbose=False):
        response = openai.chat.completions.create(
            model=model, 
            temperature=temperature,
            messages=[
                {"role":"system", "content":system_prompt},
                {"role":"user", "content":str(user_prompt)},
            ],
            max_tokens = 1024,
            response_format={ "type": "json_object" }
            
        )
        res = response.choices[0].message.content
        return res

    def search_dataframe(self, args, result_dict):
        response = self.get_location(system_prompt=self.system_template, user_prompt=args)
        response_dict = json.loads(response)
        result_dict['location'] = response_dict
        
        
    def custom_geo_sort(location, closest_zips):
        distances = []
        for loc in location:
            zip_code = loc.get('Location Zip')
            if zip_code in closest_zips:
                distance = closest_zips.index(zip_code)
            else:
                distance = len(closest_zips)
            distances.append((loc, distance))
        sorted_locations = [loc for loc, _ in sorted(distances, key=lambda x: x[1])]
        return sorted_locations
        
        
    
    @classmethod
    def process_query(cls, query:str, zip_code:int, radius:int = 100):
        result_dict = {}
        vectordb_thread = threading.Thread(target=cls().search_vector_db, args=(query,result_dict))
        smart_df_thread = threading.Thread(target=cls().search_dataframe, args=(query,result_dict))
        vectordb_thread.start()
        smart_df_thread.start()
        vectordb_thread.join()
        smart_df_thread.join()
        query_df = cls._query_df 
        
        
        closest_zip_codes_w_distance = cls().get_closest_zip_codes(zip_code=zip_code)
        closest_zip_codes = [item[0] for item in closest_zip_codes_w_distance]
        
        # if result_dict['location'].get('CITY', None) == '':
        #     closest_zip_codes_w_distance = cls().get_closest_zip_codes(zip_code=zip_code)
        #     closest_zip_codes = [item[0] for item in closest_zip_codes_w_distance]
        # else:
        #     closest_zip_codes_w_distance = cls().get_closest_by_city(city=result_dict['location']['CITY'])
        #     closest_zip_codes = [item[0] for item in closest_zip_codes_w_distance]
            

        
        scores_df = pd.DataFrame.from_dict(result_dict['vector_db_scores_dict'], orient='index', columns=['score'])
        scores_df.reset_index(inplace=True)
        scores_df.columns = ['NCT_NUMBER', 'score']
        
        distilled_df = query_df[query_df['NCT_NUMBER'].isin(result_dict['vector_db_nct_numbers'])]
        distilled_df = pd.merge(distilled_df, scores_df, on='NCT_NUMBER')

        print(result_dict)

        distilled_df = distilled_df.sort_values(by='score', ascending=False)        
        # distilled_df = distilled_df.dropna(subset=['ZIP'])
        
        filtered_df = pd.DataFrame()
        
        for index, row in distilled_df.iterrows():
            for location in eval(row['LOCATIONS']):
                if (location.get('Location Zip') in closest_zip_codes \
                    and location.get('Location Country') == 'United States'):
                    filtered_df = filtered_df.append(row, ignore_index=True)
                    break
                
        
        sorted_df_for_location_distance = filtered_df.copy()  
              
        if not sorted_df_for_location_distance.empty:
            sorted_df_for_location_distance['LOCATIONS'] = sorted_df_for_location_distance['LOCATIONS'].apply(lambda x: cls.custom_geo_sort(eval(x), closest_zip_codes))
            drop_cols = [col for col in sorted_df_for_location_distance.columns if 'Unnamed' in col]
            sorted_df_for_location_distance.drop(columns=drop_cols, axis=1, inplace=True)
            return sorted_df_for_location_distance
        else:
            return pd.DataFrame()
        
        #NOTE: This is for using csv with location exploded into different rows:
        
        # filtered_df = distilled_df[distilled_df['ZIP'].isin(closest_zip_codes)]
        # drop_cols = [col for col in filtered_df.columns if 'Unnamed' in col]
        # filtered_df.drop(columns=drop_cols, axis=1, inplace=True)
        # filtered_df = filtered_df[filtered_df['COUNTRY'] == 'United States']
        
    
    
    def __init__(self):
        self.return_value = None
        
        
        

        
        
# if __name__ == '__main__':
#     items = ProcessQuery.process_query("What are potential clinical trial options for a patient with metastatic breast cancer in Irvine area?")
#     print(items)