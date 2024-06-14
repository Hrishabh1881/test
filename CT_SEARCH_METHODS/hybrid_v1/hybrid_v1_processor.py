import threading
import time
import openai
from threading import Lock
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
import json
from langchain_community.vectorstores import Chroma
import chromadb
import pandas as pd
import sys
from uszipcode import SearchEngine
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from chromadb import Settings
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
        self.vector_database = Chroma(persist_directory='/code/CT_VDB/VDB_V_02_ALPHA', embedding_function=OpenAIEmbeddings())
        
    
    def _initialize_query_df(self):
        if ProcessQueryLocationList._query_df is None:
            ProcessQueryLocationList._query_df = pd.read_csv('/code/ct_csv/CT_CSV_17_05.csv')
            
            
    def get_nct_scores(self, docs:list) -> dict:
        ct_score_dict = {}
        for doc in docs:
            ct_score_dict[doc[0].metadata['nct_number']] = doc[1]  
        return ct_score_dict  
    
    
    def search_vector_db(self, args, result_dict):
        vector_db = self.vector_database
        result = vector_db.similarity_search_with_relevance_scores(args, k=10)
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
        
        
        if result_dict['location']['CITY']:

            closest_zip_codes_w_distance = cls().get_closest_by_city(city=result_dict['location']['CITY'])
            closest_zip_codes = [item[0] for item in closest_zip_codes_w_distance]
            print(closest_zip_codes)
            
            scores_df = pd.DataFrame.from_dict(result_dict['vector_db_scores_dict'], orient='index', columns=['score'])
            print(scores_df)
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
                        or ('United States' == location.get('Location Country')):
                        filtered_df = filtered_df.append(row, ignore_index=True)
                        break
            # drugs = cls().get_drugs_biomarkers(user_prompt=filtered_df['STUDY_TITLE'], system_prompt=cls().drugs_biomarkers_template)
            # print(drugs)
            if not filtered_df.empty:
                dnb_result = filtered_df['INTERVENTIONS'].apply(lambda row: cls().get_drugs_biomarkers(user_prompt=row, system_prompt=cls().drugs_biomarkers_template))
                filtered_df['DRUGS_AND_BIOMARKERS'] = dnb_result; filtered_df['DRUGS_AND_BIOMARKERS'].apply(lambda content: eval(content))
                sorted_df_for_location_distance = filtered_df.copy()    
                sorted_df_for_location_distance['LOCATIONS'] = sorted_df_for_location_distance['LOCATIONS'].apply(lambda x: cls.custom_geo_sort(eval(x), closest_zip_codes))
                sorted_df_for_location_distance['PHASES'] = sorted_df_for_location_distance['PHASES'].apply(lambda element: eval(element) if isinstance(element, str) else element)
                sorted_df_for_location_distance['CONDITIONS'] = sorted_df_for_location_distance['CONDITIONS'].apply(lambda element: eval(element) if isinstance(element, str) else element)
                sorted_df_for_location_distance['POINT_OF_CONTACT'] = sorted_df_for_location_distance['POINT_OF_CONTACT'].apply(lambda element: eval(element) if isinstance(element, str) else element)
                
                return sorted_df_for_location_distance
            else:
                return filtered_df
        
        else:
            scores_df = pd.DataFrame.from_dict(result_dict['vector_db_scores_dict'], orient='index', columns=['score'])
            scores_df.reset_index(inplace=True)
            scores_df.columns = ['NCT_NUMBER', 'score']
            distilled_df = query_df[query_df['NCT_NUMBER'].isin(result_dict['vector_db_nct_numbers'])]
            distilled_df = pd.merge(distilled_df, scores_df, on='NCT_NUMBER')
            distilled_df = distilled_df.sort_values(by='score', ascending=False)
            drop_cols = [col for col in distilled_df.columns if 'Unnamed' in col]
            distilled_df.drop(columns=drop_cols, axis=1, inplace=True)
            dnb_result = distilled_df['INTERVENTIONS'].apply(lambda row: cls().get_drugs_biomarkers(user_prompt=row, system_prompt=cls().drugs_biomarkers_template))
            distilled_df['DRUGS_AND_BIOMARKERS'] = dnb_result; distilled_df['DRUGS_AND_BIOMARKERS'].apply(lambda content: eval(content))
            sorted_df_for_location_distance = distilled_df.copy() 
            sorted_df_for_location_distance['PHASES'] = sorted_df_for_location_distance['PHASES'].apply(lambda element: eval(element) if isinstance(element, str) else element)
            sorted_df_for_location_distance['CONDITIONS'] = sorted_df_for_location_distance['CONDITIONS'].apply(lambda element: eval(element))   
            sorted_df_for_location_distance['LOCATIONS'] = sorted_df_for_location_distance['LOCATIONS'].apply(lambda element: eval(element) if isinstance(element, str) else element)
            sorted_df_for_location_distance['POINT_OF_CONTACT'] = sorted_df_for_location_distance['POINT_OF_CONTACT'].apply(lambda element: eval(element) if isinstance(element, str) else element)
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
                cls._instance._initialize_corss_encoder()
                cls._instance._initialize_query_df()
        return cls._instance
    
    
    def _initialize_vector_db(self):
        # chroma_client = chromadb.Client()
        # client = chromadb.PersistentClient(path="./chroma")
        # chroma_client = chromadb.Client(Settings(persist_directory='/code/CT_VDB/VDB_V_02_ALPHA'))
        # self.search_vector_db = chroma_client
        self.vector_database = Chroma(persist_directory='/code/CT_VDB/VDB_V_02_ALPHA', embedding_function=OpenAIEmbeddings())
        
    
    def _initialize_corss_encoder(self):
        from sentence_transformers import CrossEncoder
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    
    def _initialize_query_df(self):
        if ProcessQueryZipLocator._query_df is None:
            ProcessQueryZipLocator._query_df = pd.read_csv('/code/ct_csv/CT_CSV_23_05_ONLYUS_ZIPSTR.csv')
            self.nct_filter_df = pd.read_csv('/code/ct_csv/CT_CSV_23_05_ONLYUS_ZIPSTR.csv')
            

    def calculate_distance(my_zipcode, location):
        
        def get_coordinates(zip_code):
            search_engine = SearchEngine()
            zip_info = search_engine.by_zipcode(zip_code)
            center_lat, center_lon = zip_info.lat, zip_info.lng
            return (center_lat, center_lon)

        loc_w_dist = []
        
        coords_1 = get_coordinates(my_zipcode)
        for loc in location:
            zip_code = loc.get('Location Zip')
            coords_2 = get_coordinates(zip_code)
            distance = geodesic(coords_1, coords_2).miles
            loc['Distance'] = round(distance, 0)
            loc_w_dist.append(loc)
        return loc_w_dist
            
    
    def get_closest_zip_codes(self, zip_code:int, radius, num_closest:int=50):
        print('radius in zip codes', radius)
        distances = []
        search_engine = SearchEngine()
        zip_info = search_engine.by_zipcode(zip_code)
        if zip_info:
            center_lat, center_lon = zip_info.lat, zip_info.lng
            nearby_zip_codes = search_engine.by_coordinates(center_lat, center_lon, radius=radius, returns=0)
            for zip_instance in nearby_zip_codes:
                if zip_instance.lat and zip_instance.lng:
                    distance = geodesic((center_lat, center_lon), (zip_instance.lat, zip_instance.lng)).miles
                    distances.append((zip_instance.zipcode, distance))
            closest_zip_codes = [(item[0], round(item[1], 2)) for item in sorted(distances, key=lambda x: x[1])]
            print(len(closest_zip_codes))
            return closest_zip_codes
        else:
            return []
    
    
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
    
    
    def contains_number(self, row, zip_list):
        if isinstance(row, str):
            if len(row.split()) > 0:
                # print(zip_list, 'zip split')
                # print(bool(set(zip_list) in set(row.split())))
                return bool(set(zip_list) in set(row.split()))
        else: 
            return False
        
    def has_overlap(self, row, zip_list):
        set1 = set(row)
        set2 = set(zip_list)
        return not set1.isdisjoint(set2)
    
    def get_drugs_biomarkers(system_prompt, user_prompt, model='gpt-4-0125-preview', temperature=0, verbose=False):
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
    
    
    def doc_ranker(self, query, text_chunks, topN:int):
        import numpy as np
        reranker = self.cross_encoder
        scores = reranker.predict([[query, doc] for doc in text_chunks])
        top_indices = np.argsort(scores)[::-1][:topN]
        top_pairs = [text_chunks[index] for index in top_indices]
        return top_pairs
        
    
    def search_vector_db(self, args, result_dict, zip_codes, radius):
        if zip_codes:
            chroma_client = chromadb.Client()
            df = self.nct_filter_df
            masks = []
            for zips in df['ZIP_STR'].apply(lambda x: x.split() if isinstance(x, str) else []):
                masks.append(self.has_overlap(zip_codes, zips))
            zipped_nct_list = list(df[masks]['NCT_NUMBER'])
            # zipped_nct_list = list(df[df['ZIP_STR'].apply(lambda x: self.has_overlap(x, zip_codes))]['NCT_NUMBER'])
            nct_filter = {"nct_number": {"$in": zipped_nct_list}}
            vector_db = self.vector_database
            retriever = vector_db.as_retriever(search_type='similarity', search_kwargs={'k': 300, 'filter': nct_filter})
            result_docs = retriever.invoke(args)
            chroma_client = chromadb.PersistentClient(path='/code/CT_VDB/VDB_V_02_ALPHA')
            
           
            # print([eval(doc).get('nct_number') for doc in ranker_results])
            #NCT01570998
            
            
            
            if f'{zip_codes[0]}_{str(radius)}' not in [col_obj.name for col_obj in chroma_client.list_collections()]:
                print(f'-----CREATING {zip_codes[0]}- radius_{str(radius)} COLLECTION-----')
                collection = chroma_client.create_collection(name=f'{zip_codes[0]}_{str(radius)}', metadata={"hnsw:space": "cosine"})
                collection.upsert(
                    documents=[doc.page_content for doc in result_docs],
                    metadatas=[doc.metadata for doc in result_docs],
                    ids=[doc.metadata['nct_number'] for doc in result_docs]
                )
            else:
                print(f'-----ACQUIRED {zip_codes[0]}- radius_{str(radius)} COLLECTION-----')
                collection = chroma_client.get_collection(name=f'{zip_codes[0]}_{str(radius)}')
            # # if collection.name in [col_obj.name for col_obj in chroma_client.list_collections()]:
            # print(args, 'this is args')
            if args != '':
                print('in here')
                nct_number_list, nct_relevance_scores = collection.query(query_texts=[args], 
                                                                        n_results=300,
                                                                        ).get('ids')[0],collection.query(query_texts=[args], n_results=300).get('distances')[0]
                # nct_number_list = [doc.metadata['nct_number'] for doc in result_docs]
                # result = vector_db.similarity_search_with_relevance_scores(args, k=10)
                # nct_score_dict = self.get_nct_scores(result)
                # result_dict['vector_db_scores_dict'] = nct_score_dict
                
                # ranker_results = self.doc_ranker(query=args, text_chunks=[doc.page_content for doc in result_docs], topN=len(result_docs))
                # ranked_nct_numbers = [json.loads(doc)['nct_number'] for doc in ranker_results]
                # result_dict['vector_db_nct_numbers'] = ranked_nct_numbers
                # print('reranked_results')
                # from pprint import pprint
                
                # print(nct_number_list)
                result_dict['vector_db_nct_numbers'] = nct_number_list
                # result_dict['nct_scores'] = {item[0]:item[1] for item in zip(nct_number_list, nct_relevance_scores)}
            else:
                result_dict['vector_db_nct_numbers'] = zipped_nct_list
                result_dict['nct_scores'] = []
        else:
            print('GOING HERE')
            result_dict['vector_db_nct_numbers'] = []
            result_dict['nct_scores'] = []
            
        
        
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
        
    def filter_zip_codes(location_list, zip_codes):
        return [location for location in location_list if location["Location Zip"] in zip_codes]
    
    @classmethod
    def process_query(cls, query, zip_code:int, radius:int = 120):
        print('user provided radius: {}'.format(radius))
        result_dict = {}
        closest_zip_codes_w_distance = cls().get_closest_zip_codes(zip_code=zip_code, radius=radius)
        closest_zip_codes = [item[0] for item in closest_zip_codes_w_distance]
        vectordb_thread = threading.Thread(target=cls().search_vector_db, args=(query, result_dict, closest_zip_codes, radius))
        smart_df_thread = threading.Thread(target=cls().search_dataframe, args=(query,result_dict))
        vectordb_thread.start()
        smart_df_thread.start()
        vectordb_thread.join()
        smart_df_thread.join()
        query_df = cls._query_df 
        
        print(cls._query_df.columns)
        
        
        # closest_zip_codes_w_distance = cls().get_closest_zip_codes(zip_code=zip_code, radius=)
        # closest_zip_codes = [item[0] for item in closest_zip_codes_w_distance]
        
        # closest_zip_codes_w_distance_dict = {zip_code:dist for zip_code, dist in closest_zip_codes_w_distance}
        # print(closest_zip_codes_w_distance_dict)
        
        # if result_dict['location'].get('CITY', None) == '':
        #     closest_zip_codes_w_distance = cls().get_closest_zip_codes(zip_code=zip_code)
        #     closest_zip_codes = [item[0] for item in closest_zip_codes_w_distance]
        # else:
        #     closest_zip_codes_w_distance = cls().get_closest_by_city(city=result_dict['location']['CITY'])
        #     closest_zip_codes = [item[0] for item in closest_zip_codes_w_distance]
            

        
        # scores_df = pd.DataFrame.from_dict(result_dict['vector_db_scores_dict'], orient='index', columns=['score'])
        # scores_df.reset_index(inplace=True)
        # scores_df.columns = ['NCT_NUMBER', 'score']
        

        distilled_df = query_df[query_df['NCT_NUMBER'].isin(result_dict['vector_db_nct_numbers'])]
        distilled_df.set_index('NCT_NUMBER', inplace=True)
        distilled_df = distilled_df.reindex(result_dict['vector_db_nct_numbers'])
        distilled_df.reset_index(inplace=True)
        # if result_dict['nct_scores']:
        #     distilled_df['REL_SCORES'] = distilled_df['NCT_NUMBER'].map(result_dict['nct_scores'])
        #     distilled_df = distilled_df.sort_values(by='REL_SCORES', ascending=True)
        # distilled_df = pd.merge(distilled_df, scores_df, on='NCT_NUMBER')

        # print(distilled_df['LOCATIONS'])

        # distilled_df = distilled_df.sort_values(by='score', ascending=False)        
        # # distilled_df = distilled_df.dropna(subset=['ZIP'])
        
        # filtered_df = pd.DataFrame()
        
        # for index, row in distilled_df.iterrows():
        #     for location in eval(row['LOCATIONS']):
        #         if (location.get('Location Zip') in list(closest_zip_codes_w_distance_dict.keys()) \
        #             and location.get('Location Country') == 'United States'):
        #             filtered_df = filtered_df.append(row, ignore_index=True)
        #             break
                
        
        sorted_df_for_location_distance = distilled_df.copy()  

        print(sorted_df_for_location_distance, 'here here')
        
        if not sorted_df_for_location_distance.empty:
            print('in here')
            sorted_df_for_location_distance['LOCATIONS'] = sorted_df_for_location_distance['LOCATIONS'].apply(lambda x: cls.filter_zip_codes(eval(x), closest_zip_codes))
            sorted_df_for_location_distance['LOCATIONS'] = sorted_df_for_location_distance['LOCATIONS'].apply(lambda x: cls.custom_geo_sort(x, closest_zip_codes))
            sorted_df_for_location_distance['PHASES'] = sorted_df_for_location_distance['PHASES'].apply(lambda element: eval(element) if isinstance(element, str) else element)
            sorted_df_for_location_distance['CONDITIONS'] = sorted_df_for_location_distance['CONDITIONS'].apply(lambda element: eval(element))   
            sorted_df_for_location_distance['LOCATIONS'] = sorted_df_for_location_distance['LOCATIONS'].apply(lambda element: eval(element) if isinstance(element, str) else element)
            sorted_df_for_location_distance['DRUGS'] = sorted_df_for_location_distance['DRUGS'].apply(lambda element: eval(element) if isinstance(element, str) else element)
            sorted_df_for_location_distance['BIOMARKERS'] = sorted_df_for_location_distance['BIOMARKERS'].apply(lambda element: eval(element) if isinstance(element, str) else element)
            sorted_df_for_location_distance['POINT_OF_CONTACT'] = sorted_df_for_location_distance['POINT_OF_CONTACT'].apply(lambda element: eval(element) if isinstance(element, str) else element)
            drop_cols = [col for col in sorted_df_for_location_distance.columns if 'Unnamed' in col]
            drop_cols.extend(['ZIP_STR'])
            sorted_df_for_location_distance.drop(columns=drop_cols, axis=1, inplace=True)
            # dnb_result = sorted_df_for_location_distance['INTERVENTIONS'].apply(lambda row: cls.get_drugs_biomarkers(user_prompt=row, system_prompt=cls.drugs_biomarkers_template))
            # sorted_df_for_location_distance['DRUGS_AND_BIOMARKERS'] = dnb_result; sorted_df_for_location_distance['DRUGS_AND_BIOMARKERS'].apply(lambda content: eval(content))
            
            ### NEED TO WORK ON THIS ONE! OR MAKE IT BETTER
            sorted_df_for_location_distance['LOCATIONS'] = sorted_df_for_location_distance['LOCATIONS'].apply(lambda loc: cls.calculate_distance(location=loc, my_zipcode=zip_code))
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