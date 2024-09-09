import sys
if '/code/CT_SEARCH_METHODS/' not in sys.path:
    sys.path.append('/code/CT_SEARCH_METHODS/')
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
from CT_SEARCH_METHODS.hybrid_v1.data_dedplication_helper import create_attribute_mapping
sys.path.append("POCClinicalTrial")
import os
from copy import deepcopy
from textblob import TextBlob
#OPENAI_API_KEY removed from here
# os.environ["OPENAI_API_KEY"] = ""

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
        self.vector_database = Chroma(persist_directory='/code/CT_VDB/VDB_V_02_ALPHA_TITLE', embedding_function=OpenAIEmbeddings(openai_api_key="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"))
        
    
    def _initialize_corss_encoder(self):
        from sentence_transformers import CrossEncoder
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    
    def _initialize_query_df(self):
        if ProcessQueryZipLocator._query_df is None:
            ProcessQueryZipLocator._query_df = pd.read_csv('/code/ct_csv/CT_CSV_23_05_ONLYUS_ZIPSTR.csv')
            self.nct_filter_df = pd.read_csv('/code/ct_csv/CT_CSV_23_05_ONLYUS_ZIPSTR.csv')
            deduplication_fields = ["BIOMARKERS", "DRUGS"]
            field_mapping = create_attribute_mapping(ProcessQueryZipLocator._query_df, deduplication_fields)
            ProcessQueryZipLocator.deduplication_mapping = field_mapping
            # ProcessQueryZipLocator.remove_duplicates_in_response(["HER2", " estrogen receptor", " RCB"], "BIOMARKERS")

    @classmethod
    def remove_duplicates_in_response(cls, value_in, feild_name):
        if isinstance(value_in, str):
            response_value = eval(value_in)
        else:
            response_value = value_in
        if isinstance(response_value, list):
            with open("CT_SEARCH_METHODS/hybrid_v1/global_deduplication_mapping.json", "r") as fin:
                mapping_json = json.loads(fin.read())
            field_mapping_data = mapping_json[feild_name]
            temp_query_mapping = deepcopy(cls.deduplication_mapping)
            temp_query_mapping[feild_name].update({k.lower():v for k,v in field_mapping_data.items()})
            final_response_value = []
            # final_response_value = [temp_query_mapping[each_val.strip().lower()] if each_val.strip().lower() in temp_query_mapping else each_val for each_val in response_value]
            for each_val in response_value:
                if each_val.strip().lower() in temp_query_mapping[feild_name]:
                    final_response_value.append(temp_query_mapping[feild_name][each_val.strip().lower()])
                else:
                    each_val
            cls.deduplication_mapping = temp_query_mapping            
            return final_response_value
        return value_in

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
        print('CURRENT RADIUS', radius)
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
            
            # Create a sub-vector database consisting of documents of NCT_NUMBERS that are within a user defined geo-radius (default is 120 miles). 
            for zips in df['ZIP_STR'].apply(lambda x: x.split() if isinstance(x, str) else []):
                masks.append(self.has_overlap(zip_codes, zips))
            zipped_nct_list = list(df[masks]['NCT_NUMBER'])
            nct_filter = {"nct_number": {"$in": zipped_nct_list}}
            vector_db = self.vector_database
            retriever = vector_db.as_retriever(search_type='similarity', search_kwargs={'k': 300, 'filter': nct_filter})
            result_docs = retriever.invoke(args)
            chroma_client = chromadb.PersistentClient(path='/code/CT_VDB/VDB_V_02_ALPHA_TITLE')
            
            
            
            # If a collection of a given zipcode and the user defined radius has not been created, a collection will be created using
            # documents from the result_docs
            if f'{zip_codes[0]}_{str(radius)}' not in [col_obj.name for col_obj in chroma_client.list_collections()]:
                print(f'-----CREATING {zip_codes[0]}- radius_{str(radius)} COLLECTION-----')
                collection = chroma_client.create_collection(name=f'{zip_codes[0]}_{str(radius)}', metadata={"hnsw:space": "cosine"})
                collection.upsert(
                    documents=[doc.page_content for doc in result_docs],
                    metadatas=[doc.metadata for doc in result_docs],
                    ids=[doc.metadata['nct_number'] for doc in result_docs]
                )
            # Else the collection is retrieved 
            else:
                print(f'-----RETRIEVED {zip_codes[0]}- radius_{str(radius)} COLLECTION-----')
                collection = chroma_client.get_collection(name=f'{zip_codes[0]}_{str(radius)}')

            # If filtering keyword is present, a query is run against the collection to filter the most relevant clinical trials
            if args != '':
                args = str(TextBlob(args).correct())
                print(f'KEYWORD INPUT: {args}')
                from hybrid_v1.keyword_parser import keyword_extractor
                keywords = keyword_extractor.invoke({"query":args}).keyword_list
                print(keywords)
                if len(keywords) > 1:
                    nct_number_list, nct_relevance_scores = collection.query(query_texts=[args],
                                                                        where_document={"$and":[{"$contains": word} for word in keywords]},
                                                                        n_results=300,
                                                                        ).get('ids')[0],collection.query(query_texts=[args],  where_document={"$or":[{"$contains": word} for word in keywords]}, n_results=300).get('distances')[0]
                else:
                    nct_number_list, nct_relevance_scores = collection.query(query_texts=[args],
                                                                        where_document={"$contains": args},
                                                                        n_results=300,
                                                                        ).get('ids')[0],collection.query(query_texts=[args],  where_document={"$contains": args}, n_results=300).get('distances')[0]

                
                if len(nct_number_list) == 0:
                    #NOTE: HOW DO WE WANT TO HANDLE IT IF A KEYWORD IS INPUT BUT TRIAL LIST IS 0
                    nct_number_list, nct_relevance_scores = collection.query(query_texts=[args],
                                                                        n_results=300,
                                                                        ).get('ids')[0],collection.query(query_texts=[args], n_results=300).get('distances')[0]
                # NOTE: USE BELOW IF USING CUSTOM CROSS ENCODER
                # ==================================================================================================
                # ranker_results = self.doc_ranker(query=args, text_chunks=[doc.page_content for doc in result_docs], topN=len(result_docs))
                # ranked_nct_numbers = [json.loads(doc)['nct_number'] for doc in ranker_results]
                # result_dict['vector_db_nct_numbers'] = ranked_nct_numbers
                # print('reranked_results')
                # from pprint import pprint
                
                # print(nct_number_list)
                result_dict['vector_db_nct_numbers'] = nct_number_list
                
                # NOTE: USE BELOW IF USING CUSTOM CROSS ENCODER
                # ==================================================================================================
                result_dict['nct_scores'] = {item[0]:item[1] for item in zip(nct_number_list, nct_relevance_scores)}
            
            # Else all the clinical trials within a 120 mile radius of the zip code are returned
            else:
                print(f'KEYWORD INPUT: NOT ENTERED')
                result_dict['vector_db_nct_numbers'] = zipped_nct_list
                result_dict['nct_scores'] = []
        else:
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
        print(query, zip_code, radius)
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
        distilled_df = query_df[query_df['NCT_NUMBER'].isin(result_dict['vector_db_nct_numbers'])]
        distilled_df.set_index('NCT_NUMBER', inplace=True)
        distilled_df = distilled_df.reindex(result_dict['vector_db_nct_numbers'])
        distilled_df.reset_index(inplace=True)
        # if result_dict['nct_scores']:
        #     distilled_df['REL_SCORES'] = distilled_df['NCT_NUMBER'].map(result_dict['nct_scores'])
        #     distilled_df = distilled_df[distilled_df.REL_SCORES < 0.55]
        
        # NOTE: UNCOMMENT THE CODE BELOW IF USING EXTERNAL CROSS ENCODER
        # ==================================================================================================
        # if result_dict['nct_scores']:
        #     distilled_df['REL_SCORES'] = distilled_df['NCT_NUMBER'].map(result_dict['nct_scores'])
        #     distilled_df = distilled_df.sort_values(by='REL_SCORES', ascending=True)
        # distilled_df = pd.merge(distilled_df, scores_df, on='NCT_NUMBER')


                
        
        sorted_df_for_location_distance = distilled_df.copy()  
        
        if not sorted_df_for_location_distance.empty:
            sorted_df_for_location_distance['LOCATIONS'] = sorted_df_for_location_distance['LOCATIONS'].apply(lambda x: cls.filter_zip_codes(eval(x), closest_zip_codes))
            sorted_df_for_location_distance['LOCATIONS'] = sorted_df_for_location_distance['LOCATIONS'].apply(lambda x: cls.custom_geo_sort(x, closest_zip_codes))
            sorted_df_for_location_distance['PHASES'] = sorted_df_for_location_distance['PHASES'].apply(lambda element: eval(element) if isinstance(element, str) else element)
            sorted_df_for_location_distance['CONDITIONS'] = sorted_df_for_location_distance['CONDITIONS'].apply(lambda element: eval(element))   
            sorted_df_for_location_distance['LOCATIONS'] = sorted_df_for_location_distance['LOCATIONS'].apply(lambda element: eval(element) if isinstance(element, str) else element)
            # sorted_df_for_location_distance['DRUGS'] = sorted_df_for_location_distance['DRUGS'].apply(lambda element: eval(element) if isinstance(element, str) else element)
            # sorted_df_for_location_distance['BIOMARKERS'] = sorted_df_for_location_distance['BIOMARKERS'].apply(lambda element: eval(element) if isinstance(element, str) else element)
            sorted_df_for_location_distance['DRUGS'] = sorted_df_for_location_distance['DRUGS'].apply(cls.remove_duplicates_in_response, args=("DRUGS",))
            sorted_df_for_location_distance['BIOMARKERS'] = sorted_df_for_location_distance['BIOMARKERS'].apply(cls.remove_duplicates_in_response, args=("BIOMARKERS",))
            sorted_df_for_location_distance['POINT_OF_CONTACT'] = sorted_df_for_location_distance['POINT_OF_CONTACT'].apply(lambda element: eval(element) if isinstance(element, str) else element)
            drop_cols = [col for col in sorted_df_for_location_distance.columns if 'Unnamed' in col]
            drop_cols.extend(['ZIP_STR'])
            sorted_df_for_location_distance.drop(columns=drop_cols, axis=1, inplace=True)
            
            # NOTE: CURRENTLY CALCULATING EUCLIDEAN DISTANCE 
            # ==================================================================================================
            sorted_df_for_location_distance['LOCATIONS'] = sorted_df_for_location_distance['LOCATIONS'].apply(lambda loc: cls.calculate_distance(location=loc, my_zipcode=zip_code))
            # sorted_df_for_location_distance["BIOMARKERS"] = cls.deduplication_mapping
            biomarker_list = [each_val for val in sorted_df_for_location_distance["BIOMARKERS"] for each_val in val]
            drugs_list = [each_val for val in sorted_df_for_location_distance["DRUGS"] for each_val in val]
            unique_biomarkers = sorted(list(set(biomarker_list)), key = lambda x:x.lower())
            unique_drugs = sorted(list(set(drugs_list)), key = lambda x:x.lower())
            return sorted_df_for_location_distance, unique_biomarkers, unique_drugs
        else:
            return pd.DataFrame(), [], []
        
        # NOTE: THIS IS FOR USING CSV WITH LOCATION EXPLODED INTO DIFFERENT ROWS
        # ==================================================================================================
        # filtered_df = distilled_df[distilled_df['ZIP'].isin(closest_zip_codes)]
        # drop_cols = [col for col in filtered_df.columns if 'Unnamed' in col]
        # filtered_df.drop(columns=drop_cols, axis=1, inplace=True)
        # filtered_df = filtered_df[filtered_df['COUNTRY'] == 'United States']
        
    
    
    def __init__(self):
        self.return_value = None
        
        
        

        
#NOTE: TESTING
# ==================================================================================================       
# if __name__ == '__main__':
#     items = ProcessQueryZipLocator.process_query("breast cancer clinical trials for HER2/Neu Negative biomarkers", 60631)
#     print(items)