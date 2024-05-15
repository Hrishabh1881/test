from django.shortcuts import render
import pandas as pd
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from .serializers import *
import os
from rest_framework.generics import CreateAPIView, ListAPIView
from rest_framework.views import APIView
from django.http import JsonResponse
import openai 
import json
from utils.preprocessing import ct_preprocessing
from django.core.cache import cache
from utils.config import *
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from utils.config import *
from utils.filter import *

from CT_SEARCH_METHODS.hybrid_v1 import hybrid_v1_processor
os.environ['OPENAI_API_KEY'] = OPEN_API_KEY 


# Create your views here.
class ClinicalTrialsLLMView(CreateAPIView):
    # permission_classes = (IsAuthenticated,)
    # authentication_classes = (TokenAuthentication,)
    serializer_class = DorisChatSerializer
    
    df = None
    converted_list_to_dict = None
    
    @classmethod
    def get_df(self, desired_keys):
        if self.df is not None:
            return self.df

        count = 0
        self.df = pd.DataFrame()

        for file in os.listdir('/code/clinical_trials/breast_clinial_trials'):
            count += 1
            print('JSON LOADED: {}'.format(count))
            with open(f'/code/clinical_trials/breast_clinial_trials/{file}') as f:
                json_data = json.load(f)
            filtered_json_data = {key: json_data.get(key) for key in desired_keys}
            temp_df = pd.DataFrame([filtered_json_data])
            if len(temp_df['Locations'][0]) > 1:
                exp_location = temp_df.explode('Locations')
                self.df = pd.concat([self.df, exp_location])
            else:
                self.df = pd.concat([self.df, temp_df])
        return self.df
    
    def convert_list_to_dicts(self, cell):
        if isinstance(cell, list):
            temp_dict = {}
            for d in cell:
                if isinstance(d, dict):
                    temp_dict.update(d)
            self.converted_list_to_dict = True
            return temp_dict
        return cell
    
    def strip_location(self, df):
        for col in df.columns:
            if 'Location' in col:
                new_col_name = col.replace('Location', '')
                df.rename(columns={col: new_col_name.strip()}, inplace=True)
        return df

    def normalize_cols(self, df):
        location_normalized = pd.json_normalize(df['Locations'])
        df = pd.concat([df.reset_index(drop=True), location_normalized.reset_index(drop=True)],axis=1)
        df.drop('Locations', axis=1, inplace=True)
        condition_normalized  = pd.json_normalize(df['Conditions'])
        condition_normalized['Condition'] = condition_normalized['Condition'].apply(lambda x : x[0])
        df = pd.concat([df, condition_normalized], axis=1);df.drop('Conditions', axis=1, inplace=True)
        return df
    
    @classmethod
    def make_df_if_csv_exists(self, llm):
        if self.df is not None:
            # self.s_df = SmartDataframe(self.df, config={"llm":llm})
            return self.s_df
        if os.path.exists('/code/ct_csv/bc_ct.csv'):
            print('Here')
            self.df = pd.read_csv('/code/ct_csv/bc_ct.csv', usecols=lambda x: 'Unnamed' not in x)
            self.s_df = SmartDataframe(self.df, config={"llm":llm})
            return self.s_df
        else:
            raise Exception(f"CSV file for does not exist.")
    
    def post(self, request, *args, **kwargs):
        llm = OpenAI(api_token=OPEN_API_KEY, model_name='gpt-4-0125-preview')
        self.df = self.make_df_if_csv_exists(llm=llm)
        query = request.data.get('query', [])
        if query:
            openai.api_key = OPEN_API_KEY
            payload = self.df.chat(query)
            chopped_payload = payload.head(10)
            if payload.empty:
                json_payload_dict = {}
            else:
                json_payload = pd.DataFrame(chopped_payload, columns=payload.columns).to_json(orient='records')
                json_payload_dict = json.loads(json_payload)
                
            return JsonResponse({'Message':json_payload_dict})
        else:
            return JsonResponse({'Message':{}})
        
        
class ClinicalTrialsLLMViewHybrid(CreateAPIView):
    serializer_class = DorisChatSerializer
    def post(self, request, *args, **kwargs):
        query = request.data.get('query', [])
        payload = hybrid_v1_processor.ProcessQuery.process_query(query)
        if query:
            if payload.empty:
                json_payload_dict = {}
            else:
                json_payload = pd.DataFrame(payload, columns=payload.columns).to_json(orient='records')
                json_payload_dict = json.loads(json_payload)
                
            return JsonResponse({'Message':json_payload_dict})
        else:
            return JsonResponse({'Message':{}})
        
        
class ClinicalTrialsLLMViewHybridLocationList(CreateAPIView):
    
    permission_classes = (IsAuthenticated,)
    authentication_classes = (TokenAuthentication,)
    serializer_class = DorisChatSerializer
    
    def post(self, request, *args, **kwargs):
        query = request.data.get('query', [])
        
        payload = hybrid_v1_processor.ProcessQueryLocationList.process_query(query=query)
        
        if query:
            if payload.empty:
                json_payload_dict = {}
            else:
                json_payload = pd.DataFrame(payload, columns=payload.columns).to_json(orient='records')
                json_payload_dict = json.loads(json_payload)
                for item in json_payload_dict:
                    drugs_and_biomarkers = json.loads(item['DRUGS_AND_BIOMARKERS'])
                    item.update(drugs_and_biomarkers)
                    del item['DRUGS_AND_BIOMARKERS']
                
                
                cache.set(f'clinical_trials/{self.request.user.user_id}', json_payload_dict, timeout=180)
                # request.session[session_key] = json_payload_dict

            return JsonResponse({'Message':json_payload_dict})
        else:
            return JsonResponse({'Message':{}})
        
        
    def get(self, request, *args, **kwargs):
        
        
        def dict_to_tuple(d):
            if isinstance(d, dict):
                return tuple((k, dict_to_tuple(v)) for k, v in sorted(d.items()))
            elif isinstance(d, list):
                return tuple(sorted([dict_to_tuple(item) for item in d]))
            else:
                return d
            
        def convert_to_hashable(value):
                if isinstance(value, list):
                    return tuple(value)
        
        
        
        payload = cache.get(f'clinical_trials/{self.request.user.user_id}')
        payload_df = pd.DataFrame(payload)
        
        
              
        if payload:
            ##DRUGS & BIOMARKERS FOR POST QUERY FILTERING
            drugs = request.query_params.getlist('drugs', [])
            biomarkers = request.query_params.getlist('biomarkers', [])
            zip_code = request.query_params.get('zip_code')
            print(zip_code)
            
            if drugs:
                if len(drugs[0].split(',')) > 1:
                    drugs = drugs[0].split(',')
            if biomarkers:
                if len(biomarkers[0].split(',')) > 1:
                    biomarkers = biomarkers[0].split(',')
            
            ##ADD PARAMS HERE
            if not drugs and not biomarkers and not zip_code:
                return JsonResponse({'filtered_trials': payload})

            filtered_trials = []
            for trial in payload:
                if any(drug in trial['DRUGS'] for drug in drugs) or any(biomarker in trial['BIOMARKERS'] for biomarker in biomarkers):
                    filtered_trials.append(trial)
                    
            if zip_code:
                filter = filter_by_value()
                zip_filtered_trials = filter.filter_current_by_zipcode(df=payload_df, zipcode=zip_code)
                drop_cols = [col for col in zip_filtered_trials.columns if 'Unnamed' in col]
                zip_filtered_trials.drop(columns=drop_cols, axis=1, inplace=True)
                json_payload = pd.DataFrame(zip_filtered_trials, columns=zip_filtered_trials.columns).to_json(orient='records')
                json_payload_dict = json.loads(json_payload)
                filtered_trials.append(json_payload_dict)
        
        
            if filtered_trials:
                # # unique_dicts = {tuple((k, convert_to_hashable(v)) for k, v in d.items()) for d in filtered_trials}
                # # filtered_trials = [dict((k, list(v)) if isinstance(v, tuple) else (k, v) for k, v in items) for items in unique_dicts]
                # # unique_dicts = [dict(t) for t in {dict_to_tuple(d) for d in filtered_trials}]
                # unique_tuples = {dict_to_tuple(d) for d in filtered_trials}
                # print(unique_tuples)
                # unique_dicts = [dict(t) for t in unique_tuples]
                return JsonResponse({'filtered_trials': filtered_trials}, status=200)
            else:
                return JsonResponse({'message':'No trails match the filter parameters', 'filtered_trials':payload}, status=400)
        
        

class ClinicalTrialsLLMViewHybridZipLocator(CreateAPIView):
    
    serializer_class = DorisChatSerializer
    
    def post (self, request, *args, **kwargs):
        query = request.data.get('query', [])
        zipcode = request.data.get('zip_code', None)
        radius = request.data.get('zip_code', None)
        if zipcode:
            payload = hybrid_v1_processor.ProcessQueryZipLocator.process_query(query=query, zip_code=zipcode)
            if query:
                if payload.empty:
                    json_payload_dict = {}
                else:
                    json_payload = pd.DataFrame(payload, columns=payload.columns).to_json(orient='records')
                    json_payload_dict = json.loads(json_payload)  
                return JsonResponse({'Message':json_payload_dict})
            else:
                return JsonResponse({'Message':{}})
        else:
            return JsonResponse({'Message':'Zip code not entered'})
        


class GetClinicalTrialDetailsView(APIView):
    _instance = None  

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.Filter = filter_by_value()
        
        
        
    def get_drugs_biomarkers(self, user_prompt, model='gpt-4-0125-preview', temperature=0, verbose=False):
        
        system_prompt=f'''
    Please extract all the names of the drugs and biomarkers from the information provided
    ---BEGIN FORMAT TEMPLATE---
{{"DRUGS":"list of name of the drugs"
"BIOMARKERS":"list of name of the biomarker"}}
---END FORMAT TEMPLATE---
Give the output of the format template in json format
    '''
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
    
    def get(self, request, *args, **kwargs):
        filter_params = dict(request.query_params) 
        nct_number = filter_params.get('nct_number', [])
        if nct_number:
            payload = self.Filter.filter_by_nct_number(nct_number=nct_number)
            if not payload.empty:
                drop_cols = [col for col in payload.columns if 'Unnamed' in col]
                payload.drop(columns=drop_cols, axis=1, inplace=True)
                dnb_result = payload['INTERVENTIONS'].apply(lambda row: self.get_drugs_biomarkers(user_prompt=row))
                payload['DRUGS_AND_BIOMARKERS'] = dnb_result; payload['DRUGS_AND_BIOMARKERS'].apply(lambda content: eval(content))
                json_payload = pd.DataFrame(payload, columns=payload.columns).to_json(orient='records')
                json_payload_dict = json.loads(json_payload)
                for item in json_payload_dict:
                    drugs_and_biomarkers = json.loads(item['DRUGS_AND_BIOMARKERS'])
                    item.update(drugs_and_biomarkers)
                    del item['DRUGS_AND_BIOMARKERS']
                return JsonResponse({'filtered_trials':json_payload_dict}, status=200)
            else:
                return  JsonResponse({'message':f'Information about trial {nct_number[0]} currently unavailable'}, status=400)
            
            
              

class FilterClinicalTrialsDatabseView(APIView):
    
    _instance = None  

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.Filter = filter_by_value()
    
    def get(self, request, *args, **kwargs):
        filter_params = dict(request.query_params)
        
        
        filtered_trials = []
                
        
        if 'condition_type' in filter_params.keys():
            conditions = filter_params.get('condition_type',[])
            for condition in conditions: 
               results = self.Filter.filter_by_cancer(cancer_type=condition)
               drop_cols = [col for col in results.columns if 'Unnamed' in col]
               results.drop(columns=drop_cols, axis=1, inplace=True)
               json_payload = pd.DataFrame(results, columns=results.columns).to_json(orient='records')
               json_payload_dict = json.loads(json_payload)
               filtered_trials.append(json_payload_dict)  
               
        if 'zipcode' in filter_params.keys():
            zipcodes = filter_params.get('zipcode',[])
            for zipcode in zipcodes:
                zip_code_filter = self.Filter.filter_by_zipcode(zipcode=zipcode)
                drop_cols = [col for col in zip_code_filter.columns if 'Unnamed' in col]
                zip_code_filter.drop(columns=drop_cols, axis=1, inplace=True)
                json_payload = pd.DataFrame(zip_code_filter, columns=zip_code_filter.columns).to_json(orient='records')
                json_payload_dict = json.loads(json_payload)
                filtered_trials.append(json_payload_dict)
        
        return JsonResponse({'filtered_trials':filtered_trials})
    
        
        