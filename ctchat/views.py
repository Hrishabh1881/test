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
                
                request.session['current_trial_data'] = json_payload_dict
                print(request.session['current_trial_data'])
            return JsonResponse({'Message':json_payload_dict})
        else:
            return JsonResponse({'Message':{}})
        
        
    def get(self, request, *args, **kwargs):
        payload = request.session.get('current_trial_data',[])
        if payload:
            
            ##DRUGS & BIOMARKERS FOR POST QUERY FILTERING
            drugs = request.query_params.getlist('drugs', [])
            biomarkers = request.query_params.getlist('biomarkers', [])
            if drugs:
                if len(drugs[0].split(',')) > 1:
                    drugs = drugs[0].split(',')
            if biomarkers:
                if len(biomarkers[0].split(',')) > 1:
                    biomarkers = biomarkers[0].split(',')
            
            ##ADD PARAMS HERE
            if not drugs and not biomarkers:
                return JsonResponse({'filtered_trials': payload})

            filtered_trials = []
            for trial in payload:
                if any(drug in trial['DRUGS'] for drug in drugs) or any(biomarker in trial['BIOMARKERS'] for biomarker in biomarkers):
                    filtered_trials.append(trial)
        
        
            if filtered_trials:
                return JsonResponse({'filtered_trials': filtered_trials}, status=200)
            else:
                return JsonResponse({'message':'No trails match the filter parameters'}, status=400)
            
        
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
        
        

class FilterClinicalTrialsDatabseView(APIView):
    
    def get(self, request, *args, **kwargs):
        payload = dict(request.query_params)
        return JsonResponse(dict(self.request.query_params))
        
        