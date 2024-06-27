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
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from doris_schema.models import *
from uszipcode import SearchEngine
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

from CT_SEARCH_METHODS.hybrid_v1 import hybrid_v1_processor
os.environ['OPENAI_API_KEY'] = OPEN_API_KEY 



class ClinicalTrialsLLMViewHybridZipLocator(CreateAPIView):
    
    """
    API view to handle clinical trial queries with ZIP code and radius for location-based filtering.
    Uses a hybrid processor to fetch relevant clinical trial information based on the query and location.
    """
    
    serializer_class = ClinicalTrials_w_ZipSerializer
    
    def post (self, request, *args, **kwargs):
        query = request.data.get('query', '')
        zipcode = request.data.get('zip_code', None)
        radius = request.data.get('radius', 120)
        if zipcode:
            payload, unique_biomarkers, unique_drugs = hybrid_v1_processor.ProcessQueryZipLocator.process_query(query=query, zip_code=zipcode, radius=radius)
            if query:
                if payload.empty:
                    json_payload_dict = {}
                else:
                    json_payload = pd.DataFrame(payload, columns=payload.columns).to_json(orient='records')
                    json_payload_dict = json.loads(json_payload)  
                return JsonResponse({
                    'Message':json_payload_dict,
                    'BIOMARKERS':unique_biomarkers,
                    'DRUGS':unique_drugs
                    })
            else:
                if payload.empty:
                    json_payload_dict = {}
                else:
                    json_payload = pd.DataFrame(payload, columns=payload.columns).to_json(orient='records')
                    json_payload_dict = json.loads(json_payload)  
                return JsonResponse({
                    'Message':json_payload_dict,
                    'BIOMARKERS':unique_biomarkers,
                    'DRUGS':unique_drugs
                    })
        else:
            return JsonResponse({'Message':'Zip code not entered'}, status=400)
        


class GetClinicalTrialDetailsView(APIView):
    
    authentication_classes=[TokenAuthentication,]
    permission_classes=[IsAuthenticated,]
    """
    API view to fetch detailed information about clinical trials based on NCT number.
    Utilizes a singleton pattern to ensure a single instance of the view.
    Extracts drugs and biomarkers from the trial interventions using GPT-4 model.
    """

    _instance = None
    
    search_param = openapi.Parameter(
        'nct_number', 
        openapi.IN_QUERY,
        description="nct_number to get clinical trial specific information",
        type=openapi.TYPE_STRING
)
    

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.Filter = filter_by_value()

    def get_drugs_biomarkers(self, user_prompt, model='gpt-4-0125-preview', temperature=0, verbose=False):
        """
        Extracts names of drugs and biomarkers from the provided user prompt using GPT-4.
        
        Args:
            user_prompt (str): The input text containing information about the clinical trial interventions.
            model (str): The model name to be used for the extraction. Defaults to 'gpt-4-0125-preview'.
            temperature (int): The temperature setting for the model. Defaults to 0.
            verbose (bool): Whether to run in verbose mode. Defaults to False.

        Returns:
            str: JSON-formatted string containing the extracted drugs and biomarkers.
        """
        system_prompt = '''
        Please extract all the names of the drugs and biomarkers from the information provided
        ---BEGIN FORMAT TEMPLATE---
        {"DRUGS":"list of name of the drugs"
        "BIOMARKERS":"list of name of the biomarker"}
        ---END FORMAT TEMPLATE---
        Give the output of the format template in json format
        '''
        
        response = openai.chat.completions.create(
            model=model, 
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(user_prompt)},
            ],
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        
        res = response.choices[0].message.content
        return res

    @swagger_auto_schema(manual_parameters=[search_param])
    def get(self, request, *args, **kwargs):
        
        def calculate_distance(my_zipcode, location):
        
            def get_coordinates(zip_code):
                search_engine = SearchEngine()
                zip_info = search_engine.by_zipcode(zip_code)
                center_lat, center_lon = zip_info.lat, zip_info.lng
                return (center_lat, center_lon)

            loc_w_dist = []
            
            coords_1 = get_coordinates(my_zipcode)
            for loc in location:
                if len(loc.get('Location Zip')) == 5:
                    zip_code = loc.get('Location Zip')
                    coords_2 = get_coordinates(zip_code)
                    distance = geodesic(coords_1, coords_2).miles
                    loc['Distance'] = round(distance, 0)
                    loc_w_dist.append(loc)
                else:
                    loc['Distance'] = float('inf')
            return loc_w_dist
        
        
        # Extract filter parameters from the request query parameters
        filter_params = dict(request.query_params)
        nct_number = filter_params.get('nct_number', [])
        
        # Get user zip - a.) to sort the locations
        
        user_zip = Preference.objects.get(user_id_id=self.request.user.user_id).zip_code
        
        if nct_number:
            # Filter the payload by NCT number
            payload = self.Filter.filter_by_nct_number(nct_number=nct_number)
            
            if not payload.empty:
                # Drop columns with 'Unnamed' in their name
                drop_cols = [col for col in payload.columns if 'Unnamed' in col]
                payload.drop(columns=drop_cols, axis=1, inplace=True)
                
                # Extract drugs and biomarkers for each row in 'INTERVENTIONS'
                dnb_result = payload['INTERVENTIONS'].apply(lambda row: self.get_drugs_biomarkers(user_prompt=row))
                payload['DRUGS_AND_BIOMARKERS'] = dnb_result
                payload['DRUGS_AND_BIOMARKERS'].apply(lambda content: eval(content))
                
                # Convert string content to list if necessary
                payload['PHASES'] = payload['PHASES'].apply(lambda content: eval(content) if isinstance(content, str) else content)
                payload['LOCATIONS'] = payload['LOCATIONS'].apply(lambda content: eval(content) if isinstance(content, str) else content)
                payload['LOCATIONS'].apply(lambda loc: calculate_distance(location=loc, my_zipcode=user_zip))
                payload['LOCATIONS'] = payload['LOCATIONS'].apply(lambda x: sorted(x, key=lambda y: y['Distance']))
                payload['CONDITIONS'] = payload['CONDITIONS'].apply(lambda content: eval(content) if isinstance(content, str) else content)
                
                
                payload.drop('ZIP_STR', axis=1, inplace=True)
                # Convert the payload to JSON format and parse it into a dictionary
                json_payload = pd.DataFrame(payload, columns=payload.columns).to_json(orient='records')
                json_payload_dict = json.loads(json_payload)
                
                # Update each item in the dictionary with extracted drugs and biomarkers
                for item in json_payload_dict:
                    drugs_and_biomarkers = json.loads(item['DRUGS_AND_BIOMARKERS'])
                    item.update(drugs_and_biomarkers)
                    del item['DRUGS_AND_BIOMARKERS']
                
                # Return the filtered trials as JSON response
                return JsonResponse({'filtered_trials': json_payload_dict}, status=200)
            else:
                # Return an error message if the trial information is unavailable
                return JsonResponse({'message': f'Information about trial {nct_number[0]} currently unavailable'}, status=400)         
            
              

    
        
        