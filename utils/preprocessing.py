import pandas_gpt
import pandas as pd
import openai
import os
import numpy as np
import json


class ct_preprocessing:
    
    df = None
    
    @classmethod
    def create_ct_gpt_df(cls, path:str, desired_keys:list)-> pd.DataFrame():
        
        if cls.df is not None:
            return cls.df
        
        cls.df = pd.DataFrame()
        count = 0
        for file in os.listdir(path):
            count += 1
            print('JSON LOADED: {}'.format(count))
            with open(f'{path}/{file}') as f:
                json_data = json.load(f)
            filtered_json_data = {key: json_data.get(key) for key in desired_keys}
            temp_df = pd.DataFrame([filtered_json_data])
            if len(temp_df['Locations'][0]) > 1:
                exp_location = temp_df.explode('Locations')
                cls.df = pd.concat([cls.df,exp_location])
            else:
                cls.df = pd.concat([cls.df,temp_df])
        return cls.df 
    
    def convert_list_to_dicts(self, cell):
        if isinstance(cell, list):
            temp_dict = {}
            for d in cell:
                if isinstance(d, dict):
                    temp_dict.update(d)
            return temp_dict
        return cell

    
    def json_normalize_column(self, df:pd.DataFrame(), column_name:str):
        if column_name in df.keys():
            normalized_df = pd.json_normalize(df[column_name])
            concat_df = pd.concat([df.reset_index(drop=True), normalized_df.reset_index(drop=True)],axis=1)
            concat_df.drop(column_name, axis=1, inplace=True)
            return concat_df
        else:
            pass
        
    
    
    
    def replace_or_rename_column(self, df:pd.DataFrame(), replace:bool=True, new_col_name:str=None, current_col_name:str=None, word_to_replace:str=None):
        if replace:
            try:
                for col in df.columns:
                    if word_to_replace in col:
                        new_col_name = col.replace(word_to_replace, '')
                        df.rename(columns={col: new_col_name.strip()}, inplace=True)
                    else:
                        pass
                return df
            except Exception as e:
                print(f'Error: {e}')
                return e 
        else:
            try:
                df.rename(columns={current_col_name: new_col_name.strip()}, inplace=True)
                return df
            except Exception as e:
                print(f'Error: {e}')
                return e 
            
            
    def unwrap_column(self, df, col_name:str):
        if col_name in df.keys():
            max_conditions = max(df[col_name].apply(len))
            for i in range(1, max_conditions + 1):
                new_column_name = f'{col_name} {i}'
                df[new_column_name] = df[f'{col_name}'].apply(lambda x: x[i-1] if len(x) >= i else None)
            df.drop(axis=1, columns=col_name, inplace=True)
            return df
        else:
            pass
        
    
    def get_cleaned_csv_if_exists(cls, ct_type):
        if cls.df is not None:
            return cls.df
        cls.df = pd.DataFrame()
        with open('/code/ct_csv/trials.json') as ct_json:
            ct_json = json.load(ct_json)
        if os.path.exists(ct_json[ct_type]):
            cls.df = pd.read_csv(ct_json['Breast Cancer'])
            return cls.df
        else:
            raise Exception(f"CSV file for '{ct_type}' does not exist.")
