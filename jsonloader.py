from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os


def add_metadata(record: dict, metadata: dict):
    print('Helllo')
    if 'call_count' not in add_metadata.__dict__:
        add_metadata.call_count = 0
    add_metadata.call_count += 1
    
    
    if add_metadata.call_count > 1:
        metadata['trial_id'] = record.get('NCT Number')
        metadata["source"] = 'Clinical Trials'
        # metadata['trial_title'] = record.get('Study Title')
        # metadata['page_link'] = record.get('Study URL')
        if 'seq_num' in metadata:
            metadata.pop('seq_num')
    return metadata





loader = JSONLoader(
    file_path='/Users/suryabhosale/Documents/projects/DORIS/src/POCClinicalTrial/eligibility_criteria_test.json',
    text_content=False,
    jq_schema='.clinical_trial[]',
    metadata_func=add_metadata
)

data = loader.load()
from pprint import pprint 
print(loader._metadata_func.__init__())
# for datum in data:
#     datum.metadata['salutation'] = 'hello'
#     print(datum)