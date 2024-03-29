from .base import *

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'postgres', #os.environ.get('POSTGRES_NAME')
	    'USER': 'doris', #os.environ.get('POSTGRES_USER')
	    'PASSWORD': 'postgresdoris', #os.environ.get('POSTGRES_PASSWORD')
	    'HOST':'doris-dev.covbkwii8k5d.us-east-1.rds.amazonaws.com', #db
	    'PORT':5432, #5433
    }

}