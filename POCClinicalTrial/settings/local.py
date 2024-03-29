from .base import *

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'doris_local_db',
	    'USER': 'postgres',
	    'PASSWORD': 'postgres',
	    'HOST': 'db',
	    'PORT':5432
    }
}