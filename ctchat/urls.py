from .views import *
from rest_framework import routers
from django.urls import path, include


urlpatterns = [
    path('clinicaltrialwithzip', ClinicalTrialsLLMViewHybridZipLocator.as_view(), name='clinical-trial-zip'),
    path('clinicaltrialsdetails', GetClinicalTrialDetailsView.as_view(), name='detailsview')]