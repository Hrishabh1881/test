from .views import *
from rest_framework import routers
from django.urls import path, include


urlpatterns = [
    path('clinicaltrialsinfo', ClinicalTrialsLLMView.as_view(), name='clinical-trial-info'),
    path('clinicaltrialsinfo2', ClinicalTrialsLLMViewHybrid.as_view(), name='clinical-trial-info-hybrid'),
    path('clinicaltrialsinfo3', ClinicalTrialsLLMViewHybridLocationList.as_view(), name='clinical-trial-location-list'),
    path('clinicaltrialwithzip', ClinicalTrialsLLMViewHybridZipLocator.as_view(), name='clinical-trial-zip'),
    path('filtertrials', FilterClinicalTrialsDatabseView.as_view(), name='directfilter'),
    path('clinicaltrialsdetails', GetClinicalTrialDetailsView.as_view(), name='detailsview')]