from rest_framework import serializers

class ClinicalTrials_w_ZipSerializer(serializers.Serializer):
    query = serializers.CharField()
    zip_code = serializers.CharField()
    radius = serializers.IntegerField()
    
    
