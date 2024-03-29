from rest_framework import serializers

class DorisChatSerializer(serializers.Serializer):
    query = serializers.CharField()
    
    
