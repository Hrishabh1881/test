from rest_framework import serializers

class IntOrStrField(serializers.Field):
    def to_internal_value(self, data):
        if isinstance(data, int):
            return str(data)
        elif isinstance(data, str):
            return data
        else:
            raise serializers.ValidationError('This field must be a string or an integer.')

    def to_representation(self, value):
        return value
class ClinicalTrials_w_ZipSerializer(serializers.Serializer):
    query = serializers.CharField(required=False)
    zip_code = IntOrStrField(required=True)
    radius = serializers.IntegerField(required=False)
    
    
