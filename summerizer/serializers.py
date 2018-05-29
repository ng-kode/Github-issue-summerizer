from rest_framework import serializers
from summerizer.models import Issue
# from keras.models import load_model


class IssueSerializer(serializers.Serializer):
  body = serializers.CharField(style={'base_template': 'textarea.html'})
  title = serializers.CharField()

  def create(self, validated_data):
    """
    Create and return a new `Issue` instance, given the validated data.
    """
    # TODO: load model and predict        
    return Issue.objects.create(**validated_data)
 
  def update(self, instance, validated_data):
    """
    Update and return an existing `Issue` instance, given the validated data.
    """
    instance.title = validated_data.get('title', instance.title)
    instance.body = validated_data.get('body', instance.body)
    instance.save()
    return instance