from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from summerizer.models import Issue
from summerizer.serializers import IssueSerializer

from summerizer.model.Predict import Predictor

p = Predictor()

# Create your views here.
@api_view(['GET'])
def issue_list(request, format=None):
  """
  List all issues
  """
  if request.method == 'GET':
    issues = Issue.objects.all()
    serializer = IssueSerializer(issues, many=True)
    return Response(serializer.data)
  
  return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def generate_title(request):
  """
  Generate title by deep learning model
  """
  if request.method == 'GET':
    body = request.GET.get('body')
    
    title = None
    if body is not None:
      title = p.generate_title(body)

    return Response({ 'body': body, 'generated_title': title })