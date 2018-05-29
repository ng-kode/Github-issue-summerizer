from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from summerizer.models import Issue
from summerizer.serializers import IssueSerializer

from summerizer.model.Predict import Predictor

# load the model
p = Predictor()

# template views
def index(request):
  context = { 'testing': 'testing here' }
  return render(request, 'summerizer/index.html', context)

# api
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