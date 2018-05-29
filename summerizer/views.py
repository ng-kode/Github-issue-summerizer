from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from summerizer.models import Issue
from summerizer.serializers import IssueSerializer

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