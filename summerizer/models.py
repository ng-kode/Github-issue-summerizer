from django.db import models

# Create your models here.
class Issue(models.Model):
  created = models.DateTimeField(auto_now_add=True)
  title = models.CharField(max_length=10000)
  body = models.TextField(max_length=10000)

  class Meta:
    ordering = ('created', )