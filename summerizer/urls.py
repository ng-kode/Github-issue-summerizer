from django.conf.urls import url
from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from summerizer import views

urlpatterns = [
    # template views
    path('', views.index),

    # api
    path('api/generate-title/', views.generate_title)
]

urlpatterns = format_suffix_patterns(urlpatterns)