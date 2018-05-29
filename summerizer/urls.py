from django.conf.urls import url
from rest_framework.urlpatterns import format_suffix_patterns
from summerizer import views

urlpatterns = [
    url(r'^issues/$', views.issue_list),
    url(r'^generate-title/$', views.generate_title)
]

urlpatterns = format_suffix_patterns(urlpatterns)