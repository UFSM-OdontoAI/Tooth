from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_and_process_save, name='upload_save'),
]
