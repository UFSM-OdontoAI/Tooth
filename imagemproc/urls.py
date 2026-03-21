from django.urls import path
from . import views

app_name = 'imagemproc'

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_and_process_save, name='upload_save'),
    path('media-memory/<path:path>', views.media_memory, name='media_memory'),
    path('process/<uuid:batch_id>/', views.process_batch, name='process_batch'),
    path('result/<uuid:batch_id>/', views.view_results, name='view_results'),
    path('download/<uuid:batch_id>/',
         views.download_batch_zip, name='download_batch'),
]
