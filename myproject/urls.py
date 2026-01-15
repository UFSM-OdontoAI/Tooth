# from django.contrib import admin
# from django.urls import path, include
# from django.shortcuts import redirect
# from django.conf import settings
# from django.conf.urls.static import static
#
# urlpatterns = [
#    path('', lambda request: redirect('upload')),  # redireciona para a view “upload”
# nginx    path('imagemproc/', include('imagemproc.urls')),
#    path('', include('imagemproc.urls')),
# ]
#
# servir arquivos de media em modo debug
# if settings.DEBUG:
#    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
#    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('allauth.urls')),
    path('', include('imagemproc.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
