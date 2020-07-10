from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('druglabeldesc/', include('druglabeldesc.urls')),
    path('admin/', admin.site.urls),
]
