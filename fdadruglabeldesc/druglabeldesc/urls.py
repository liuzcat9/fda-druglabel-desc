from django.urls import path

from . import views

app_name = 'druglabeldesc'
urlpatterns = [
    # index
    path('', views.index, name='index'),
    path('page2/', views.page2, name='page2'),
]