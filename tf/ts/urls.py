from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('about/', views.about, name='about'),
    path('menu/', views.menu, name='menu'),
    path('lang', views.actualizeModel, name='lang'),
    path('ts', views.translate, name='ts')
]
