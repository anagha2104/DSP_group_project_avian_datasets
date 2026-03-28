from django.urls import path
from . import views

urlpatterns = [
    # When someone visits the base URL, load the dashboard_home view
    path('', views.dashboard_home, name='dashboard-home'),
    path('matcher/', views.reverse_matcher, name='reverse-matcher'),
    path('pca/', views.pca_analysis, name='pca-analysis'),
]