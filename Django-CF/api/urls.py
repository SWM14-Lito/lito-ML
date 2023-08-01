from django.urls import path
from api.views import CF, get_task_status


urlpatterns = [
    path('v1/problems/problem-user', CF.as_view()),
    path('v1/problems/recommend-user/<taskId>', get_task_status, name='get_task_status'), 
]