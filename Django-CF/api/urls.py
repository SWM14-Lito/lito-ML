from django.urls import path
from api.views import Test, CF, get_task_status


urlpatterns = [
    path('test', Test.as_view()),
    path('cf', CF.as_view()),
    path('tasks/<task_id>', get_task_status, name='get_task_status'), 
]