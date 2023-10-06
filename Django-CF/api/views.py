from rest_framework import views
from rest_framework.response import Response
from api.tasks import recommend_task
from django.http import JsonResponse
from celery.result import AsyncResult

class CF(views.APIView):
    def post(self, request):
        task = recommend_task.delay(request.data)
        return Response({"taskId": task.id}, status=202)

def get_task_status(request, taskId: str):
    task_result = AsyncResult(taskId)

    if task_result.result is None:
        result = {
            "taskId": taskId,
            "taskStatus": task_result.status,
            "data": [],
        }
    else:
        result = {
            "taskId": taskId,
            "taskStatus": task_result.status,
            "data": task_result.result["data"],
        }
    return JsonResponse(result, status=200)

def get_connection_status(request):
    return JsonResponse({"health": "OK"}, status=200)