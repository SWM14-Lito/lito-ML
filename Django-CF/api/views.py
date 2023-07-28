from rest_framework import views
from rest_framework.response import Response
from api.tasks import test_task, recommend_task
from django.http import JsonResponse
from celery.result import AsyncResult

class Test(views.APIView):
    def get(self, request):
        task = test_task.delay(2, 5)
        return Response({"task_id": task.id}, status=202)

class CF(views.APIView):
    def post(self, request):
        task = recommend_task.delay(request.data)
        return Response({"taskId": task.id}, status=202)

def get_task_status(request, taskId: str):
    task_result = AsyncResult(taskId)
    result = {
        "taskId": taskId,
        "taskStatus": task_result.status,
        "data": task_result.result["data"],
    }
    return JsonResponse(result, status=200)