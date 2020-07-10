from django.shortcuts import render
from functions import give_arr


def index(request):
    arr = give_arr()
    return render(request, 'druglabeldesc/index.html', {'arr': arr})

def page2(request):
    return render(request, 'druglabeldesc/page2.html')