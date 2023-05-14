from django.db import models

# Create your models here.

class Signup(models.Model):
    username = models.CharField(max_length=100)
    email = models.EmailField()
    password = models.CharField(max_length=100)

class Signin(models.Model):
    username = models.CharField(max_length=100)
    login_time = models.DateTimeField(auto_now_add=True)