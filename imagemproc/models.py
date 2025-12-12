from django.db import models

class Upload(models.Model):
    original = models.ImageField(upload_to='originals/')
    result = models.ImageField(upload_to='results/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
