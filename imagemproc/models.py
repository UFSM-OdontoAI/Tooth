from django.db import models
import uuid


class UploadBatch(models.Model):
    """Batch de múltiplos uploads processados juntos"""
    batch_id = models.UUIDField(
        default=uuid.uuid4, editable=False, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    total_images = models.IntegerField(default=0)
    processed_images = models.IntegerField(default=0)
    zip_file = models.FileField(
        upload_to='results/zips/', null=True, blank=True)

    def __str__(self):
        return f"Batch {self.batch_id} - {self.processed_images}/{self.total_images}"


class Upload(models.Model):
    batch = models.ForeignKey(
        UploadBatch, on_delete=models.CASCADE, related_name='uploads', null=True, blank=True)
    original = models.ImageField(upload_to='originals/')
    result = models.ImageField(upload_to='results/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Upload {self.id} - {self.original.name}"
