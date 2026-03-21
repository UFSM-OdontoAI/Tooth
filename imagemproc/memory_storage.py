from django.core.files.storage import Storage
from django.core.cache import cache
from io import BytesIO


class MemoryStorage(Storage):
    def __init__(self, timeout=600):
        self.timeout = timeout

    def _save(self, name, content):
        data = content.read()
        cache.set(name, data, timeout=self.timeout)
        return name

    def open(self, name, mode='rb'):
        data = cache.get(name)
        if data is None:
            raise FileNotFoundError(name)
        return BytesIO(data)

    def exists(self, name):
        return cache.get(name) is not None

    def url(self, name):
        return f"/media-memory/{name}"

    def delete(self, name):
        cache.delete(name)

    # 🔥 ESSENCIAL PARA MIGRATIONS
    def deconstruct(self):
        return (
            "imagemproc.memory_storage.MemoryStorage",
            [],
            {"timeout": self.timeout},
        )
