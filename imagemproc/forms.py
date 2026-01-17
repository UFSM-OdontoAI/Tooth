from django import forms
from django.core.exceptions import ValidationError
import os
from .constants import VALID_IMAGE_EXTENSIONS, MAX_IMAGES_PER_BATCH, MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB


class ImageUploadForm(forms.Form):

    def clean_images(self):
        images = self.files.getlist('images')

        if not images:
            raise ValidationError('Nenhuma imagem foi enviada')

        if len(images) > MAX_IMAGES_PER_BATCH:
            raise ValidationError(
                f'Máximo de {MAX_IMAGES_PER_BATCH} imagens por vez')

        for image in images:
            ext = os.path.splitext(image.name)[1].lower()

            if ext not in VALID_IMAGE_EXTENSIONS:
                raise ValidationError(
                    f"Formato inválido em '{image.name}'. Formatos aceitos: {', '.join(VALID_IMAGE_EXTENSIONS)}"
                )

            if image.size > MAX_FILE_SIZE_BYTES:
                raise ValidationError(
                    f"'{image.name}' é muito grande. Tamanho máximo: {MAX_FILE_SIZE_MB}MB")

        return images
