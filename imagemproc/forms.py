from django import forms
from django.core.exceptions import ValidationError
import os

class ImageUploadForm(forms.Form):
    image = forms.ImageField()

    def clean_image(self):
        image = self.cleaned_data.get('image')
        
        if image:
            # Verificar extensão do arquivo
            ext = os.path.splitext(image.name)[1].lower()
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            
            if ext not in valid_extensions:
                raise ValidationError(
                    f"Formato de arquivo inválido. Formatos aceitos: {', '.join(valid_extensions)}"
                )
            
            # Verificar tamanho do arquivo (máximo 10MB)
            if image.size > 10 * 1024 * 1024:
                raise ValidationError('O arquivo é muito grande. Tamanho máximo: 10MB')
        
        return image
