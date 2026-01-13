import numpy as np
import cv2
from django.shortcuts import render
from django.core.files.base import ContentFile
from django.contrib.auth.decorators import login_required
from .forms import ImageUploadForm
from .models import Upload
from .predict_unet import main
from .constants import VALID_IMAGE_EXTENSIONS, MAX_FILE_SIZE_MB


@login_required
def home(request):
    """Página inicial com informações introduútórias"""
    return render(request, 'imagemproc/home.html')


def _render_upload_error(request, form, error_message):
    """Helper para renderizar upload com erro"""
    form.add_error('image', error_message)
    return render(request, 'imagemproc/upload.html', {'form': form})


@login_required
def upload_and_process_save(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():
            try:
                uploaded = form.cleaned_data['image']
                uploaded.seek(0)
                file_bytes = uploaded.read()

                if not file_bytes:
                    return _render_upload_error(request, form, 'Arquivo de imagem vazio ou corrompido')

                np_bytes = np.frombuffer(file_bytes, dtype=np.uint8)
                test_img = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
                
                if test_img is None:
                    return _render_upload_error(
                        request, form,
                        'Não foi possível processar a imagem. Verifique se o arquivo não está corrompido.'
                    )

                outimage = main(np_bytes)
                obj = Upload.objects.create(original=uploaded)

                ok, buffer = cv2.imencode('.png', outimage)
                if not ok:
                    return _render_upload_error(request, form, 'Erro ao processar a imagem. Tente novamente.')

                obj.result.save(f"result_{obj.id}.png", ContentFile(buffer.tobytes()), save=True)
                return render(request, 'imagemproc/result.html', {'obj': obj})

            except Exception as e:
                return _render_upload_error(request, form, f'Erro ao processar a imagem: {str(e)}')
    else:
        form = ImageUploadForm()

    context = {
        'form': form,
        'valid_extensions': VALID_IMAGE_EXTENSIONS,
        'max_file_size_mb': MAX_FILE_SIZE_MB,
    }
    return render(request, 'imagemproc/upload.html', context)
