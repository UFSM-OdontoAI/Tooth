import numpy as np
import cv2
import zipfile
from io import BytesIO
from django.shortcuts import render, get_object_or_404
from django.core.files.base import ContentFile
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse
from .forms import ImageUploadForm
from .models import Upload, UploadBatch
from .predict_unet import main
from .constants import VALID_IMAGE_EXTENSIONS, MAX_FILE_SIZE_MB, MAX_IMAGES_PER_BATCH, UPLOAD_CARD_EXPAND_THRESHOLD


@login_required
def home(request):
    """Página inicial com informações introduútórias"""
    return render(request, 'imagemproc/home.html')


def _upload_template_context(form):
    return {
        'form': form,
        'valid_extensions': VALID_IMAGE_EXTENSIONS,
        'max_file_size_mb': MAX_FILE_SIZE_MB,
        'max_images_per_batch': MAX_IMAGES_PER_BATCH,
        'expand_threshold': UPLOAD_CARD_EXPAND_THRESHOLD,
    }


def _render_upload_error(request, form, error_message):
    """Helper para renderizar upload com erro"""
    form.add_error('images', error_message)
    return render(request, 'imagemproc/upload.html', _upload_template_context(form))


@login_required
def upload_and_process_save(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():
            try:
                images = request.FILES.getlist('images')

                if not images:
                    return _render_upload_error(request, form, 'Nenhuma imagem foi enviada')

                # Criar batch para as imagens
                batch = UploadBatch.objects.create(
                    total_images=len(images),
                    processed_images=0
                )

                request.session['batch_id'] = str(batch.batch_id)

                for uploaded in images:
                    Upload.objects.create(original=uploaded, batch=batch)

                # Redirecionar para página de processamento
                return render(request, 'imagemproc/processing.html', {
                    'batch_id': str(batch.batch_id),
                    'total_images': len(images)
                })

            except Exception as e:
                return _render_upload_error(request, form, f'Erro ao enviar imagens: {str(e)}')
    else:
        form = ImageUploadForm()

    return render(request, 'imagemproc/upload.html', _upload_template_context(form))


@login_required
def process_batch(request, batch_id):
    """Processa um batch de imagens e retorna o progresso"""
    try:
        batch = get_object_or_404(UploadBatch, batch_id=batch_id)
        uploads = batch.uploads.all()

        for upload in uploads:
            if not upload.result:
                # Processar a imagem
                upload.original.seek(0)
                file_bytes = upload.original.read()
                np_bytes = np.frombuffer(file_bytes, dtype=np.uint8)

                outimage = main(np_bytes)

                ok, buffer = cv2.imencode('.png', outimage)
                if ok:
                    upload.result.save(
                        f"result_{upload.id}.png",
                        ContentFile(buffer.tobytes()),
                        save=True
                    )
                    batch.processed_images += 1
                    batch.save()

                    # Retornar progresso após cada imagem processada
                    return JsonResponse({
                        'processed': batch.processed_images,
                        'total': batch.total_images,
                        'completed': batch.processed_images >= batch.total_images
                    })

        # Todas as imagens já foram processadas
        if batch.processed_images >= batch.total_images:
            # Gerar ZIP com todas as imagens
            create_batch_zip(batch)

        return JsonResponse({
            'processed': batch.processed_images,
            'total': batch.total_images,
            'completed': True
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def create_batch_zip(batch):
    """Cria um arquivo ZIP com todas as imagens processadas de um batch"""
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for upload in batch.uploads.all():
            if upload.result:
                upload.result.seek(0)
                zip_file.writestr(
                    f"resultado_{upload.id}.png",
                    upload.result.read()
                )

    zip_buffer.seek(0)
    batch.zip_file.save(
        f"batch_{batch.batch_id}.zip",
        ContentFile(zip_buffer.read()),
        save=True
    )


@login_required
def download_batch_zip(request, batch_id):
    """Download do ZIP com todas as imagens processadas"""
    batch = get_object_or_404(UploadBatch, batch_id=batch_id)

    if not batch.zip_file:
        create_batch_zip(batch)

    batch.zip_file.seek(0)
    response = HttpResponse(batch.zip_file.read(),
                            content_type='application/zip')
    response['Content-Disposition'] = f'attachment; filename="tooth_analysis_batch_{batch.batch_id}.zip"'
    return response


@login_required
def view_results(request, batch_id):
    """Visualizar resultados de um batch processado"""
    batch = get_object_or_404(UploadBatch, batch_id=batch_id)
    uploads = batch.uploads.filter(result__isnull=False).order_by('id')

    return render(request, 'imagemproc/result.html', {
        'batch_id': str(batch.batch_id),
        'uploads': uploads,
    })
