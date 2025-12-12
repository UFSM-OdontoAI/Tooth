import cv2
from django.shortcuts import render
from django.core.files.base import ContentFile
from .forms import ImageUploadForm
from .utils import process_image
from .models import Upload
#from PIL import Image
from io import BytesIO
from .predict_unet import main

def upload_and_process_save(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded = form.cleaned_data['image']
            obj = Upload.objects.create(original=uploaded)

            outimage = main(uploaded)
#            pil_image = Image.open(uploaded)
#            result_img = process_image(pil_image)
            buf = BytesIO()
            ret, encoded = cv2.imencode('.png', outimage)
            buf.write(encoded.tobytes())
#            cv2.imwrite(buf,outimage)
#            result_img.save(buf, format='PNG')
            buf.seek(0)
            filecontent = ContentFile(buf.read(), name=f"result_{obj.id}.png")
            obj.result.save(filecontent.name, filecontent)
            obj.save()
            return render(request, 'imagemproc/result.html', {'obj': obj})
    else:
        form = ImageUploadForm()
    return render(request, 'imagemproc/upload.html', {'form': form})
