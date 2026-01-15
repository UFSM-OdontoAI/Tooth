# Constantes compartilhadas para validação de upload

VALID_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_IMAGES_PER_BATCH = 50
UPLOAD_CARD_EXPAND_THRESHOLD = 6

TOOTH_COLORMAP = {
    0: dict(name=11, color=(173, 216, 230)),
    1: dict(name=12, color=(0, 191, 255)),
    2: dict(name=13, color=(30, 144, 255)),
    3: dict(name=14, color=(0,   0, 255)),
    4: dict(name=15, color=(0,   0, 139)),
    5: dict(name=16, color=(72,  61, 139)),
    6: dict(name=17, color=(123, 104, 238)),
    7: dict(name=18, color=(138,  43, 226)),
    8: dict(name=21, color=(128,   0, 128)),
    9: dict(name=22, color=(218, 112, 214)),
    10: dict(name=23, color=(255,   0, 255)),
    11: dict(name=24, color=(255,  20, 147)),
    12: dict(name=25, color=(176,  48,  96)),
    13: dict(name=26, color=(220,  20,  60)),
    14: dict(name=27, color=(240, 128, 128)),
    15: dict(name=28, color=(255,  69,   0)),
    16: dict(name=31, color=(255, 165,   0)),
    17: dict(name=32, color=(244, 164,  96)),
    18: dict(name=33, color=(240, 230, 140)),
    19: dict(name=34, color=(128, 128,   0)),
    20: dict(name=35, color=(139,  69,  19)),
    21: dict(name=36, color=(255, 255,   0)),
    22: dict(name=37, color=(154, 205,  50)),
    23: dict(name=38, color=(124, 252,   0)),
    24: dict(name=41, color=(144, 238, 144)),
    25: dict(name=42, color=(143, 188, 143)),
    26: dict(name=43, color=(34, 139,  34)),
    27: dict(name=44, color=(0, 255, 127)),
    28: dict(name=45, color=(0, 255, 255)),
    29: dict(name=46, color=(0, 139, 139)),
    30: dict(name=47, color=(128, 128, 128)),
    31: dict(name=48, color=(255, 255, 255)),
}
