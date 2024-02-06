# settings.py

from pathlib import Path

ROOT = Path(__file__).parent

# Other configurations...

MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'best.pt'
# Other configurations...
# settings.py

SOURCES_LIST = ['Video', 'YouTube', 'Other']
IMAGES_DIR = '/path/to/your/images'
