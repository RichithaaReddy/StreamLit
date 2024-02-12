from pathlib import Path
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'best.pt'
SOURCES_LIST = ['Video', 'YouTube', 'Other']
IMAGES_DIR = '/path/to/your/images'
