import os

os.system('kaggle competitions download -p {} -c {} -f {}'.format(
    os.getenv('KAGGLE_PATH'),
    'diabetic-retinopathy-detection',
    'test.zip.007'
))