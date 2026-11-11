import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import gaussian_kde
import pdfplumber
from PIL import Image

target_size = (1920, 1080)

# Check if ts_slide_2..9 exist, if not let's run generate_all_type_shift_slides
conn = sqlite3.connect('data/data.db')
df = pd.read_sql('SELECT * FROM sales_data', conn)
conn.close()

orig_pdf = '/Users/macbookm1/.gemini/antigravity/brain/fc7215d1-a3da-43f6-a923-1fa94a1e00fb/.user_uploaded/media_1792565053766_91991799.pdf'
with pdfplumber.open(orig_pdf) as pdf:
    im_s1 = pdf.pages[0].to_image(resolution=150).original.resize(target_size, Image.Resampling.LANCZOS)

print("Slide 1 loaded.")
