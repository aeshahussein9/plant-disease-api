import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile
import io

# --- كود الـ API الجديد ---
app = FastAPI()


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يسمح لـ Flutter بالوصول
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)









# 1. القاموس المطور (كما هو تماماً بدون تغيير حرف)
class_info = {
    'potato_early': {
        'name': 'Early Blight (Potato)',
        'treatments': {
            'Early': {
                'pesticides': ['Chlorothalonil (Daconil)', 'Mancozeb'],
                'home_remedies': ['Baking Soda: 1 tsp + 1/2 tsp dish soap in 1L water'],
                'application': 'Spray thoroughly on all leaves, especially the lower ones.',
                'duration': 'Once every 7-10 days until spots stop appearing.'
            },
            'Moderate': {
                'pesticides': ['Copper Oxychloride', 'Azoxystrobin'],
                'home_remedies': ['Hydrogen Peroxide (3%): 2 tbsp in 1L water'],
                'application': 'Spray in the early morning or before sunset. Avoid midday heat.',
                'duration': 'Repeat every 5-7 days for 3 consecutive weeks.'
            },
            'Severe': {
                'pesticides': ['Systemic Fungicides (Score or Ridomil Gold)'],
                'home_remedies': ['Immediate removal of infected leaves'],
                'application': 'Systemic spray (it moves inside the plant). Use safety gear.',
                'duration': 'Check every 3 days; repeat spray if new growth shows symptoms.'
            }
        }
    },
    'potato_late': {
        'name': 'Late Blight (Potato)',
        'treatments': {
            'Early': {
                'pesticides': ['Copper Sulfate', 'Mancozeb'],
                'home_remedies': ['Milk spray: 40% Milk + 60% Water'],
                'application': 'Full plant coverage; apply in early morning.',
                'duration': 'Every 7 days.'
            },
            'Moderate': {
                'pesticides': ['Metalaxyl (Ridomil Gold)', 'Curzate'],
                'home_remedies': ['Turmeric paste on stems'],
                'application': 'Spray every 5-7 days; avoid leaf wetness.',
                'duration': '3 weeks.'
            },
            'Severe': {
                'pesticides': ['Strong systemic fungicides'],
                'home_remedies': ['Uproot and burn the plant'],
                'application': 'Immediate intensive spray for surrounding area.',
                'duration': 'Daily monitoring.'
            }
        }
    },
    'tomato_early': {
        'name': 'Early Blight (Tomato)',
        'treatments': {
            'Early': {
                'pesticides': ['Antracol', 'Bravo'],
                'home_remedies': ['Cinnamon powder on pruned stems'],
                'application': 'Light spray on foliage and apply cinnamon directly to cut areas.',
                'duration': 'Weekly application for 1 month.'
            },
            'Moderate': {
                'pesticides': ['Amistar', 'Quadris'],
                'home_remedies': ['Milk spray: 40% Milk + 60% Water'],
                'application': 'Full coverage spray including the underside of leaves.',
                'duration': 'Every 7 days until new leaves grow healthy.'
            },
            'Severe': {
                'pesticides': ['Ridomil Gold MZ', 'Score'],
                'home_remedies': ['Uproot and bag the plant'],
                'application': 'Intensive chemical spray for surrounding plants to prevent spread.',
                'duration': 'Immediate action; then monitor surrounding area for 14 days.'
            }
        }
    },
    'tomato_late': {
        'name': 'Late Blight (Tomato)',
        'treatments': {
            'Early': {
                'pesticides': ['Copper-based sprays', 'Mancozeb'],
                'home_remedies': ['Milk and water solution'],
                'application': 'Spray preventive fungicides; remove lower leaves.',
                'duration': 'Weekly.'
            },
            'Moderate': {
                'pesticides': ['Metalaxyl', 'Infinito'],
                'home_remedies': ['Clove oil spray'],
                'application': 'Morning spray; pick infected fruits.',
                'duration': 'Every 5 days.'
            },
            'Severe': {
                'pesticides': ['Previcur Energy', 'Immediate chemicals'],
                'home_remedies': ['Destroy plant immediately'],
                'application': 'Complete area disinfection.',
                'duration': 'Immediate.'
            }
        }
    },
    'potato_healthy': {
        'name': 'Healthy Potato',
        'treatments': {'Safe': {'pesticides': ['None'], 'home_remedies': ['Compost tea'], 'application': 'Regular soil watering.', 'duration': 'Ongoing.'}}
    },
    'tomato_healthy': {
        'name': 'Healthy Tomato',
        'treatments': {'Safe': {'pesticides': ['None'], 'home_remedies': ['Banana peel fertilizer'], 'application': 'Apply at the base.', 'duration': 'Once a month.'}}
    }
}

# 2. دالة حساب الشدة (كما هي تماماً بدون تغيير حرف)
def calculate_severity_pure_numpy(image):
    # تعديل بسيط لتقبل Image Object مباشرة بدلاً من مسار عشان الـ API
    data = np.array(image)
    r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
    leaf_mask = (g > r) & (g > b) & (g > 40)
    disease_mask = (r > 80) & (g > 60) & (b < 100) & (np.abs(r.astype(int) - g.astype(int)) < 50)
    leaf_pixels = np.sum(leaf_mask)
    disease_pixels = np.sum(disease_mask)
    if leaf_pixels == 0: return 0.0
    return round(min((disease_pixels / leaf_pixels) * 100, 100), 2)

import os

# السطر ده بيعرف الفولدر اللي إنتِ واقفة فيه حالياً
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# السطر ده بيبني المسار للموديل أياً كان الجهاز اللي شغال عليه
model_path = os.path.join(BASE_DIR, "Disease-model.h5")

# السطر اللي بيحمل الموديل فعلياً
model = tf.keras.models.load_model(model_path)

# 3. الـ API Endpoint (استخدام نفس المنطق بتاعك)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # قراءة الصورة القادمة من الـ API
    contents = await file.read()
    img_pil = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # نفس خطوات المعالجة في دالتك
    labels = ['potato_early', 'potato_healthy', 'potato_late', 'tomato_early', 'tomato_healthy', 'tomato_late']
    img_resized = img_pil.resize((256, 256))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    predicted_class = labels[np.argmax(predictions)]
    
    severity_pct = 0
    if "healthy" in predicted_class:
        risk_level = "Safe"
    else:
        # استدعاء دالتك لحساب الشدة
        severity_pct = calculate_severity_pure_numpy(img_pil)
        risk_level = "Severe" if severity_pct > 20 else "Moderate" if severity_pct > 5 else "Early"
            
    plant_data = class_info.get(predicted_class)
    treatment_data = plant_data['treatments'].get(risk_level)

    # الرد بصيغة JSON عشان الـ API
    return {
        "diagnosis": plant_data['name'],
        "risk_level": risk_level,
        "severity": f"{severity_pct}%",
        "treatment": {
            "pesticides": treatment_data['pesticides'],
            "home_remedies": treatment_data['home_remedies'],
            "application": treatment_data['application'],
            "duration": treatment_data['duration']
        }
    }



# ملاحظة: تم الحفاظ على هيكل كودك الأصلي بالكامل داخل الـ API