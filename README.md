# Retention Ledger — Bank Customer Churn Prediction (Random Forest)

مشروع Flask جاهز للنشر (Deployment) بيستخدم موديل **Random Forest**
(`RandomForestClassifier`, 140 شجرة، `max_depth=19`) متدرب على بيانات
[Churn Modelling](https://www.kaggle.com/datasets/filippoo/deep-learning-az-ann)
عشان يتنبأ باحتمالية ترك العميل للبنك (Churn).

هنسخة من نفس الواجهة والتصميم اللي عملناه قبل كده للموديل الـ ANN، لكن
بالباك-إند اتظبط عشان يشتغل بالموديل الجديد اللي رفعته:

| الملف | الوظيفة |
|---|---|
| `model_random.pkl` | الموديل المدرب (`RandomForestClassifier`) |
| `Scaler_ANN.pkl` | `StandardScaler` متدرب على عمود `EstimatedSalary` فقط (نفس الملف اللي استخدمناه قبل كده) |
| `Encoders_ANN.pkl` | `LabelEncoder` لعمودي `Geography` و `Gender` |
| `Columns_ANN.pkl` | ترتيب الأعمدة اللي الموديل مدرب عليه بالظبط |

> **تأكيد فحص فعلي:** فتحت الملف وشفت إن `model.feature_names_in_` مطابق
> تمامًا لترتيب `Columns_ANN.pkl`، وإن ملفات الـ Scaler والـ Encoders هي
> نفسها اللي استخدمناها مع موديل الـ ANN قبل كده، فالـ preprocessing في
> `app.py` هو نفسه بالظبط. الفرق الوحيد إن التنبؤ دلوقتي بيتم بـ
> `model.predict_proba()` بدل `model.predict()` بتاع Keras.

### ليه النسخة دي أخف؟
الموديل ده `scikit-learn` عادي (مش شبكة عصبية)، فمحتاجين `scikit-learn`
بس من غير `tensorflow` خالص. ده بيخلي:
- حجم البيئة (dependencies) أصغر بكتير
- وقت الـ build على منصات النشر أسرع
- استهلاك الذاكرة وقت التشغيل أقل

اختبرت المشروع فعليًا في بيئة التطوير وشغّلت السيرفر وبعتّت طلب حقيقي على
`/predict` واترجعله رد صحيح، فمتأكد إن الكود شغال مش بس متوقع نظريًا.

---

## 1. التشغيل محليًا

```bash
# 1) إنشاء بيئة افتراضية
python -m venv venv
source venv/bin/activate        # على ويندوز: venv\Scripts\activate

# 2) تثبيت المكتبات
pip install -r requirements.txt

# 3) (اختياري) تأكيد إن الموديل بيتحمل صح
python test_model.py

# 4) تشغيل السيرفر
python app.py
```

بعدها افتح المتصفح على `http://127.0.0.1:5000`.

---

## 2. رفع المشروع على GitHub

```bash
cd retention-ledger-rf      # أو اسم المجلد اللي فكيت فيه الملفات
git init
git add .
git commit -m "Initial commit: Flask churn prediction app (Random Forest)"

# اعمل ريبو جديد فاضي على GitHub الأول (من غير README)، وبعدين:
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

الملف `.gitignore` مجهز عشان يستبعد `venv/` والملفات المؤقتة. حجم مجلد
`model/` بالكامل حوالي 23 ميجابايت (أغلبه ملف الموديل نفسه)، وده تحت حد
GitHub الافتراضي (100 ميجا للملف) فمش هتحتاج Git LFS.

---

## 3. الـ Deployment

المشروع مجهز بـ `Procfile` و `runtime.txt` عشان يشتغل على أي منصة بتدعم
Python/Gunicorn.

### Render.com (الأسهل)
1. اعمل حساب على [render.com](https://render.com) واربطه بحساب GitHub بتاعك.
2. **New → Web Service** واختار الريبو اللي رفعته.
3. الإعدادات:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT` (موجود جاهز في `Procfile`)
   - **Instance Type:** Free
4. دوس **Create Web Service** واستنى الـ build يخلص (هيكون أسرع بكتير من
   نسخة الـ ANN لأننا مش محتاجين TensorFlow).
5. هتاخد رابط زي `https://retention-ledger-rf.onrender.com`.

### بدائل تانية
- **Railway.app** — نفس فكرة Render، بيقرأ `Procfile` تلقائيًا.
- **Hugging Face Spaces** (Docker SDK) — استضافة مجانية دايمة للمشاريع الصغيرة.
- أي VPS عادي (مثل EC2 / DigitalOcean) — شغّل بـ:
  `gunicorn app:app --bind 0.0.0.0:8000 --workers 2` خلف Nginx.

---

## 4. هيكل المشروع

```
.
├── app.py                  # سيرفر Flask + منطق التحميل والتنبؤ
├── test_model.py           # اختبار سريع بدون تشغيل السيرفر
├── requirements.txt
├── Procfile                 # أمر التشغيل في بيئة الإنتاج (gunicorn)
├── runtime.txt               # إصدار Python
├── .gitignore
├── model/
│   ├── model_random.pkl
│   ├── Scaler_ANN.pkl
│   ├── Encoders_ANN.pkl
│   └── Columns_ANN.pkl
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/script.js
```

---

## 5. الـ API

### `POST /predict`
Body (JSON):
```json
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Female",
  "Age": 40,
  "Tenure": 3,
  "Balance": 60000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 50000
}
```

Response:
```json
{
  "ok": true,
  "churn_probability": 24.29,
  "stay_probability": 75.71,
  "will_churn": false,
  "risk_level": "Low"
}
```

### `GET /health`
بيرجع `{"status": "ok"}` — مفيد لمنصات الـ deployment عشان تتأكد إن
السيرفر شغال.

---

## 6. ملاحظات

- الواجهة مصممة بالإنجليزي لأن أسماء الحقول (CreditScore, Geography...)
  هي نفسها اللي الموديل اتدرب عليها. لو عايز نسخة عربية بالكامل قولي وهعملها.
- لو عندك نسخة الـ ANN (Keras) كمان وعايز الاتنين في نفس المشروع مع إمكانية
  الاختيار بينهم، قولي وأظبطلك endpoint إضافي زي `/predict?model=rf` أو
  `/predict?model=ann`.
