FILES INCLUDED
- app.py -> Flask backend
- templates/index.html -> website page
- static/style.css -> separated CSS file
- requirements.txt -> Python dependencies
- render.yaml -> optional Render blueprint
- export_artifacts_from_notebook.py -> helper script to export trained model assets

IMPORTANT
The app will not produce live predictions until you place these files in artifacts/:
- model.joblib
- imputer.joblib
- scaler.joblib
- selector.joblib
- label_encoder.joblib
- feature_columns.joblib
- optional: selected_features.joblib

LOCAL RUN
1. pip install -r requirements.txt
2. python app.py
3. Open http://127.0.0.1:5000

RENDER DEPLOYMENT
1. Upload this folder to GitHub.
2. Make sure the artifact files are included in artifacts/.
3. In Render, create a new Web Service from the repo.
4. Build command: pip install -r requirements.txt
5. Start command: gunicorn app:app
