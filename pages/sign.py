python3.8 -m venv venv
source venv/bin/activate
# or on fish shell
source venv/bin/activate.fish

pip install -r requirements.txt

python -m streamlit run app.py
