start:
	pip install -r requirements.txt && uvicorn app:app --reload --host 0.0.0.0 --port 8001