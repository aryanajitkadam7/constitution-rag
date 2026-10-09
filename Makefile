export PYTHONPATH := src

install:
	pip install -r requirements.txt

eval:            ## retrieval benchmark (no API key needed) -> reports/retrieval_results.md
	python -m crag.evaluate

eval-ce:         ## + cross-encoder reranker (pip install sentence-transformers)
	python -m crag.evaluate --cross-encoder

eval-gen:        ## end-to-end answers with Gemini (export GEMINI_API_KEY=...)
	python -m crag.eval_generation

test:
	python -m pytest -q

serve:
	uvicorn crag.api:app --reload --port 8000
