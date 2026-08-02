.PHONY: assets coverage demo real-demo scale-download lint receipt test typecheck verify clean

assets:
	python3 scripts/generate_readme_assets.py

demo:
	PYTHONPATH=src python3 -m marketplace_recommender.cli demo --config conf/local.yml

real-demo:
	PYTHONPATH=src python3 -m marketplace_recommender.cli real-demo --config conf/real_local.yml

scale-download:
	PYTHONPATH=src python3 -m marketplace_recommender.cli download-category \
		--category Appliances --output artifacts/scale-appliances

lint:
	ruff check src scripts tests
	ruff format --check src scripts tests

typecheck:
	mypy src

receipt:
	PYTHONPATH=src python3 -m marketplace_recommender.cli verify-receipt \
		--root artifacts/local --receipt artifacts/local/monitoring/run_receipt.json

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v

coverage:
	PYTHONPATH=src coverage run -m unittest discover -s tests -p 'test_*.py'
	coverage report

verify:
	python3 -m compileall -q src scripts tests
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) coverage
	$(MAKE) demo
	$(MAKE) demo
	$(MAKE) receipt
	PYTHONPATH=src python3 scripts/verify_local.py
	python3 scripts/generate_readme_assets.py --check
	python3 -m pip wheel . -w dist --no-deps

clean:
	python3 -c 'from pathlib import Path; import shutil; p=Path("artifacts/local"); shutil.rmtree(p) if p.exists() else None'
