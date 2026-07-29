.PHONY: assets demo real-demo lint receipt test verify clean

assets:
	python3 scripts/generate_readme_assets.py

demo:
	PYTHONPATH=src python3 -m marketplace_recommender.cli demo --config conf/local.yml

real-demo:
	PYTHONPATH=src python3 -m marketplace_recommender.cli real-demo --config conf/real_local.yml

lint:
	ruff check src scripts tests
	ruff format --check src scripts tests

receipt:
	PYTHONPATH=src python3 -m marketplace_recommender.cli verify-receipt \
		--root artifacts/local --receipt artifacts/local/monitoring/run_receipt.json

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v

verify:
	python3 -m compileall -q src scripts tests
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) demo
	$(MAKE) demo
	$(MAKE) receipt
	PYTHONPATH=src python3 scripts/verify_local.py
	$(MAKE) assets
	git diff --exit-code -- assets
	python3 -m pip wheel . -w dist --no-deps

clean:
	python3 -c 'from pathlib import Path; import shutil; p=Path("artifacts/local"); shutil.rmtree(p) if p.exists() else None'
