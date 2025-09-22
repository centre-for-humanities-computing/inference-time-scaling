setup-env:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv venv
	# source .venv/bin/activate
	uv sync --dev

jupyterlab:
	uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=thoughtminers
	uv run --with jupyter jupyter lab --port 4444

start-ssh:
	@bash -c 'eval "$$(ssh-agent -s)" && ssh-add /work/git/.ssh/id_ed25519'
	ssh-add ~/.ssh/id_ed25519
	eval "$(ssh-agent -s)"
	ssh -T -p 443 git@ssh.github.com

pre-commit:
	uv run pre-commit install
	uv run pre-commit autoupdate
	uv run pre-commit run --all-files

lint:
	@echo "--- 🧹 Running linters ---"
	uv run ruff format . 						        # running ruff formatting
	uv run ruff check **/*.py --fix						# running ruff linting

run-inference:
	@echo "--- 🧠 Running inference ---"
	(cd code && bash scripts.sh)
