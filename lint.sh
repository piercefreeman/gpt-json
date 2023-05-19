# /bin/bash
set -e

poetry run autoflake --in-place --remove-all-unused-imports -r .
poetry run isort .
poetry run black .

poetry run mypy gpt_json
