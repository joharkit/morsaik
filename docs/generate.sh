set -e

poetry run sphinx-apidoc -e -o docs/source/mod morsaik
poetry run sphinx-build -M html docs/source docs/build
