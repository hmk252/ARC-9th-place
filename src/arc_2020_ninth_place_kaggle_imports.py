import subprocess


def initialize_ninth_place():
    subprocess.run("pip install . --no-color --verbose --no-deps --disable-pip-version-check".split(), capture_output=False)
