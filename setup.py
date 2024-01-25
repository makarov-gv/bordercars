from setuptools import setup

setup(
    name="bordercars",
    description="Border cars detection on aerial images or orthophotoplanes in TIFF image format",
    version="1.0.0",
    license="Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)",
    package_data={'': ['models/*']},
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "opencv-python",
        "tifffile",
        "openmim"
    ],
    author="Georgy Makarov",
    author_email="makarov@paradoxical.ru",
    python_requires=">=3.8",
    url="https://gitverse.ru/makarov/bordercars"
)
