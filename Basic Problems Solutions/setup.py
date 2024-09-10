from setuptools import setup, find_packages

setup(
    name="my_ml_project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "datasets",
        "peft",
        "fastapi",
        "uvicorn"
    ],
    entry_points={
        "console_scripts": [
            "run_generate=generate_text:main",
            "run_finetune=fine_tune_models:main"
            #"run_usemodel=use_fine_tuned_model:main"
        ]
    },
    python_requires='>=3.6',
)
