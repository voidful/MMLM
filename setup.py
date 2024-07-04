from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()
    required = [i for i in required if "@" not in i]

setup(
    name='mmlm',
    version='0.0.1',
    description='',
    url='https://github.com/voidful/MMLM',
    author='Voidful',
    author_email='voidful.stack@gmail.com',
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 4 - Beta',
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python"
    ],
    packages=find_packages(),
    install_requires=required,
    zip_safe=False,
)
