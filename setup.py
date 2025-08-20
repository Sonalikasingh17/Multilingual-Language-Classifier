from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='multilingual-language-classifier',
    version='0.0.1',
    author='Sonalika Singh',
    author_email='singhsonalika5@gmail.com',
    description='An end-to-end multilingual language classification system using the MASSIVE dataset',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Sonalikasingh17/multilingual-language-classifier',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    keywords='machine learning, nlp, language classification, multilingual, text classification',
    project_urls={
        'Bug Reports': 'https://github.com/Sonalikasingh17/multilingual-language-classifier/issues',
        'Source': 'https://github.com/Sonalikasingh17/multilingual-language-classifier',
    },
)
