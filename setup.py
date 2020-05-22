import os
from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()
with open('requirements.txt', 'r') as f:
    INSTALL_REQUIRES = f.read().strip().split('\n')

setup(
    name='pyblaze',
    version=os.getenv('CIRCLE_TAG'),

    author='Oliver Borchert',
    author_email='borchero@icloud.com',

    description='Large-Scale Machine and Deep Learning in PyTorch.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',

    url='https://github.com/borchero/pyblaze',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries'
    ],
    python_requires='>=3.7',
    install_requires=INSTALL_REQUIRES,

    license='License :: OSI Approved :: MIT License',
    zip_safe=False
)
