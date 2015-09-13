"""
Project website: https://bitbucket.org/yymao/helpers
Copyright (c) 2015 Yao-Yuan Mao (yymao)
"""

from setuptools import setup, find_packages

setup(
    name='helpers',
    version='0.1.0',
    description='A collection of some useful, but not necessarily related, Python scripts that carry out or accelerate various tasks, most of which involve dark matter simulations.',
    url='https://bitbucket.org/yymao/helpers',
    author='Yao-Yuan Mao',
    author_email='yymao@alumni.stanford.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 2 :: Only',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    use_2to3=True,
    packages=find_packages(),
    extras_require = {
        'full':  ['numpy','scipy','fast3tree'],
    },
    entry_points={
        'console_scripts': [
            'helpers-readGadgetSnapshot=readGadgetSnapshot:main',
            'helpers-getMUSICregion=getMUSICregion:main',
            'helpers-getSDSSId=getSDSSId:main',
            'helpers-distributeMUSICBndryPart=distributeMUSICBndryPart:main',
        ],
    },
)
