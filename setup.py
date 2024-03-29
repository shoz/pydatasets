from setuptools import setup

def readme():
    with open("README.rst") as f:
        return f.read()

setup(
    name='pydatasets',
    version='0.0.1',
    description='pydatasets',
    long_description=readme(),
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        ],
    keywords='data structure dictionary fuzzy',
    author='Shoji Ihara',
    author_email='shoji.ihara@gmail.com',
    url='https://github.com/shoz/pydatasets',
    packages=['pydatasets'],
    license='MIT'
)
