from setuptools import setup

README = open('README.md').read()
REQUIREMENTS_BASE = open('requirements.txt').read().splitlines()

# Optional dependencies
REQUIREMENTS_AZ = open(
    'requirements_az.txt').read().splitlines()
REQUIREMENTS_PG = open(
    'requirements_pg.txt').read().splitlines()
REQUIREMENTS_ALL = (REQUIREMENTS_BASE + REQUIREMENTS_AZ + REQUIREMENTS_PG)

setup(name='corso',
      classifiers=['Intended Audience :: Developers',
                   'Programming Language :: Python :: 3 :: Only',
                   'Topic :: Games/Entertainment'],
      python_requires='>=3.9',
      version='1.1.0',
      description='Environment for the game of Corso + AI tools.',
      long_description_content_type='text/markdown',
      long_description=README,
      url='http://github.com/Ball-Man/corso',
      author='Francesco Mistri',
      author_email='franc.mistri@gmail.com',
      license='MIT',
      packages=['corso'],
      extras_require={
            'az': REQUIREMENTS_AZ,
            'pg': REQUIREMENTS_PG,
            'all': REQUIREMENTS_ALL},
      install_requires=REQUIREMENTS_BASE
      )
