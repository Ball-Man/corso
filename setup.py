from setuptools import setup

README = open('README.md').read()
REQUIREMENTS = open('requirements.txt').read().splitlines()

setup(name='corso',
      classifiers=['Intended Audience :: Developers',
                   'Programming Language :: Python :: 3 :: Only',
                   'Topic :: Games/Entertainment'],
      python_requires='>=3.9',
      version='1.0.0',
      description='Environment for the game of Corso + AI tools.',
      long_description_content_type='text/markdown',
      long_description=README,
      url='http://github.com/Ball-Man/corso',
      author='Francesco Mistri',
      author_email='franc.mistri@gmail.com',
      # license='MIT',
      packages=['corso'],
      install_requires=REQUIREMENTS
      )
