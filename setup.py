from setuptools import setup, find_packages

setup(
  name = 'mixture-of-experts',
  packages = find_packages(),
  version = '0.2.3',
  license='MIT',
  description = 'Sparsely-Gated Mixture of Experts for Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/mixture-of-experts',
  keywords = ['artificial intelligence', 'deep learning', 'transformers', 'mixture of experts'],
  install_requires=[
      'torch'
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)
