from setuptools import setup, find_packages

setup(
    name = "cg_lstm",
    version = "0.0.0",
    author = "Sharad Vikram",
    author_email = "sharad.vikram@gmail.com",
    description = "",
    license = "MIT",
    keywords = "theano",
    url = "https://github.com/sharadmv/cg_lstm",
    install_requires=['theano', 'theanify'],
    packages=find_packages(include=[
        'cg_lstm'
    ]),
    classifiers=[
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',

     'License :: OSI Approved :: MIT License',

    'Programming Language :: Python :: 2.7',
],
)
