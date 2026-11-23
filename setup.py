
from setuptools import setup, find_packages
exec(open('RTxploitation/version.py').read())

setup(
    name=__package__,
    version=__version__,
    packages=find_packages(exclude=['build']),
    package_data={'':['*.so','*h','*angles*','*.txt','*.csv','*.dat']},
    #     # If any package contains *.txt files, include them:
    #     '': ['*.txt'],
    #     'lut': ['data/lut/*.nc'],
    #     'aux': ['data/aux/*']
    # },
    include_package_data=True,

    url='https://github.com/Tristanovsk/RTxploitation',
    project_urls={'Documentation': 'https://rtxploitation.readthedocs.io'},
    license='MIT',
    author='T. Harmel',
    author_email='tristan.harmel@gmail.com',
    description='tools to simulate and visualize Radiative transfer related parameters',

    # Dependent packages (distributions)
    install_requires=['pandas','numpy','scipy','xarray','netCDF4','lmfit',
                      'matplotlib','cmocean','docopt','Py6S'],
    extras_require={
        # build the documentation (see docs/ and .readthedocs.yaml)
        'docs': ['sphinx>=7', 'sphinx-book-theme>=1.1', 'sphinx-copybutton',
                 'myst-nb>=1.0', 'linkify-it-py', 'ipython'],
    },

    entry_points={
          'console_scripts': [
              #'RTxploitation = RTxploitation.visu:main'
          ]}
)
