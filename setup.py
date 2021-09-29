
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

    url='',
    license='MIT',
    author='T. Harmel',
    author_email='tristan.harmel@gmail.com',
    description='tools to simulate and visualize Radiative transfer related parameters',

    # Dependent packages (distributions)
    install_requires=['pandas','numpy','xarray','netCDF4','lmfit',
                      'matplotlib','docopt','Py6S'],

    entry_points={
          'console_scripts': [
              #'RTxploitation = RTxploitation.visu:main'
          ]}
)
