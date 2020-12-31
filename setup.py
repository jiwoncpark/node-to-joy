from setuptools import setup, find_packages
#print(find_packages())
#required_packages = []
#with open('requirements.txt') as f:
#    required_packages = f.read().splitlines()
#required_packages += ['corner.py @ https://github.com/jiwoncpark/corner.py/archive/master.zip']
#print(required_packages)

setup(
      name='n2j',
      version='v0.10',
      author='Ji Won Park, Rodrigo Castellon',
      author_email='jiwon.christine.park@gmail.com, rjcaste@stanford.edu',
      packages=find_packages(),
      license='LICENSE.md',
      description='Methods for inference of external convergence',
      long_description=open("README.rst").read(),
      long_description_content_type='text/markdown',
      url='https://github.com/jiwoncpark/node-to-joy',
      #install_requires=required_packages,
      #dependency_links=['http://github.com/jiwoncpark/corner.py/tarball/master#egg=corner_jiwoncpark'],
      include_package_data=True,
      test_suite='nose.collector',
      tests_require=['nose'],
      classifiers=['Development Status :: 4 - Beta',
      'License :: OSI Approved :: BSD License',
      'Intended Audience :: Developers',
      'Intended Audience :: Science/Research',
      'Operating System :: OS Independent',
      'Programming Language :: Python'],
      keywords='physics'
      )
