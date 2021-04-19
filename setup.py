import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()
    print(required)

setuptools.setup(
    name='Covid inference',
    version='0.1',
    packages=setuptools.find_packages(),
    install_requires=required,
    url='https://github.com/edupooch/covid-inference',
)