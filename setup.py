from setuptools import setup

setup(
    name='learning_scenario_generators',
    version='0.1.0',
    packages=['learningscenariogenerators'],
    url='',
    license='',
    author='Patrick Westphal',
    author_email='',
    description='',
    install_requires=[
        'morelia_noctua',
        'python-lorem==1.1.2',
    ],
    scripts=[
        'bin/create_sa_scenario',
    ]
)
