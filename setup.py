from setuptools import setup

setup(
	name='enhancerkit',
	version='0.1',
	packages=[
		'enhancerkit',
		'enhancerkit.nets',
	],
	install_requires=[
		'realesrgan',
		'facexlib'
	]
)