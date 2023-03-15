from setuptools import setup

setup(
	name='enhancekit',
	version='0.1',
	packages=[
		'enhancekit',
		'enhancekit.nets',
	],
	install_requires=[
		'realesrgan',
		'facexlib'
	]
)