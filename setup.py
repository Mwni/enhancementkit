from setuptools import setup

setup(
	name='enhancementkit',
	version='0.1',
	packages=[
		'enhancementkit',
		'enhancementkit.nets',
	],
	install_requires=[
		'realesrgan',
		'facexlib'
	]
)