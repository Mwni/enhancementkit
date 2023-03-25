from setuptools import setup

setup(
	name='enhancementkit',
	version='0.2',
	packages=[
		'enhancementkit',
		'enhancementkit.nets',
	],
	install_requires=[
		'realesrgan',
		'facexlib'
	]
)