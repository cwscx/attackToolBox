from setuptools import setup

def readme():
	with open("README.rst") as f:
		return f.read()

setup(name="KNN-attackToolBox",
	  version="0.2",
	  description="Attack tool box for KNN algo",
	  classifiers=[
	  	"License :: OSI Approved :: MIT License",
	  	"Operating System :: Microsoft :: Windows :: Windows 10",
	  	"Programming Language :: Python :: 3.5",
	  	"Topic :: Multimedia :: Graphics :: Graphics Conversion"
	  ],
	  url="https://github.com/cwscx/attackToolBox",
	  author="Shengyang Shi",
	  author_email="shengyang.shi@hotmail.com",
	  license="MIT",
	  packages=["attackToolBox"],
	  install_requires=[
	  	"numpy",
	  	"tensorflow-gpu",
	  	"scikit-learn",
	  	"matplotlib"
	  ],
	  zip_safe=False)