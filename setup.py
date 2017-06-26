from setuptools import setup

setup(name="KNN-attackToolBox",
	  version="0.2",
	  description="Attack tool box for KNN algo",
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