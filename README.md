# PL-SVO #

This code contains an algorithm to compute monocular visual odometry by using both point and line segment features, based on the open source version of [SVO](https://github.com/uzh-rpg/rpg_svo).

**Authors:** [Ruben Gomez-Ojeda](http://mapir.isa.uma.es/mapirwebsite/index.php/people/164-ruben-gomez), [Jesus Briales](http://mapir.isa.uma.es/mapirwebsite/index.php/people/165-jesus-briales) and [Javier Gonzalez-Jimenez](http://mapir.isa.uma.es/mapirwebsite/index.php/people/95-javier-gonzalez-jimenez)

**Related publication:** [*PL-SVO: Semi-direct monocular visual odometry by combining points and line segments*](http://mapir.isa.uma.es/mapirwebsite/index.php/people/164-ruben-gomez)

If you use PL-SVO in your research work, please cite:

    @inproceedings{  gomez2016plsvo,
      title        = {{PL-SVO: Semi-direct monocular visual odometry by combining points and line segments}},
      author       = {Gomez-Ojeda, Ruben and Briales, Jesus and Gonzalez-Jimenez, Javier},
      booktitle    = {Intelligent Robots and Systems (IROS), 2016 IEEE/RSJ International Conference on},
      pages        = {4211--4216},
      year         = {2016},
      organization = {IEEE}
    }

The provided code is published under the General Public License Version 3 (GPL v3). More information can be found in the "LICENCE" also included in the repository.

Please do not hesitate to contact the authors if you have any further questions.


## 1. Prerequisites and dependencies

We have tested PL-SVO with Ubuntu 12.04, 14.04 and 16.04, but it should be straightforward to compile along other platforms. Please notice that several internal processes, such as feature detection and matching can work in parallel, for which a powerful computer would be useful if working with the parallel configuration (change flags in the config file).

### SVO
See the SVO Wiki for more instructions: https://github.com/uzh-rpg/rpg_svo/wiki

### MRPT
In case of using the provided representation. 
```
sudo apt-get install libmrpt-dev
```

Download and install instructions can be also found at: http://www.mrpt.org/ .

### Line descriptor (in 3rdparty folder)
We have modified the [*line_descriptor*](https://github.com/opencv/opencv_contrib/tree/master/modules/line_descriptor) module from the [OpenCV/contrib](https://github.com/opencv/opencv_contrib) library (both BSD) which is included in the *3rdparty* folder.


## 2. Configuration and generation

Executing the file *build.sh* will configure and generate the *line_descriptor* module, and then will configure and generate the *PL-SVO* library for which we generate: **libplsvo.so** in the lib folder, and the application **run_pipeline** that works with our dataset format (explained in the next section).


## 3. Dataset format and usage

The **run_pipeline** basic usage is: 
```
./run_pipeline  <dataset_path>  
```

where *<dataset_path>* refers to the sequence folder relative to the environment variable *${DATASETS_DIR}* that must be previously set. That sequence folder must contain the dataset configuration file named **dataset_params.yaml** following the examples in **pl-svo/config**, where **images_subfolder** refers to the image subfolder.

