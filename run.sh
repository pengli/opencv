#!/bin/bash

export OPENCV_OPENCL_ENABLE_PROFILING=1
pushd ./build/bin
./example_dnn-age_gender
popd
