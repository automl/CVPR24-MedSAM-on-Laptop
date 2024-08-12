Our C++ implementation is based on the C++ implementation of [medficientsam](https://github.com/hieplpvip/medficientsam/tree/59504938bb37ab7e2832ede358051976e740efe5/cpp).
We adjusted it to work with our inference approach:
+ select DAFT model according to file name
+ seperate OpenVINO artifacts for prompt encoder and mask decoder
+ use different interpolation method for downsizing
+ we use 0-255 for boxes, original code used 0-1, we also increase the box computed from the previous slice by 3 pixels for 3D images

We also fixed some bugs that existed in the original code:
+ the interpolation method is the 6th argument of cv::resize, but it was provided as 4th
+ OpenVINO modifies the input tensor, so the cached image embeddings would be modified, we create a copy of the tensor to avoid this
+ overwrite existing segmentation value in output file, instead of adding another key-value pair with key `segs` to the numpy file, if it already exists

## Building

+ [install openvino](https://docs.openvino.ai/2024/get-started/install-openvino.html?PACKAGE=OPENVINO_BASE&VERSION=v_2024_3_0&OP_SYSTEM=LINUX&DISTRIBUTION=APT)
+ build opencv
```
wget https://github.com/opencv/opencv/archive/4.9.0.zip && unzip 4.9.0.zip && wget https://raw.githubusercontent.com/hieplpvip/medficientsam/d24c8523ae7b0c32c6befa1547a8e3aecb265b23/cpp/opencv4_cmake_options.txt && cmake \
    -DENABLE_LTO=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_LIBS="core,imgproc"\
    -DCPU_BASELINE=AVX2 \
    -DCPU_DISPATCH= \
    `cat opencv4_cmake_options.txt` \
    -S opencv-4.9.0 \
    -B opencv-4.9.0/build

cmake --build opencv-4.9.0/build --target install --parallel 8
```
+ build our code via `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`
