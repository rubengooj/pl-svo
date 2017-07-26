echo "Building 3rdparty/line_descriptor ... "
cd 3rdparty/line_descriptor
mkdir build
cd build
cmake ..
make -j
cd ../../../

echo "Building PL-SVO ... "
mkdir build
cd build
cmake ..
make -j
