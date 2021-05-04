# cd build

# # cmake \
# # -DCMAKE_PREFIX_PATH=/mnt/nfs/chengyong/Workspace/compile/protobuf \
# # -DCMAKE_PREFIX_PATH=/mnt/nfs/chengyong/Workspace/compile/pytorch/tools/build \
# # -DCMAKE_PREFIX_PATH=/mnt/nfs/chengyong/Workspace/compile/opencv-3.4.10/build  \
# # ..

# cmake \
# -DCMAKE_PREFIX_PATH="/mnt/nfs/chengyong/Workspace/compile/pytorch/tools/build:/mnt/nfs/chengyong/Workspace/compile/protobuf:/mnt/nfs/chengyong/Workspace/compile/opencv-3.4.10/build" \
# ..

# make -j1


# rm build -rf
# mkdir build
cd build

# export CMAKE_PREFIX_PATH=/mnt/nfs/chengyong/Workspace/compile/opencv4/opencv/build:${CMAKE_PREFIX_PATH}

echo 'CPATH='${CPATH}
echo 'CMAKE_PREFIX_PATH='${CMAKE_PREFIX_PATH}

cmake .. && make
cd ..