//
// Created by linghu8812 on 2021/2/8.
//

#ifndef LENET_TRT_COMMON_H
#define LENET_TRT_COMMON_H

#include <cuda_runtime_api.h>
#include <numeric>
#include <fstream>
#include <dirent.h>
#include "NvOnnxParser.h"
#include "logging.h"
#include <map>

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

void setReportableSeverity(Logger::Severity severity);
std::vector<std::string>readFolder(const std::string &image_path);
std::map<int, std::string> readImageNetLabel(const std::string &fileName);
std::map<int, std::string> readCOCOLabel(const std::string &fileName);
bool readTrtFile(const std::string &engineFile, //name of the engine file
                 nvinfer1::ICudaEngine *&engine);
void onnxToTRTModel(const std::string &modelFile, // name of the onnx model
                    const std::string &filename,  // name of saved engine
                    nvinfer1::ICudaEngine *&engine, const int &BATCH_SIZE);

#endif //LENET_TRT_COMMON_H
