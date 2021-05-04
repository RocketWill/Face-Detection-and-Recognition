#include <algorithm>
#include <chrono>

#include "arcface.h"
#include "yaml-cpp/yaml.h"
#include "common.h"
#include <iostream>
#include <fstream>

ArcFace::ArcFace(const std::string &config_file) {
    YAML::Node root = YAML::LoadFile(config_file);
    YAML::Node config = root["arcface"];
    onnx_file = config["onnx_file"].as<std::string>();
    engine_file = config["engine_file"].as<std::string>();
    BATCH_SIZE = config["BATCH_SIZE"].as<int>();
    INPUT_CHANNEL = config["INPUT_CHANNEL"].as<int>();
    IMAGE_WIDTH = config["IMAGE_WIDTH"].as<int>();
    IMAGE_HEIGHT = config["IMAGE_HEIGHT"].as<int>();
}

ArcFace::~ArcFace() {
    // destroy the engine
    context->destroy();
    engine->destroy();
}

void ArcFace::LoadEngine() {
    // create and load engine
    std::fstream existEngine;
    existEngine.open(engine_file, std::ios::in);
    if (existEngine) {
        readTrtFile(engine_file, engine);
        assert(engine != nullptr);
    } else {
        onnxToTRTModel(onnx_file, engine_file, engine, BATCH_SIZE);
        assert(engine != nullptr);
    }
}

cv::Mat ArcFace::InferenceImage(const cv::Mat &aligned_face) {
    // std::vector<std::string> sample_images = readFolder(folder_name);
    std::vector<cv::Mat> images = {aligned_face};
    //get context
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);

    //get buffers
    assert(engine->getNbBindings() == 2);
    void *buffers[2];
    std::vector<int64_t> bufferSize;
    int nbBindings = engine->getNbBindings();
    bufferSize.resize(nbBindings);

    for (int i = 0; i < nbBindings; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
        bufferSize[i] = totalSize;
        std::cout << "binding" << i << ": " << totalSize << std::endl;
        cudaMalloc(&buffers[i], totalSize);
    }

    //get stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int outSize = bufferSize[1] / sizeof(float) / BATCH_SIZE;

    cv::Mat feature = EngineInference(images, outSize, buffers, bufferSize, stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    return feature;
}

cv::Mat ArcFace::EngineInference(const std::vector<cv::Mat> &image_list, const int &outSize, void **buffers,
                              const std::vector<int64_t> &bufferSize, cudaStream_t stream) {
    int index = 0;
    int batch_id = 0;
    std::vector<cv::Mat> vec_Mat(BATCH_SIZE);
    cv::Mat face_feature(image_list.size(), outSize, CV_32FC1);
    float total_time = 0;
    for (const cv::Mat &src_img : image_list)
    {
        index++;
        // std::cout << "Processing: " << image_name << std::endl;
        // cv::Mat src_img = cv::imread(image_name);
        if (src_img.data)
        {
            vec_Mat[batch_id] = src_img.clone();
            batch_id++;
        }
        if (batch_id == BATCH_SIZE or index == image_list.size())
        {
            auto t_start_pre = std::chrono::high_resolution_clock::now();
            std::cout << "prepareImage" << std::endl;
            std::vector<float>curInput = prepareImage(vec_Mat);
            auto t_end_pre = std::chrono::high_resolution_clock::now();
            float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
            std::cout << "prepare image take: " << total_pre << " ms." << std::endl;
            total_time += total_pre;
            batch_id = 0;
            if (!curInput.data()) {
                std::cout << "prepare images ERROR!" << std::endl;
                continue;
            }
            // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
            std::cout << "host2device" << std::endl;
            cudaMemcpyAsync(buffers[0], curInput.data(), bufferSize[0], cudaMemcpyHostToDevice, stream);

            // do inference
            std::cout << "execute" << std::endl;
            auto t_start = std::chrono::high_resolution_clock::now();
            context->execute(BATCH_SIZE, buffers);
            auto t_end = std::chrono::high_resolution_clock::now();
            float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
            std::cout << "Inference take: " << total_inf << " ms." << std::endl;
            total_time += total_inf;
            std::cout << "execute success" << std::endl;
            std::cout << "device2host" << std::endl;
            std::cout << "post process" << std::endl;
            auto r_start = std::chrono::high_resolution_clock::now();
            float out[outSize * BATCH_SIZE];
            cudaMemcpyAsync(out, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            int rowSize = index % BATCH_SIZE == 0 ? BATCH_SIZE : index % BATCH_SIZE;

            cv::Mat feature(rowSize, outSize, CV_32FC1);
            ReshapeandNormalize(out, feature, rowSize, outSize);
            // cv::Mat feature(rowSize, outSize, CV_32FC1, out);
            feature.copyTo(face_feature.rowRange(index - rowSize, index));
            auto r_end = std::chrono::high_resolution_clock::now();
            float total_res = std::chrono::duration<float, std::milli>(r_end - r_start).count();
            std::cout << "Post process take: " << total_res << " ms." << std::endl;
            total_time += total_res;
            vec_Mat = std::vector<cv::Mat>(BATCH_SIZE);
        }
    }
    // std::cout << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
    // cv::Mat similarity = face_feature * face_feature.t();
    // std::cout << "The similarity matrix of the image folder is:\n" << (similarity + 1) / 2 << "!" << std::endl;
    return face_feature;
}

std::vector<float> ArcFace::prepareImage(std::vector<cv::Mat> &vec_img) {
    std::vector<float> result(BATCH_SIZE * IMAGE_WIDTH * IMAGE_HEIGHT * INPUT_CHANNEL);
    float *data = result.data();
    for (const cv::Mat &src_img : vec_img)
    {
        if (!src_img.data)
            continue;
        cv::Mat flt_img;
        cv::resize(src_img, flt_img, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
        flt_img.convertTo(flt_img, CV_32FC3);

        //HWC TO CHW
        std::vector<cv::Mat> split_img(INPUT_CHANNEL);
        cv::split(flt_img, split_img);

        int channelLength = IMAGE_WIDTH * IMAGE_HEIGHT;
        for (int i = 0; i < INPUT_CHANNEL; ++i)
        {
            memcpy(data, split_img[i].data, channelLength * sizeof(float));
            data += channelLength;
        }
    }
    return result;
}

void ArcFace::ReshapeandNormalize(float *out, cv::Mat &feature, const int &MAT_SIZE, const int &outSize) {
    for (int i = 0; i < MAT_SIZE; i++)
    {
        cv::Mat onefeature(1, outSize, CV_32FC1, out + i * outSize);
        cv::normalize(onefeature, onefeature);
        onefeature.copyTo(feature.row(i));
    }
}

void ArcFace::PrepareFacebank(const std::map<std::string, std::vector<cv::String>> &mapping, std::string save_path) {
    //get context
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);

    //get buffers
    assert(engine->getNbBindings() == 2);
    void *buffers[2];
    std::vector<int64_t> bufferSize;
    int nbBindings = engine->getNbBindings();
    bufferSize.resize(nbBindings);

    for (int i = 0; i < nbBindings; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
        bufferSize[i] = totalSize;
        std::cout << "binding" << i << ": " << totalSize << std::endl;
        cudaMalloc(&buffers[i], totalSize);
    }

    //get stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int outSize = bufferSize[1] / sizeof(float) / BATCH_SIZE;

    WriteFacebank(mapping, outSize, buffers, bufferSize, stream, save_path);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
}

void ArcFace::WriteFacebank(const std::map<std::string, std::vector<cv::String>> &mapping, const int &outSize, void **buffers,
                         const std::vector<int64_t> &bufferSize, cudaStream_t stream, std::string save_path) {
    int index = 0;
    int batch_id = 0;
    std::vector<cv::Mat> vec_Mat(BATCH_SIZE);
    cv::Mat face_features;
    float facebank[mapping.size()*outSize];
    int idx = 0;

    for (auto &face: mapping) {
        cv::Mat one_feature;
        auto face_files = face.second;
        std::cout << "Processing: " << face.first << std::endl;
        int flag = 0;
        for (auto &face_file: face_files) {
            cv::Mat src_img = cv::imread(face_file);
            vec_Mat[batch_id] = src_img.clone();
            std::vector<float>curInput = prepareImage(vec_Mat);
            cudaMemcpyAsync(buffers[0], curInput.data(), bufferSize[0], cudaMemcpyHostToDevice, stream);
            context->execute(BATCH_SIZE, buffers);
            float out[outSize * BATCH_SIZE];
            cudaMemcpyAsync(out, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            cv::Mat feature(1, outSize, CV_32FC1);
            ReshapeandNormalize(out, feature, 1, outSize);
            if (flag == 0) {
                one_feature = feature;
            }
            else {
                one_feature += feature;
            }
            ++flag;
            vec_Mat = std::vector<cv::Mat>(BATCH_SIZE);
        }
        one_feature = one_feature / (float)face_files.size();
        one_feature = one_feature.reshape(1,1);
        for (size_t i=0; i<outSize; ++i) {
            float ele = one_feature.at<float>(0, i);
            facebank[idx++] = ele;
        }
    }

    std::cout << idx << std::endl;

    // for (size_t x=0; x<100; ++x) {
    //     std::cout << facebank[x] << std::endl;
    // }

    std::ofstream ofp(save_path, std::ios::out | std::ios::binary);
    ofp.write((char*)facebank, mapping.size()*outSize*sizeof(float));
    ofp.close();
}