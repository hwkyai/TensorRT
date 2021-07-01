/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_CTCBEAMSEARCHCUSTOM_PLUGIN_H
#define TRT_CTCBEAMSEARCHCUSTOM_PLUGIN_H

#include "NvInferPlugin.h"
#include "plugin.h"
#include <cstdlib>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
class CtcBeamSearchCustom : public IPluginV2IOExt
{
public:
    CtcBeamSearchCustom(const void* data, size_t length);

    ~CtcBeamSearchCustom() override;

    // CtcBeamSearchCustom() = delete;
    CtcBeamSearchCustom() = default;

// IPluginV2
    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    void destroy() override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

// IPluginV2Ext
    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

    void detachFromContext() override;

// IPluginV2IOExt
    void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;

    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override;

    IPluginV2IOExt* clone() const override;

private:
    std::string mPluginNamespace;
};

class CtcBeamSearchCustomPluginCreator : public BaseCreator
{
public:
    CtcBeamSearchCustomPluginCreator();

    ~CtcBeamSearchCustomPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    CtcBeamSearchCustom* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    CtcBeamSearchCustom* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_CTCBEAMSEARCHCUSTOM_PLUGIN_H
