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

#include "ctcBeamSearchDecoderCustom.h"
#include <algorithm>
#include <cstring>
#include <cudnn.h>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::CtcBeamSearchCustom;
using nvinfer1::plugin::CtcBeamSearchCustomPluginCreator;

namespace {
static const char* CTCBEAMSEARCHCUSTOM_PLUGIN_VERSION{"1"};
static const char* CTCBEAMSEARCHCUSTOM_PLUGIN_NAME{"CTCBeamSearchDecoder"};
}

PluginFieldCollection CtcBeamSearchCustomPluginCreator::mFC = {};

CtcBeamSearchCustom::~CtcBeamSearchCustom() {}

int CtcBeamSearchCustom::getNbOutputs() const
{
    return 1;
}

DimsExprs CtcBeamSearchCustom::getOutputDimensions (int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder)
{
    ASSERT(outputIndex < nbInputs);
    return inputs[outputIndex];
}

int CtcBeamSearchCustom::initialize()
{
    return STATUS_SUCCESS;
}

void CtcBeamSearchCustom::terminate() {}

size_t CtcBeamSearchCustom::getWorkspaceSize (const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const
{
    return 0;
}

int CtcBeamSearchCustom::enqueue (const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream)
{
    auto* output = static_cast<float*>(outputs[0]);
    auto* input = static_cast<const float*>(inputs[0]);

    *output = *input;

    return STATUS_SUCCESS;
}

size_t CtcBeamSearchCustom::getSerializationSize() const
{
    return 0;
}

void CtcBeamSearchCustom::serialize(void* buffer) const
{
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void CtcBeamSearchCustom::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void CtcBeamSearchCustom::detachFromContext() {}

// Set plugin namespace
void CtcBeamSearchCustom::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* CtcBeamSearchCustom::getPluginNamespace() const
{
    return mPluginNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index
DataType CtcBeamSearchCustom::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    ASSERT(index < nbInputs);
    return inputTypes[index];
}

void CtcBeamSearchCustom::configurePlugin (const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs)
{
}

bool CtcBeamSearchCustom::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    return true;
}
const char* CtcBeamSearchCustom::getPluginType() const
{
    std::cout << "getPluginType()" << std::endl;
    return "CTCBeamSearchDecoder";
}

const char* CtcBeamSearchCustom::getPluginVersion() const
{
    std::cout << "getPluginVersion()" << std::endl;
    return "1";
}

void CtcBeamSearchCustom::destroy()
{
    delete this;
}

IPluginV2DynamicExt* CtcBeamSearchCustom::clone() const
{
    auto* plugin = new CtcBeamSearchCustom();
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    return plugin;
}

CtcBeamSearchCustomPluginCreator::CtcBeamSearchCustomPluginCreator()
{
    mFC.nbFields = 0;
    mFC.fields = nullptr;
}

const char* CtcBeamSearchCustomPluginCreator::getPluginName() const
{
    std::cout << "PluginCreator::getPluginName()" << std::endl;
    return CTCBEAMSEARCHCUSTOM_PLUGIN_NAME;
}

const char* CtcBeamSearchCustomPluginCreator::getPluginVersion() const
{
    std::cout << "PluginCreator::getPluginVersion()" << std::endl;
    return CTCBEAMSEARCHCUSTOM_PLUGIN_VERSION;
}

const PluginFieldCollection* CtcBeamSearchCustomPluginCreator::getFieldNames()
{
    return &mFC;
}

CtcBeamSearchCustom* CtcBeamSearchCustomPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    auto* plugin = new CtcBeamSearchCustom();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

CtcBeamSearchCustom* CtcBeamSearchCustomPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call Concat::destroy()
    // IPluginV2Ext* plugin = new CtcBeamSearchCustom();
    auto* plugin = new CtcBeamSearchCustom();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
