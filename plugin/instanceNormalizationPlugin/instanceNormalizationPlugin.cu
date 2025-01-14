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

#include "instanceNormalizationPlugin.h"
#include <cuda_fp16.h>
#include <stdexcept>

using namespace nvinfer1;
using nvinfer1::plugin::InstanceNormalizationPlugin;
using nvinfer1::plugin::InstanceNormalizationPluginCreator;

inline bool is_CHW(nvinfer1::Dims const& dims)
{
    return (dims.nbDims == 3 && dims.type[0] == nvinfer1::DimensionType::kCHANNEL
        && dims.type[1] == nvinfer1::DimensionType::kSPATIAL && dims.type[2] == nvinfer1::DimensionType::kSPATIAL);
}

template <typename T, int THREADS_PER_CTA>
__global__ __launch_bounds__(THREADS_PER_CTA) void in3d_relu_activation(
    T* __restrict dst, T* __restrict src, float alpha, int count)
{
    int idx = blockIdx.x * THREADS_PER_CTA + threadIdx.x;
    if (idx >= count)
        return;

    float val = src[idx];
    dst[idx] = (val < 0.f) ? val * alpha : val;
}

// This is derived from: https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
inline float half_to_float_fast(unsigned short value)
{
    union F32 {
        unsigned int u;
        float f;
    };
    static const F32 magic = {(254 - 15) << 23};
    static const F32 was_infnan = {(127 + 16) << 23};
    F32 result;
    result.u = (value & 0x7fff) << 13; // exponent/mantissa bits
    result.f *= magic.f;               // exponent adjust
    if (result.f >= was_infnan.f)
    { // make sure Inf/NaN survive
        result.u |= 255 << 23;
    }
    result.u |= (value & 0x8000) << 16; // sign bit
    return result.f;
}

cudnnStatus_t convert_trt2cudnn_dtype(nvinfer1::DataType trt_dtype, cudnnDataType_t* cudnn_dtype)
{
    switch (trt_dtype)
    {
    case nvinfer1::DataType::kFLOAT: *cudnn_dtype = CUDNN_DATA_FLOAT; break;
    case nvinfer1::DataType::kHALF: *cudnn_dtype = CUDNN_DATA_HALF; break;
    default: return CUDNN_STATUS_BAD_PARAM;
    }
    return CUDNN_STATUS_SUCCESS;
}

namespace
{
constexpr const char* INSTANCE_PLUGIN_VERSION{"1"};
constexpr const char* INSTANCE_PLUGIN_NAME{"InstanceNormalization_TRT"};
} // namespace

PluginFieldCollection InstanceNormalizationPluginCreator::mFC{};
std::vector<PluginField> InstanceNormalizationPluginCreator::mPluginAttributes;

InstanceNormalizationPlugin::InstanceNormalizationPlugin(
    float epsilon, const std::vector<float>& scale, const std::vector<float>& bias, int relu, float alpha)
    : mEpsilon(epsilon)
    , mNchan(scale.size())
    , mHostScale(scale)
    , mHostBias(bias)
    , mRelu(relu)
    , mAlpha(alpha)
    , mInputScale(-1.f)
    , mOutputScale(-1.f)
    , mDeviceScale(nullptr)
    , mDeviceBias(nullptr)
    , mDeviceBytes(0)
{
    ASSERT(scale.size() == bias.size());
}

InstanceNormalizationPlugin::InstanceNormalizationPlugin(
    float epsilon, nvinfer1::Weights const& scale, nvinfer1::Weights const& bias, int relu, float alpha)
    : mEpsilon(epsilon)
    , mNchan(scale.count)
    , mRelu(relu)
    , mAlpha(alpha)
    , mInputScale(-1.f)
    , mOutputScale(-1.f)
    , mDeviceScale(nullptr)
    , mDeviceBias(nullptr)
    , mDeviceBytes(0)
{
    ASSERT(scale.count == bias.count);
    if (scale.type == nvinfer1::DataType::kFLOAT)
    {
        mHostScale.assign((float*) scale.values, (float*) scale.values + scale.count);
    }
    else if (scale.type == nvinfer1::DataType::kHALF)
    {
        mHostScale.reserve(mNchan);
        for (int c = 0; c < mNchan; ++c)
        {
            unsigned short value = ((unsigned short*) scale.values)[c];
            mHostScale.push_back(__internal_half2float(value));
        }
    }
    else
    {
        throw std::runtime_error("Unsupported scale dtype");
    }
    if (bias.type == nvinfer1::DataType::kFLOAT)
    {
        mHostBias.assign((float*) bias.values, (float*) bias.values + bias.count);
    }
    else if (bias.type == nvinfer1::DataType::kHALF)
    {
        mHostBias.reserve(mNchan);
        for (int c = 0; c < mNchan; ++c)
        {
            unsigned short value = ((unsigned short*) bias.values)[c];
            mHostBias.push_back(__internal_half2float(value));
        }
    }
    else
    {
        throw std::runtime_error("Unsupported bias dtype");
    }
}

InstanceNormalizationPlugin::InstanceNormalizationPlugin(void const* serialData, size_t serialLength)
{
    deserialize_value(&serialData, &serialLength, &mEpsilon);
    deserialize_value(&serialData, &serialLength, &mNchan);
    deserialize_value(&serialData, &serialLength, &mHostScale);
    deserialize_value(&serialData, &serialLength, &mHostBias);
    deserialize_value(&serialData, &serialLength, &mRelu);
    deserialize_value(&serialData, &serialLength, &mAlpha);
    deserialize_value(&serialData, &serialLength, &mInputScale);
    deserialize_value(&serialData, &serialLength, &mOutputScale);
}

InstanceNormalizationPlugin::~InstanceNormalizationPlugin()
{
    terminate();
}

// InstanceNormalizationPlugin returns one output.
int InstanceNormalizationPlugin::getNbOutputs() const
{
    return 1;
}

DimsExprs InstanceNormalizationPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    nvinfer1::DimsExprs output(inputs[0]);
    return output;
}

int InstanceNormalizationPlugin::initialize()
{
    if (!mInitialized)
    {
        CHECK_CUDNN(cudnnCreate(&mCudnnHandle));

        CHECK_CUDNN(cudnnCreateTensorDescriptor(&mBDescriptor));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&mXDescriptor));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&mYDescriptor));

        // NDHWC path
        // Device info.
        int device;
        CHECK_CUDA(cudaGetDevice(&device));
        cudaDeviceProp props;
        CHECK_CUDA(cudaGetDeviceProperties(&props, device));

        mContext.sm_count = props.multiProcessorCount;
        mContext.sm_shared_size = props.sharedMemPerMultiprocessor;
        mContext.sm_version = props.major * 100 + props.minor * 10;

        memset(&mParams, 0, sizeof(mParams));

        CHECK_CUDA(cudaMalloc(&mDeviceScale, mNchan * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&mDeviceBias, mNchan * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(mDeviceScale, &mHostScale[0], mNchan * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(mDeviceBias, &mHostBias[0], mNchan * sizeof(float), cudaMemcpyHostToDevice));
    }
    mInitialized = true;

    return 0;
}

void InstanceNormalizationPlugin::terminate()
{
    if (mInitialized)
    {
        cudnnDestroyTensorDescriptor(mYDescriptor);
        cudnnDestroyTensorDescriptor(mXDescriptor);
        cudnnDestroyTensorDescriptor(mBDescriptor);

        cudnnDestroy(mCudnnHandle);

        cudaFree(mDeviceBias);
        cudaFree(mDeviceScale);
    }
    mInitialized = false;
}

size_t InstanceNormalizationPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const
{
    nvinfer1::Dims input_dims = inputs[0].dims;

    if (input_dims.nbDims <= 4)
    {
        return 0;
    }

    if (inputs[0].format == nvinfer1::PluginFormat::kLINEAR)
    {
        nvinfer1::Dims input_dims = inputs[0].dims;

        int n = input_dims.d[0];
        int c = input_dims.d[1];

        size_t nchan_bytes = c * sizeof(float);
        size_t scale_size = n * nchan_bytes;
        size_t bias_size = n * nchan_bytes;

        size_t total_wss = scale_size + bias_size;

        return total_wss;
    }
    else if (inputs[0].format == nvinfer1::PluginFormat::kDHWC8 || inputs[0].format == nvinfer1::PluginFormat::kCDHW32)
    {
        int input_data_type = (inputs[0].type == nvinfer1::DataType::kHALF) ? 1 : 2;
        int output_data_type = (outputs[0].type == nvinfer1::DataType::kHALF) ? 1 : 2;
        nvinfer1::Dims input_dims = inputs[0].dims;

        int n = input_dims.d[0];
        int c = input_dims.d[1];
        int d = input_dims.d[2];
        int h = input_dims.d[3];
        int w = input_dims.d[4];

        InstanceNormFwdParams params;
        // only these parameters are required for workspace computation
        params.nhw = d * h * w;
        params.c = c;
        params.n = n;
        // Reserve memory for the workspaces.
        size_t size_sums, size_counts, size_retired_ctas;
        instance_norm_buffer_sizes_dispatch(
            mContext, params, size_sums, size_counts, size_retired_ctas, input_data_type, output_data_type);
        size_t size_nc = n * c * sizeof(float);
        size_nc = ((size_nc + 256 - 1) / 256) * 256;
        return size_sums + size_counts + size_retired_ctas + 4 * size_nc;
    }
    else
    {
        ASSERT(0);
    }
}

int InstanceNormalizationPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    nvinfer1::Dims input_dims = inputDesc[0].dims;

    if (input_dims.nbDims <= 4)
    {
        nvinfer1::Dims input_dims = inputDesc[0].dims;
        int n = input_dims.d[0];
        int c = input_dims.d[1];
        int h = input_dims.d[2];
        int w = input_dims.d[3] > 0 ? input_dims.d[3] : 1;
        size_t nchan_bytes = c * sizeof(float);

        // Note: We repeat the data for each batch entry so that we can do the full
        //       computation in a single CUDNN call in enqueue().
        if (mDeviceBytes < n * nchan_bytes)
        {
            cudaFree(mDeviceBias);
            cudaFree(mDeviceScale);
            mDeviceBytes = n * nchan_bytes;
            CHECK_CUDA(cudaMalloc((void**) &mDeviceScale, mDeviceBytes));
            CHECK_CUDA(cudaMalloc((void**) &mDeviceBias, mDeviceBytes));
        }
        for (int i = 0; i < n; ++i)
        {
            CHECK_CUDA(cudaMemcpy(mDeviceScale + i * c, mHostScale.data(), nchan_bytes, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(mDeviceBias + i * c, mHostBias.data(), nchan_bytes, cudaMemcpyHostToDevice));
        }

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(mBDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, n * c, 1, 1));
        cudnnDataType_t cudnn_dtype{};
        CHECK_CUDNN(convert_trt2cudnn_dtype(inputDesc[0].type, &cudnn_dtype));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(mXDescriptor, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n * c, h, w));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(mYDescriptor, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n * c, h, w));
        float alpha = 1;
        float beta = 0;
        void const* x_ptr = inputs[0];
        void* y_ptr = outputs[0];
        CHECK_CUDNN(cudnnSetStream(mCudnnHandle, stream));
        // Note: Use of CUDNN_BATCHNORM_SPATIAL_PERSISTENT can cause numerical
        //       overflows (NaNs) for fp32 data in some circumstances. The lower-
        //       performance CUDNN_BATCHNORM_SPATIAL should be used if this is not
        //       acceptable.
        CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(mCudnnHandle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &alpha,
            &beta, mXDescriptor, x_ptr, mYDescriptor, y_ptr, mBDescriptor, mDeviceScale, mDeviceBias, 1., nullptr,
            nullptr, mEpsilon, nullptr, nullptr));
    }
    else
    {
        if (inputDesc[0].format == nvinfer1::PluginFormat::kLINEAR)
        {
            CHECK_CUDNN(cudnnSetStream(mCudnnHandle, stream));
            nvinfer1::Dims input_dims = inputDesc[0].dims;
            int n = input_dims.d[0];
            int c = input_dims.d[1];
            int d = input_dims.d[2];
            int h = input_dims.d[3];
            int w = input_dims.d[4];
            size_t nchan_bytes = c * sizeof(float);

            // Note: We repeat the data for each batch entry so that we can do the full
            //       computation in a single CUDNN call in enqueue().
            float* _d_array = (float*) workspace;
            float* d_scale = &_d_array[0];
            float* d_bias = &_d_array[n * c];
            for (int i = 0; i < n; ++i)
            {
                CHECK_CUDA(
                    cudaMemcpyAsync(d_scale + i * c, mDeviceScale, nchan_bytes, cudaMemcpyDeviceToDevice, stream));
                CHECK_CUDA(cudaMemcpyAsync(d_bias + i * c, mDeviceBias, nchan_bytes, cudaMemcpyDeviceToDevice, stream));
            }

            int nc_dimA[] = {1, n * c, 1, 1, 1};
            int nc_strideA[] = {nc_dimA[1] * nc_dimA[2] * nc_dimA[3] * nc_dimA[4], nc_dimA[2] * nc_dimA[3] * nc_dimA[4],
                nc_dimA[3] * nc_dimA[4], nc_dimA[4], 1};
            int img_dimA[] = {1, n * c, d, h, w};
            int img_strideA[] = {img_dimA[1] * img_dimA[2] * img_dimA[3] * img_dimA[4],
                img_dimA[2] * img_dimA[3] * img_dimA[4], img_dimA[3] * img_dimA[4], img_dimA[4], 1};

            CHECK_CUDNN(cudnnSetTensorNdDescriptor(mBDescriptor, CUDNN_DATA_FLOAT, 5, nc_dimA, nc_strideA));
            cudnnDataType_t cudnn_dtype;
            CHECK_CUDNN(convert_trt2cudnn_dtype(inputDesc[0].type, &cudnn_dtype));
            CHECK_CUDNN(cudnnSetTensorNdDescriptor(mXDescriptor, cudnn_dtype, 5, img_dimA, img_strideA));
            CHECK_CUDNN(cudnnSetTensorNdDescriptor(mYDescriptor, cudnn_dtype, 5, img_dimA, img_strideA));
            float alpha = 1;
            float beta = 0;

            // cudaStreamSynchronize(stream);
            void const* x_ptr = inputs[0];
            void* y_ptr = outputs[0];
            // Note: Use of CUDNN_BATCHNORM_SPATIAL_PERSISTENT can cause numerical
            //       overflows (NaNs) for fp32 data in some circumstances. The lower-
            //       performance CUDNN_BATCHNORM_SPATIAL should be used if this is not
            //       acceptable.
            CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(mCudnnHandle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &alpha,
                &beta, mXDescriptor, x_ptr, mYDescriptor, y_ptr, mBDescriptor, d_scale, d_bias, 1., nullptr, nullptr,
                mEpsilon, nullptr, nullptr));

            if (mRelu > 0)
            {
                int count = n * c * d * h * w;
                const int BLOCK_SZ = 256;
                if (inputDesc[0].type == nvinfer1::DataType::kFLOAT)
                {
                    in3d_relu_activation<float, BLOCK_SZ><<<(count + BLOCK_SZ - 1) / BLOCK_SZ, BLOCK_SZ, 0, stream>>>(
                        (float*) y_ptr, (float*) y_ptr, mAlpha, count);
                }
                else if (inputDesc[0].type == nvinfer1::DataType::kHALF)
                {
                    in3d_relu_activation<__half, BLOCK_SZ><<<(count + BLOCK_SZ - 1) / BLOCK_SZ, BLOCK_SZ, 0, stream>>>(
                        (__half*) y_ptr, (__half*) y_ptr, mAlpha, count);
                }
                else
                {
                    ASSERT(0);
                }
            }
        }
        else if (inputDesc[0].format == nvinfer1::PluginFormat::kDHWC8
            || inputDesc[0].format == nvinfer1::PluginFormat::kCDHW32)
        {
            int input_data_type = (inputDesc[0].type == nvinfer1::DataType::kHALF) ? 1 : 2;
            int output_data_type = (outputDesc[0].type == nvinfer1::DataType::kHALF) ? 1 : 2;

            nvinfer1::Dims input_dims = inputDesc[0].dims;
            int n = input_dims.d[0];
            int c = input_dims.d[1];
            int d = input_dims.d[2];
            int h = input_dims.d[3];
            int w = input_dims.d[4];

            mParams.nhw = d * h * w;
            mParams.c = c;
            mParams.n = n;

            size_t size_sums, size_counts, size_retired_ctas;
            instance_norm_buffer_sizes_dispatch(
                mContext, mParams, size_sums, size_counts, size_retired_ctas, input_data_type, output_data_type);

            size_t size_nc = n * c * sizeof(float);
            size_nc = ((size_nc + 256 - 1) / 256) * 256;

            char* d_buf = reinterpret_cast<char*>(workspace);

            mParams.gmem_sums = reinterpret_cast<GMEM_SUMS_TYPE*>(d_buf);
            d_buf += size_sums;
            mParams.gmem_counts = reinterpret_cast<int*>(d_buf);
            d_buf += size_counts;
            mParams.gmem_retired_ctas = reinterpret_cast<int*>(d_buf);
            d_buf += size_retired_ctas;
            mParams.gmem_running_mean = reinterpret_cast<float*>(d_buf);
            d_buf += size_nc;
            mParams.gmem_running_var = reinterpret_cast<float*>(d_buf);
            d_buf += size_nc;
            mParams.gmem_saved_mean = reinterpret_cast<float*>(d_buf);
            d_buf += size_nc;
            mParams.gmem_saved_var = reinterpret_cast<float*>(d_buf);
            d_buf += size_nc;

            mParams.gmem_src = const_cast<void*>(inputs[0]);
            mParams.gmem_dst = outputs[0];
            mParams.gmem_bias = mDeviceBias;
            mParams.gmem_scale = mDeviceScale;

            mParams.var_eps = mEpsilon;
            mParams.exp_avg_factor = 1.f; //(float)exp_avg_factor;
            mParams.use_relu = mRelu;     // use_relu;
            mParams.relu_alpha = mAlpha;  // relu_alpha;

            mParams.in_scale = mInputScale;
            mParams.out_scale = 1.f / mOutputScale;

            int loop = instance_norm_fwd_dispatch(mContext, mParams, stream, input_data_type, output_data_type);
        }
        else
        {
            ASSERT(false && "Unexpected input format");
        }
    }
    return 0;
}

size_t InstanceNormalizationPlugin::getSerializationSize() const
{
    return (serialized_size(mEpsilon) + serialized_size(mNchan) + serialized_size(mHostScale)
        + serialized_size(mHostBias) + serialized_size(mRelu) + serialized_size(mAlpha) + serialized_size(mInputScale)
        + serialized_size(mOutputScale));
}

void InstanceNormalizationPlugin::serialize(void* buffer) const
{
    serialize_value(&buffer, mEpsilon);
    serialize_value(&buffer, mNchan);
    serialize_value(&buffer, mHostScale);
    serialize_value(&buffer, mHostBias);
    serialize_value(&buffer, mRelu);
    serialize_value(&buffer, mAlpha);
    serialize_value(&buffer, mInputScale);
    serialize_value(&buffer, mOutputScale);
}

// Needs more work
bool InstanceNormalizationPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));

    bool support_fp32_linear
        = (inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
            && inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format);

    bool support_fp16_dhwc8
        = (inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == nvinfer1::PluginFormat::kDHWC8
            && inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format);

    bool support_int8_cdhw32
        = (inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == nvinfer1::PluginFormat::kCDHW32
            && inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format);

    ASSERT(pos == 0 || pos == 1);

    return support_fp32_linear || support_fp16_dhwc8 || support_int8_cdhw32;
}

const char* InstanceNormalizationPlugin::getPluginType() const
{
    return INSTANCE_PLUGIN_NAME;
}

const char* InstanceNormalizationPlugin::getPluginVersion() const
{
    return INSTANCE_PLUGIN_VERSION;
}

void InstanceNormalizationPlugin::destroy()
{
    delete this;
}

IPluginV2DynamicExt* InstanceNormalizationPlugin::clone() const
{
    auto* plugin = new InstanceNormalizationPlugin{mEpsilon, mHostScale, mHostBias, mRelu, mAlpha};
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    plugin->initialize();
    return plugin;
}

// Set plugin namespace
void InstanceNormalizationPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* InstanceNormalizationPlugin::getPluginNamespace() const
{
    return mPluginNamespace.c_str();
}

nvinfer1::DataType InstanceNormalizationPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    ASSERT(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void InstanceNormalizationPlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void InstanceNormalizationPlugin::detachFromContext() {}

void InstanceNormalizationPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    auto input_dims = in[0].desc.dims;
    for (int i = 0; i < nbInputs; i++)
    {
        for (int j = 0; j < input_dims.nbDims; j++)
        {
            // Do not support dynamic dimensions
            ASSERT(input_dims.d[j] != -1);
        }
    }

    int n = input_dims.d[0];
    int c = input_dims.d[1];
    size_t nchan_bytes = c * sizeof(float);

    if (mDeviceBytes < n * nchan_bytes)
    {
        cudaFree(mDeviceBias);
        cudaFree(mDeviceScale);
        mDeviceBytes = n * nchan_bytes;
        cudaMalloc((void**) &mDeviceScale, mDeviceBytes);
        cudaMalloc((void**) &mDeviceBias, mDeviceBytes);
    }

    mInputScale = in[0].desc.scale;
    mOutputScale = out[0].desc.scale;
}

// InstanceNormalizationPluginCreator methods
InstanceNormalizationPluginCreator::InstanceNormalizationPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("scales", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("relu", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("alpha", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* InstanceNormalizationPluginCreator::getPluginName() const
{
    return INSTANCE_PLUGIN_NAME;
}

const char* InstanceNormalizationPluginCreator::getPluginVersion() const
{
    return INSTANCE_PLUGIN_VERSION;
}

const PluginFieldCollection* InstanceNormalizationPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2DynamicExt* InstanceNormalizationPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc)
{
    std::vector<float> scaleValues;
    std::vector<float> biasValues;
    float epsilon{};
    int relu{};
    float alpha{};
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "epsilon"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            epsilon = *(static_cast<const float*>(fields[i].data));
        }
        else if (!strcmp(attrName, "scales"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            scaleValues.reserve(size);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                scaleValues.push_back(*w);
                w++;
            }
        }
        else if (!strcmp(attrName, "bias"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            biasValues.reserve(size);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                biasValues.push_back(*w);
                w++;
            }
        }
        else if (!strcmp(attrName, "relu"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            relu = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "alpha"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            alpha = *(static_cast<const float*>(fields[i].data));
        }
    }

    Weights scaleWeights{DataType::kFLOAT, scaleValues.data(), (int64_t) scaleValues.size()};
    Weights biasWeights{DataType::kFLOAT, biasValues.data(), (int64_t) biasValues.size()};

    InstanceNormalizationPlugin* obj = new InstanceNormalizationPlugin(epsilon, scaleWeights, biasWeights, relu, alpha);
    obj->setPluginNamespace(mNamespace.c_str());
    obj->initialize();
    return obj;
}

IPluginV2DynamicExt* InstanceNormalizationPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    InstanceNormalizationPlugin* obj = new InstanceNormalizationPlugin{serialData, serialLength};
    obj->setPluginNamespace(mNamespace.c_str());
    obj->initialize();
    return obj;
}
