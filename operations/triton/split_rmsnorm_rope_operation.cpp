/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "split_rmsnorm_rope_operation.h"
#include <cstring>
#include <algorithm>
#include <iostream>
#include <sstream>
#include "acl/acl.h"
#include "operations/aclnn/utils/utils.h"
#include "torch_api/operation_factory.h"

namespace atb_speed {

namespace {
constexpr int32_t VECTOR_CORE_NUM = 32;

bool IsPowerOfTwo(int32_t value)
{
    return value > 0 && (value & (value - 1)) == 0;
}
} // namespace

SplitRmsnormRopeOperation::SplitRmsnormRopeOperation(
    const std::string &name, SplitRmsnormRopeParam param) : name_(name), param_(param)
{}

SplitRmsnormRopeOperation::~SplitRmsnormRopeOperation()
{
    ATB_SPEED_LOG_INFO(name_ + " SplitRmsnormRopeOperation deconstructor");
}

std::string SplitRmsnormRopeOperation::GetName() const
{
    return name_;
}

uint32_t SplitRmsnormRopeOperation::GetInputNum() const
{
    return atb_speed::common::NUM5;
}

uint32_t SplitRmsnormRopeOperation::GetOutputNum() const
{
    return atb_speed::common::NUM3;
}

atb::Status SplitRmsnormRopeOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                                  atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_INFO("SplitRmsnormRopeOperation Infer shape start");
    outTensorDescs.at(0).shape.dimNum = 2;
    outTensorDescs.at(1).shape.dimNum = 2;
    outTensorDescs.at(2).shape.dimNum = 2;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1] - 2 * inTensorDescs.at(1).shape.dims[1] * param_.kvHeadNum;
    outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(1).shape.dims[1] = inTensorDescs.at(1).shape.dims[1] * param_.kvHeadNum;
    outTensorDescs.at(2).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(2).shape.dims[1] = inTensorDescs.at(1).shape.dims[1] * param_.kvHeadNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(1).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(2).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(1).format = inTensorDescs.at(0).format;
    outTensorDescs.at(2).format = inTensorDescs.at(0).format;
    ATB_SPEED_LOG_INFO("SplitRmsnormRopeOperation Infer shape end");
    
    return atb::NO_ERROR;
}

std::string SplitRmsnormRopeOperation::GenerateKernelName() const
{
    std::string baseName = "split_rmsnorm_rope_kernel";
    std::string epsStr = std::to_string(param_.eps);
    epsStr.erase(std::remove(epsStr.begin(), epsStr.end(), '.'), epsStr.end());
    epsStr.erase(std::remove(epsStr.begin(), epsStr.end(), 'e'), epsStr.end());
    epsStr.erase(std::remove(epsStr.begin(), epsStr.end(), '-'), epsStr.end());

    std::vector<std::pair<std::string, std::string>> constants;

    constants.push_back({"bias", param_.hasBias ? "1" : "0"});
    constants.push_back({"eps", epsStr});
    constants.push_back({"hd", std::to_string(param_.headDim)});
    constants.push_back({"qh", std::to_string(param_.qHiddenSize)});
    constants.push_back({"kvh", std::to_string(param_.kvHiddenSize)});

    std::string kernelName = baseName;
    for (const auto &constant : constants) {
        kernelName += "_" + constant.first + constant.second;
    }
    return kernelName;
}

atb::Status SplitRmsnormRopeOperation::Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context)
{
    ATB_SPEED_LOG_INFO("SplitRmsnormRopeOperation Setup start");
    std::string kernelName = GenerateKernelName();
    ATB_SPEED_LOG_INFO("Generated kernel name: " + kernelName);

    auto& reg = xllm::kernel::npu::KernelRegistry::get_instance();
    auto kernel_path = xllm::kernel::npu::resolve_npubin_path_by_kernel(kernelName);
    if (!reg.is_kernel_registered(kernelName)) {
        if (!reg.register_kernel(kernelName, kernel_path)) {
            ATB_SPEED_LOG_ERROR("Failed to register kernel: " + kernelName);
            return atb::ERROR_RT_FAIL;
        }
    }
    auto kernelStub = reg.get_kernel_stub(kernelName);
    
    if (kernelStub == nullptr) {
        ATB_SPEED_LOG_ERROR("Kernel stub is null for '" + kernelName + "'");
        return atb::ERROR_RT_FAIL;
    }

    // OperationFactory::execute allocates Triton workspace and sync block lock
    // just like the standalone C++ unit test path, so ATB does not need to
    // provide workspace for this operation.
    workspaceSize = 0;
    ATB_SPEED_LOG_INFO("SplitRmsnormRopeOperation Setup end, and workspaceSize: " + std::to_string(workspaceSize));
    return atb::NO_ERROR;
}

atb::Status SplitRmsnormRopeOperation::Execute(const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize, atb::Context* context)
{
    ATB_SPEED_LOG_INFO("SplitRmsnormRopeOperation Execute start");
    int32_t batchSize = static_cast<int32_t>(variantPack.inTensors.at(0).desc.shape.dims[0]);
    int32_t inputHiddenSize = static_cast<int32_t>(variantPack.inTensors.at(0).desc.shape.dims[1]);
    int32_t sinHeadDim = static_cast<int32_t>(variantPack.inTensors.at(1).desc.shape.dims[1]);
    int32_t cosHeadDim = static_cast<int32_t>(variantPack.inTensors.at(2).desc.shape.dims[1]);
    if (!IsPowerOfTwo(param_.headDim) || param_.qHiddenSize % param_.kvHiddenSize != 0 ||
        param_.kvHiddenSize % param_.headDim != 0 ||
        inputHiddenSize != param_.qHiddenSize + 2 * param_.kvHiddenSize ||
        sinHeadDim != param_.headDim || cosHeadDim != param_.headDim) {
        std::stringstream ss;
        ss << "invalid split rmsnorm rope shape, inputHiddenSize: " << inputHiddenSize
           << ", qHiddenSize: " << param_.qHiddenSize << ", kvHiddenSize: " << param_.kvHiddenSize
           << ", headDim: " << param_.headDim << ", sinHeadDim: " << sinHeadDim
           << ", cosHeadDim: " << cosHeadDim;
        ATB_SPEED_LOG_ERROR(ss.str());
        return atb::ERROR_INVALID_PARAM;
    }
    int32_t gridY = static_cast<int32_t>(param_.kvHiddenSize / param_.headDim);
    if (gridY <= 0 || VECTOR_CORE_NUM % gridY != 0) {
        std::stringstream ss;
        ss << "invalid gridY: " << gridY << ", kvHiddenSize: " << param_.kvHiddenSize
           << ", headDim: " << param_.headDim;
        ATB_SPEED_LOG_ERROR(ss.str());
        return atb::ERROR_INVALID_PARAM;
    }
    int32_t gridX = VECTOR_CORE_NUM / gridY;
    int32_t gridZ = param_.gridZ;
    std::string kernelName = GenerateKernelName();

    auto& op = xllm::kernel::npu::OperationFactory::instance().split_rmsnorm_rope(kernelName);
    auto ret = op.execute(context->GetExecuteStream(), gridX, gridY, gridZ, [&](xllm::kernel::npu::ArgsBuilder& ab) {
        ab.constructArgs(variantPack.inTensors.at(0).deviceData,
                         variantPack.inTensors.at(1).deviceData,
                         variantPack.inTensors.at(2).deviceData,
                         variantPack.outTensors.at(0).deviceData,
                         variantPack.outTensors.at(1).deviceData,
                         variantPack.outTensors.at(2).deviceData,
                         variantPack.inTensors.at(3).deviceData,
                         nullptr,
                         variantPack.inTensors.at(4).deviceData,
                         nullptr,
                         batchSize);
    });
    if (ret != 0) {
        std::stringstream ss;
        ss << "rtKernelLaunch failed, error code: " << ret << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        return atb::ERROR_RT_FAIL;
    }
    ATB_SPEED_LOG_INFO("SplitRmsnormRopeOperation Execute end");
    return atb::NO_ERROR;
}

} // namespace atb_speed

