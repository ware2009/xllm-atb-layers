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

namespace atb_speed {

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
    param_.headDim = inTensorDescs.at(1).shape.dims[1];
    param_.qHiddenSize = outTensorDescs.at(0).shape.dims[1];
    param_.kvHiddenSize = param_.headDim * param_.kvHeadNum;
    
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
    int64_t vectorCoreNum;
    int64_t gridX = variantPack.inTensors.at(0).desc.shape.dims[0];
    int64_t gridY = variantPack.inTensors.at(0).desc.shape.dims[1] / param_.headDim;
    int32_t deviceId;
    int ret = aclrtGetDevice(&deviceId);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR("aclrtGetDevice failed, error code: " + std::to_string(ret)); 
        return atb::ERROR_RT_FAIL;
    }
    ret = aclrtGetDeviceInfo(deviceId, aclrtDevAttr::ACL_DEV_ATTR_VECTOR_CORE_NUM, &vectorCoreNum);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR("aclrtGetDeviceInfo failed, error code: " + std::to_string(ret));
        return atb::ERROR_RT_FAIL;
    }
    args_.gridX = gridX;
    args_.gridY = gridY;
    args_.gridZ = param_.gridZ;
    blockNum_ = std::min(gridX * gridY * param_.gridZ, vectorCoreNum);

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
    kernel_stub_ = reg.get_kernel_stub(kernelName);
    
    if (kernel_stub_ == nullptr) {
        ATB_SPEED_LOG_ERROR("Kernel stub is null for '" + kernelName + "'");
        return atb::ERROR_RT_FAIL;
    }

    int64_t per_block_ws = -1;
    int64_t lock_init_value = 0;
    int64_t lock_num = -1;
    reg.get_kernel_workspace_config(kernelName, per_block_ws, lock_init_value, lock_num);
    per_block_workspace_size_ = (per_block_ws > 0) ? per_block_ws : 0;
    workspaceSize = per_block_workspace_size_ * blockNum_;
    ATB_SPEED_LOG_INFO("SplitRmsnormRopeOperation Setup end, and workspaceSize: " + std::to_string(workspaceSize));
    std::stringstream ss;
    ss << "SplitRmsnormRopeOperation gridX " << gridX << " , gridY " << gridY
       << " , gridZ " << param_.gridZ << ", vectorCoreNum " << vectorCoreNum << std::endl;
    ATB_SPEED_LOG_INFO(ss.str());
    return atb::NO_ERROR;
}

atb::Status SplitRmsnormRopeOperation::Execute(const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize, atb::Context* context)
{
    ATB_SPEED_LOG_INFO("SplitRmsnormRopeOperation Execute start");
    void *fftsAddr = nullptr;
    uint32_t fftsLen;
    int ret = rtGetC2cCtrlAddr((uint64_t*)&fftsAddr, &fftsLen);
    if (ret != 0) {
        std::stringstream ss;
        ss << "rtGetC2cCtrlAddr failed, error code: " << ret << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        return atb::ERROR_RT_FAIL;
    }
    args_.fftsAddr = fftsAddr;
    args_.workspaceAddr = workspace;
    args_.input = variantPack.inTensors.at(0).deviceData;
    args_.sin = variantPack.inTensors.at(1).deviceData;
    args_.cos = variantPack.inTensors.at(2).deviceData;
    args_.qOutput = variantPack.outTensors.at(0).deviceData;
    args_.kOutput = variantPack.outTensors.at(1).deviceData;
    args_.vOutput = variantPack.outTensors.at(2).deviceData;
    args_.qWeight = variantPack.inTensors.at(3).deviceData;
    args_.kWeight = variantPack.inTensors.at(4).deviceData;
    args_.batchSize = static_cast<int32_t>(variantPack.inTensors.at(0).desc.shape.dims[0]);

    ret = rtKernelLaunch(kernel_stub_, blockNum_, static_cast<void *>(&args_), sizeof(args_), NULL, context->GetExecuteStream());
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
