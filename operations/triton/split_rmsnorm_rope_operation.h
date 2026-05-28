/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_SPEED_SPLIT_RMSNORM_ROPE_OPERATION_H
#define ATB_SPEED_SPLIT_RMSNORM_ROPE_OPERATION_H
#include <string>
#include <acl/acl.h>
#include "atb/atb_infer.h"
#include "atb/operation_infra.h"
#include "kernel_registry.h"
#include "operation_base.h"
#include "atb_speed/log.h"
namespace atb_speed {

struct SplitRmsnormRopeParam {
    bool hasBias = false;
    int32_t qHiddenSize = 1024;
    int32_t kvHiddenSize = 128;
    int32_t headDim = 128;
    int32_t kvHeadNum = 1;
    float eps = 1e-6;
    
    // Grid parameters
    int32_t gridZ = 1;
};

class SplitRmsnormRopeOperation : public atb::OperationInfra {
public:
    explicit SplitRmsnormRopeOperation(const std::string &name, SplitRmsnormRopeParam param);
    ~SplitRmsnormRopeOperation() override;
    std::string GetName() const override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context) override;
    atb::Status Execute(const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
        atb::Context* context) override;

private:
    std::string GenerateKernelName() const;
    std::string name_;
    SplitRmsnormRopeParam param_;
    uint32_t blockNum_ = 0;
    int64_t per_block_workspace_size_ = 0;
};
} // namespace atb_speed
#endif
