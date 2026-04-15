/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "flash_attention_score_operation.h"

#include <securec.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>

#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention_score.h"
#include "atb/types.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"

namespace atb_speed {
namespace common {

FlashAttnScoreOperation::FlashAttnScoreOperation(const std::string& name,
                                                 AclnnFAScoreParam param)
    : AclNNOperation(name), param_(param) {}

FlashAttnScoreOperation::~FlashAttnScoreOperation() {}

atb::Status FlashAttnScoreOperation::InferShape(
    const atb::SVector<atb::TensorDesc>& inTensorDescs,
    atb::SVector<atb::TensorDesc>& outTensorDescs) const {
  ATB_SPEED_LOG_DEBUG(opName_ << " infer shape start");
  // FA: [B,S,H], PA: [B,S,N,D]
  outTensorDescs.at(0) = inTensorDescs.at(0);
  // softmaxMaxOut: [B,N,S,8] where N is HeadNum
  const int dimNum = 4;
  std::vector<int> dimVec(dimNum);
  dimVec[0] = inTensorDescs.at(0).shape.dims[0];
  dimVec[1] = param_.headNum;
  dimVec[2] = inTensorDescs.at(0).shape.dims[1];
  dimVec[3] = 8;
  for (size_t i = 1; i < outTensorDescs.size(); ++i) {
    outTensorDescs.at(i) = atb::TensorDesc{};
    outTensorDescs.at(i).format = ACL_FORMAT_ND;
    outTensorDescs.at(i).shape.dimNum = dimNum;
    outTensorDescs.at(i).dtype = ACL_FLOAT;
    for (int j = 0; j < dimNum; j++) {
      outTensorDescs.at(i).shape.dims[j] = dimVec[j];
    }
  }
  ATB_SPEED_LOG_DEBUG(opName_ << " infer shape end");
  return 0;
}

uint32_t FlashAttnScoreOperation::GetInputNum() const {
  uint32_t inputNum = NUM3;
  if (param_.hasPse) {
    inputNum++;
  }
  if (param_.hasAttnMask) {
    inputNum++;
  }
  // prefixOption not yet supported
  return inputNum;
}

uint32_t FlashAttnScoreOperation::GetOutputNum() const { return NUM4; }

atb::Status FlashAttnScoreOperation::CreateAclNNInTensorVariantPack(
    const atb::VariantPack& variantPack) {
  ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNInTensorVariantPack start "
                              << GetInputNum());
  AclNNVariantPack& aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
  aclnnVariantPack.aclInTensors.resize(GetInputNum());
  uint32_t inTensorIdx = 0;
  for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
    std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
    aclnnTensor->atbTensor = variantPack.inTensors.at(i);
    aclnnTensor->tensorIdx = inTensorIdx++;
    aclnnTensor->tensorListidx = AclNNTensor::notInTensorList;
    aclnnTensor->needUpdateTensorDataPtr = true;
    atb::Tensor atbTensor = variantPack.inTensors.at(i);
    aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
    aclnnTensor->tensor = aclCreateTensor(atbTensor.desc.shape.dims,
                                          atbTensor.desc.shape.dimNum,
                                          atbTensor.desc.dtype,
                                          aclnnTensor->strides.data(),
                                          0,
                                          atbTensor.desc.format,
                                          atbTensor.desc.shape.dims,
                                          atbTensor.desc.shape.dimNum,
                                          atbTensor.deviceData);
    if (aclnnTensor->tensor == nullptr) {
      ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor index " << i
                                        << " create fail");
      return atb::ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack.aclInTensors[i] = aclnnTensor;
  }
  ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNInTensorVariantPack end");
  return atb::NO_ERROR;
}

atb::Status FlashAttnScoreOperation::CreateAclNNOutTensorVariantPack(
    const atb::VariantPack& variantPack) {
  ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNOutTensorVariantPack start");
  AclNNVariantPack& aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
  aclnnVariantPack.aclOutTensors.resize(NUM4);
  for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
    std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
    aclnnTensor->tensorIdx = i;
    aclnnTensor->tensorListidx = AclNNTensor::notInTensorList;
    aclnnTensor->needUpdateTensorDataPtr = true;
    aclnnTensor->atbTensor = variantPack.outTensors.at(i);
    atb::Tensor squeezedAtbTensor = variantPack.outTensors.at(i);
    aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
    CHECK_OPERATION_STATUS_RETURN(
        CallAclCreateTensor(squeezedAtbTensor.desc.shape,
                            squeezedAtbTensor.desc.shape,
                            squeezedAtbTensor,
                            aclnnTensor));
    aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
  }
  ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNOutTensorVariantPack end");
  return atb::NO_ERROR;
}

int FlashAttnScoreOperation::SetAclNNWorkspaceExecutor() {
  ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAttnGetWorkspaceSize start");
  /*
      1. query: aclTensor
      2. key: aclTensor
      3. value: aclTensor
      4. realShift: (pse) aclTensor
      5. dropMask: not used
      6. paddingMask: not used
      7. attenMask: mask
      8. prefix: not used
      9. qStartIdx: nullptr 10.kvStartIdx: nullptr
      11. scaleValue: double scale factor
      12. keepProb: double dropout keep probability
      13. preTokens 14. nextTokens sparse computation not used
      15. headNum
      16. inputLayout
      17. innerPrecise
      18. sparseMode
      19. pseType
      20~23. softmaxMaxOut softmaxSumOut softmaxOutOut attentionOutOut
  */
  ATB_SPEED_LOG_DEBUG(opName_
                      << "FlashAttnScoreOperation GetWorkspaceSize start");
  AclNNVariantPack& aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;

  // Dynamic tensor index calculation with bounds check
  int idx = 3;
  const size_t tensor_count = aclnnVariantPack.aclInTensors.size();
  const int expected_count =
      3 + (param_.hasPse ? 1 : 0) + (param_.hasAttnMask ? 1 : 0);
  if (tensor_count < static_cast<size_t>(expected_count)) {
    ATB_SPEED_LOG_ERROR("flash_attention_score: Expected at least "
                        << expected_count << " input tensors (hasPse="
                        << param_.hasPse << ", hasAttnMask=" << param_.hasAttnMask
                        << "), got " << tensor_count);
    return atb::ERROR_INVALID_PARAM;
  }

  aclTensor* query = aclnnVariantPack.aclInTensors.at(0)->tensor;
  aclTensor* key = aclnnVariantPack.aclInTensors.at(1)->tensor;
  aclTensor* value = aclnnVariantPack.aclInTensors.at(2)->tensor;
  aclTensor* pseShift =
      param_.hasPse ? aclnnVariantPack.aclInTensors.at(idx++)->tensor : nullptr;
  aclTensor* attenMask = param_.hasAttnMask
                             ? aclnnVariantPack.aclInTensors.at(idx)->tensor
                             : nullptr;
  char* inputLayout = const_cast<char*>(param_.inputLayout.c_str());
  auto ret = aclnnFlashAttentionScoreGetWorkspaceSize(
      query,
      key,
      value,
      pseShift,
      nullptr,
      nullptr,
      attenMask,
      nullptr,
      param_.scaleValue,
      param_.keepProb,
      param_.preTokens,
      param_.nextTokens,
      param_.headNum,
      inputLayout,
      param_.innerPrecise,
      param_.sparseMode,
      aclnnVariantPack.aclOutTensors.at(1)->tensor,
      aclnnVariantPack.aclOutTensors.at(2)->tensor,
      aclnnVariantPack.aclOutTensors.at(3)->tensor,
      aclnnVariantPack.aclOutTensors.at(0)->tensor,
      &this->aclnnOpCache_->workspaceSize,
      &this->aclnnOpCache_->aclExecutor);
  if (ret != 0) {
    ATB_SPEED_LOG_ERROR(
        opName_ << "aclnnFlashAttentionScoreGetWorkspaceSize execute failed");
  }
  ATB_SPEED_LOG_DEBUG(opName_
                      << " end, ret:" << ret << ", workspaceSize:"
                      << this->aclnnOpCache_->workspaceSize
                      << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
  return ret;
}

int FlashAttnScoreOperation::ExecuteAclNNOp(uint8_t* workspace,
                                            aclrtStream& stream) {
  ATB_SPEED_LOG_DEBUG(opName_ << "aclnnFlashAttentionScore start");
  auto ret = aclnnFlashAttentionScore(workspace,
                                      this->aclnnOpCache_->workspaceSize,
                                      this->aclnnOpCache_->aclExecutor,
                                      stream);
  if (ret != 0) {
    ATB_SPEED_LOG_ERROR(opName_ << "aclnnFlashAttentionScore execute failed");
  }
  return ret;
}
}  // namespace common
}  // namespace atb_speed
