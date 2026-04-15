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
#ifndef FLASH_ATTN_OPERATION_H
#define FLASH_ATTN_OPERATION_H
#include "atb_speed/utils/operation_util.h"
#include "cstring"
#include "operations/aclnn/core/acl_nn_operation.h"
/*
    This Operation is training Forward Op of FlashAttention
    Split Batch Template Condition:
        S2 < 128 && S1 * S2 * headnum * sizeof(dtype) < 64 * 1024 * 2
*/
namespace atb_speed {
namespace common {

constexpr int64_t PRE_NEXT_TOKENS_DEFAULT_VALUE = 2147483647;

struct AclnnFAScoreParam {
  bool hasPse{false};
  bool hasAttnMask{true};
  int64_t preTokens{PRE_NEXT_TOKENS_DEFAULT_VALUE};
  int64_t nextTokens{PRE_NEXT_TOKENS_DEFAULT_VALUE};
  int64_t headNum{0};
  double scaleValue{0.0};
  double keepProb{1.0};
  int64_t innerPrecise{0};
  int64_t sparseMode{0};
  int64_t pseType{1};
  std::string inputLayout = "";
};

class FlashAttnScoreOperation : public AclNNOperation {
 public:
  explicit FlashAttnScoreOperation(const std::string& name,
                                   AclnnFAScoreParam param);
  ~FlashAttnScoreOperation() override;
  atb::Status InferShape(
      const atb::SVector<atb::TensorDesc>& inTensorDescs,
      atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
  uint32_t GetInputNum() const override;
  uint32_t GetOutputNum() const override;

 protected:
  int SetAclNNWorkspaceExecutor() override;
  int ExecuteAclNNOp(uint8_t* workspace, aclrtStream& stream) override;
  atb::Status CreateAclNNInTensorVariantPack(
      const atb::VariantPack& variantPack) override;
  atb::Status CreateAclNNOutTensorVariantPack(
      const atb::VariantPack& variantPack) override;

 private:
  AclnnFAScoreParam param_;
};
}  // namespace common
}  // namespace atb_speed
#endif
