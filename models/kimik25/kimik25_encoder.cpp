#include "models/qwen2_5/operation/qwen_vl_operations.h"
#include "models/kimik25/kimik25_encoder.h"
#include <glog/logging.h>
#include <gflags/gflags.h>

DECLARE_bool(enable_atb_comm_multiprocess);

#include "operations/aclnn/ops/dequant_swiglu_quant_operation.h"
#include "operations/aclnn/ops/dynamic_quant_operation.h"
#include "operations/aclnn/ops/w8a8_operation.h"

namespace atb_speed {
namespace kimi {

namespace {

bool IsMlpW8A8DynamicQuant(const VisionEncoderLayerParam& param) {
  return param.MlpQuantType ==
         atb_speed::common::LinearQuantType::LINEAR_W8A8_DYNAMIC_QUANT;
}

atb::Status CreateDynamicQuantOperation(atb::Operation** op) {
  *op = new atb_speed::common::DynamicQuantOperation("KimiK25DynamicQuant");
  return atb::NO_ERROR;
}

atb::Status CreateW8A8LinearOperation(const VisionEncoderLayerParam& param,
  atb::Operation** op, bool hasBias) {
atb_speed::common::AclNNQuantMatmulParam aclnnParam;
  aclnnParam.transposeB = true;
  aclnnParam.hasPerTokenScale = true;
  aclnnParam.isBF16 = param.isBF16;
  aclnnParam.hasBias = hasBias;
  aclnnParam.matmulBackend = atb_speed::common::OpBackend::ATB;
  *op = new atb_speed::common::W8A8Operation("KimiK25W8A8Linear", aclnnParam);
  return atb::NO_ERROR;
}

atb::Status CreateDequantSwigluQuantOperation(atb::Operation** op) {
  atb_speed::common::AclNNDequantSwigluQuantParam aclnnParam;
  aclnnParam.activateLeft = true;
  aclnnParam.quantMode = "dynamic";
  aclnnParam.inTensorsNum = 1;
  *op = new atb_speed::common::DequantSwigluQuantOperation(
      "KimiK25DequantSwigluQuant", aclnnParam);
  return atb::NO_ERROR;
}

atb::Status CreateLinearParallelOperation(
  const VisionEncoderLayerParam &vision_encoder_param, atb::Operation **op,
  bool trans_weight) {
atb::infer::LinearParallelParam param;
param.type = atb::infer::LinearParallelParam::ParallelType::LINEAR_ALL_REDUCE;

// param.rank = vision_encoder_param.rank;
// param.rankSize = vision_encoder_param.worldSize;
param.rank = vision_encoder_param.mapping.Get(base::ATTN_TP).rank;
param.rankSize = vision_encoder_param.mapping.Get(base::ATTN_TP).rankIds.size();
param.backend = vision_encoder_param.backend;
vision_encoder_param.mapping.Get(base::ATTN_TP).InitCommDomain(param.hcclComm, param.commDomain);

if (!FLAGS_enable_atb_comm_multiprocess) {
    param.commMode = atb::infer::CommMode::COMM_MULTI_THREAD;
}
param.transWeight = trans_weight;
CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param, op));
return atb::NO_ERROR;
}

}  // namespace

std::map<std::string, std::vector<std::string>>
GetKimiK25LayerTensorCandidates() {
  std::map<std::string, std::vector<std::string>> layerTensorCandidates = {
      {"default_weight",
       {"in_input_norm_weight",
        "in_input_norm_bias",
        "in_post_attn_norm_weight",
        "in_post_attn_norm_bias",
        "in_qkv_weight",
        "in_qkv_bias",
        "in_attn_proj_weight",
        "in_attn_proj_bias",
        "in_linear_fc1_weight",
        "in_linear_fc1_bias",
        "in_linear_fc1_offset",
        "in_linear_fc1_scale",
        "in_linear_fc2_weight",
        "in_linear_fc2_bias",
        "in_linear_fc2_offset",
        "in_linear_fc2_scale",
        "in_q_weight",
        "in_q_bias",
        "in_k_weight",
        "in_k_bias",
        "in_v_weight",
        "in_v_bias"}},
      {"default_input",
       {
           "in_hidden_states",
           "in_cos_embedding",
           "in_sin_embedding",
           "in_seq_len",
       }},
      {"out", {"out"}},
      {"parallel_intermediate_out", {"proj_add_bias_out", "fc2_add_bias_out"}},
      {"intermediate_out",
       {"input_norm_out",
        "qkv_linear_out",
        "intermediate_q",
        "intermediate_k",
        "intermediate_v",
        "intermediate_rope_k",
        "intermediate_rope_q",
        "intermediate_atten_out",
        "intermediate_proj_out",
        "intermediate_add_out",
        "intermediate_post_atten_norm_out",
        "intermediate_fc1_out",
        "intermediate_activation_out",
        "intermediate_fc2_out"}},
      {"quant", {
        "intermediate_mlp_quant_input", 
        "intermediate_mlp_quant_input_scale",
        "intermediate_activation_out_bf16",
        "intermediate_activation_quant_scale",
        "intermediate_fc2_out_s"
      }},
      {"q_len", {"in_q_len"}}};
  return layerTensorCandidates;
}

static std::map<std::string, uint32_t> ConstructTensorMap(
    const VisionEncoderLayerParam& param,
    uint32_t& inTensorNum,
    uint32_t& outTensorNum,
    uint32_t& internalTensorNum) {
  auto layerTensorCandidates = GetKimiK25LayerTensorCandidates();
  std::vector<std::string> inTensorList = {};
  std::vector<std::string> intermediateTensorList = {};
  std::vector<std::string> outTensorList = {};
  // weight
  atb_speed::common::AddTensorToList(
      layerTensorCandidates, "default_weight", inTensorList);
  // input
  atb_speed::common::AddTensorToList(
      layerTensorCandidates, "default_input", inTensorList);
  // out
  atb_speed::common::AddTensorToList(
      layerTensorCandidates, "out", outTensorList);
  // intermediate
  atb_speed::common::AddTensorToList(
      layerTensorCandidates, "intermediate_out", intermediateTensorList);
  if (param.worldSize > 1) {
    atb_speed::common::AddTensorToList(layerTensorCandidates,
                                       "parallel_intermediate_out",
                                       intermediateTensorList);
  }
  if (IsMlpW8A8DynamicQuant(param)){
    atb_speed::common::AddTensorToList(
        layerTensorCandidates, "quant", intermediateTensorList);
  }
  inTensorNum = inTensorList.size();
  outTensorNum = outTensorList.size();
  internalTensorNum = intermediateTensorList.size();

  return atb_speed::common::GetTensorMap(
      inTensorList, outTensorList, intermediateTensorList);
}

atb::Status CreateRopeOperation(
  const VisionEncoderLayerParam& param, atb::Operation **op) {
  atb::infer::RopeParam ropeParam;
  ropeParam.rotaryCoeff = param.hiddenSizePerAttentionHead;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(ropeParam, op));
  return atb::NO_ERROR;
}

atb::Status BuildKimiK25AttentionBlock(
    const VisionEncoderLayerParam& param,
    const std::map<std::string, uint32_t>& tensorMap,
    atb::GraphParam& opGraph,
    bool isTp) {
  atb::Node inputNormNode;
  atb::Node qkvlinearNode;
  atb::Node splitNode;
  atb::Node ropeNode;
  atb::Node selfAttentionNode;
  atb::Node outProjNode;
  atb::Node selfResidualAddNode;
  atb::Node proj_add_biasNode;

  // Input Norm
  qwen::CreateLayerNormOperation(param, &inputNormNode.operation);
  inputNormNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"in_hidden_states", "in_input_norm_weight", "in_input_norm_bias"});
  inputNormNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"input_norm_out"});
  opGraph.nodes.push_back(inputNormNode);

  // QKV Linear
  qwen::CreateLinearOperation(&qkvlinearNode.operation);
  qkvlinearNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"input_norm_out", "in_qkv_weight", "in_qkv_bias"});
  qkvlinearNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"qkv_linear_out"});
  opGraph.nodes.push_back(qkvlinearNode);

  // Split
  qwen::CreateSplitOperation(&splitNode.operation);
  splitNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"qkv_linear_out"});
  splitNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_q", "intermediate_k", "intermediate_v"});
  opGraph.nodes.push_back(splitNode);

  // Rope
  CreateRopeOperation(param, &ropeNode.operation);
  ropeNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap,
                                          {"intermediate_q",
                                           "intermediate_k",
                                           "in_cos_embedding",
                                           "in_sin_embedding",
                                           "in_seq_len"});
  ropeNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_rope_q", "intermediate_rope_k"});
  opGraph.nodes.push_back(ropeNode);

  // Self Attention
  qwen::CreateSelfAttentionOperation(param, &selfAttentionNode.operation);
  selfAttentionNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap,
                                          {"intermediate_rope_q",
                                           "intermediate_rope_k",
                                           "intermediate_v",
                                           "in_seq_len"});
  selfAttentionNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_atten_out"});
  opGraph.nodes.push_back(selfAttentionNode);

  // Output Projection
  if (isTp) {
    CreateLinearParallelOperation(param, &outProjNode.operation);
    outProjNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_atten_out", "in_attn_proj_weight"});
  } else {
    qwen::CreateLinearOperation(&outProjNode.operation);
    outProjNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap,
        {"intermediate_atten_out", "in_attn_proj_weight", "in_attn_proj_bias"});
  }
  outProjNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_proj_out"});
  opGraph.nodes.push_back(outProjNode);

  // Tp Bias Add
  if (isTp) {
    qwen::CreateAddOperation(&proj_add_biasNode.operation);
    proj_add_biasNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_proj_out", "in_attn_proj_bias"});
    proj_add_biasNode.outTensorIds =
        atb_speed::common::GetTensorIdxList(tensorMap, {"proj_add_bias_out"});
    opGraph.nodes.push_back(proj_add_biasNode);
  }

  // Residual Add
  qwen::CreateAddOperation(&selfResidualAddNode.operation);
  std::string residualInput =
      isTp ? "proj_add_bias_out" : "intermediate_proj_out";
  selfResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"in_hidden_states", residualInput});
  selfResidualAddNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_add_out"});
  opGraph.nodes.push_back(selfResidualAddNode);

  return atb::NO_ERROR;
}

atb::Status BuildKimiK25MLPBlock(const VisionEncoderLayerParam& param,
                                 const std::map<std::string, uint32_t>& tensorMap,
                                 atb::GraphParam& opGraph,
                                 bool isTp) {
  atb::Node postNormNode;
  atb::Node fc1Node;
  atb::Node activationNode;
  atb::Node fc2Node;
  atb::Node fc2AllReduceNode;
  atb::Node fc2_add_biasNode;
  atb::Node mlpResidualAddNode;
  const bool quantized_mlp = IsMlpW8A8DynamicQuant(param);

  // Post Norm
  qwen::CreateLayerNormOperation(param, &postNormNode.operation);
  postNormNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_add_out", "in_post_attn_norm_weight", "in_post_attn_norm_bias"});
  postNormNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_post_atten_norm_out"});
  opGraph.nodes.push_back(postNormNode);

  if (quantized_mlp) {
    atb::Node dynamicQuantNode;
    // Dynamic quant before the first W8A8 linear.
    CreateDynamicQuantOperation(&dynamicQuantNode.operation);
    dynamicQuantNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_post_atten_norm_out"});
    dynamicQuantNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap,
        {"intermediate_mlp_quant_input", "intermediate_mlp_quant_input_scale"});
    opGraph.nodes.push_back(dynamicQuantNode);

    // fc1 W8A8 Linear
    CreateW8A8LinearOperation(param, &fc1Node.operation, true);
    fc1Node.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap,
        {"intermediate_mlp_quant_input",
         "in_linear_fc1_weight",
         "in_linear_fc1_scale",
         "intermediate_mlp_quant_input_scale",
         "in_linear_fc1_bias"});
    fc1Node.outTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_fc1_out"});
    fc1Node.inTensorReshapeFuncs.resize(fc1Node.inTensorIds.size());
    fc1Node.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
      if (oldShape.dimNum > 1){
        auto newdim = newShape.dims[0];
        for (auto i = 1; i < oldShape.dimNum; i++) {
          newdim *= oldShape.dims[i];
        }
        newShape.dimNum = 1;
        newShape.dims[0] = newdim;
      }
      else {
          newShape = oldShape;
      }
    };
    opGraph.nodes.push_back(fc1Node);

    if (param.MLPActivationType == atb::infer::ACTIVATION_SWIGLU_FORWARD){  //TODO
      // swiglu + dynamic quant
      CreateDequantSwigluQuantOperation(&activationNode.operation);
      activationNode.inTensorIds = atb_speed::common::GetTensorIdxList(
          tensorMap, {"intermediate_fc1_out"});
      activationNode.outTensorIds = atb_speed::common::GetTensorIdxList(
          tensorMap,
          {"intermediate_activation_out",
          "intermediate_activation_quant_scale"});
      opGraph.nodes.push_back(activationNode);
    }
    else {
      qwen::ActivationType act_type = param.MLPActivationType;
      qwen::CreateActivateOperation(&activationNode.operation, act_type);
      activationNode.inTensorIds = atb_speed::common::GetTensorIdxList(
          tensorMap, {"intermediate_fc1_out"});
      activationNode.outTensorIds = atb_speed::common::GetTensorIdxList(
          tensorMap, {"intermediate_activation_out_bf16"});
      opGraph.nodes.push_back(activationNode);

      atb::Node ActiveQuantNode;
      // Dynamic quant before the first W8A8 linear.
      CreateDynamicQuantOperation(&ActiveQuantNode.operation);
      ActiveQuantNode.inTensorIds = atb_speed::common::GetTensorIdxList(
          tensorMap, {"intermediate_activation_out_bf16"});
          ActiveQuantNode.outTensorIds = atb_speed::common::GetTensorIdxList(
          tensorMap,
          {"intermediate_activation_out", "intermediate_activation_quant_scale"});
      opGraph.nodes.push_back(ActiveQuantNode);
    }
    if (isTp) {
      // fc2 W8A8 Linear
      CreateW8A8LinearOperation(param, &fc2Node.operation, false);
      fc2Node.inTensorIds = atb_speed::common::GetTensorIdxList(
          tensorMap,
          {"intermediate_activation_out",
          "in_linear_fc2_weight",
          "in_linear_fc2_scale",
          "intermediate_activation_quant_scale"});
      fc2Node.outTensorIds =
          atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_fc2_out_s"});
      fc2Node.inTensorReshapeFuncs.resize(fc2Node.inTensorIds.size());
      fc2Node.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        if (oldShape.dimNum > 1){
          auto newdim = newShape.dims[0];
          for (auto i = 1; i < oldShape.dimNum; i++) {
            newdim *= oldShape.dims[i];
          }
          newShape.dimNum = 1;
          newShape.dims[0] = newdim;
        }
        else {
            newShape = oldShape;
        }
      };
      opGraph.nodes.push_back(fc2Node);

      std::string downInput = "intermediate_fc2_out_s";
        
      qwen::CreateAllReduceOperation(param, &fc2AllReduceNode.operation);
      fc2AllReduceNode.inTensorIds = atb_speed::common::GetTensorIdxList(
          tensorMap, {downInput});
      fc2AllReduceNode.outTensorIds = atb_speed::common::GetTensorIdxList(
          tensorMap, {"intermediate_fc2_out"});
      opGraph.nodes.push_back(fc2AllReduceNode);
      
    }
    else {
      // fc2 W8A8 Linear
      CreateW8A8LinearOperation(param, &fc2Node.operation, true);
      fc2Node.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap,
        {"intermediate_activation_out",
        "in_linear_fc2_weight",
        "in_linear_fc2_scale",
        "intermediate_activation_quant_scale",
        "in_linear_fc2_bias"});
      fc2Node.outTensorIds =
          atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_fc2_out"});
      fc2Node.inTensorReshapeFuncs.resize(fc2Node.inTensorIds.size());
      fc2Node.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        if (oldShape.dimNum > 1){
          auto newdim = newShape.dims[0];
          for (auto i = 1; i < oldShape.dimNum; i++) {
            newdim *= oldShape.dims[i];
          }
          newShape.dimNum = 1;
          newShape.dims[0] = newdim;
        }
        else {
            newShape = oldShape;
        }
      };
      opGraph.nodes.push_back(fc2Node);
    }

  } else {
    // fc1 Linear
    qwen::CreateLinearOperation(&fc1Node.operation);
    fc1Node.inTensorIds =
        atb_speed::common::GetTensorIdxList(tensorMap,
                                            {"intermediate_post_atten_norm_out",
                                             "in_linear_fc1_weight",
                                             "in_linear_fc1_bias"});
    fc1Node.outTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_fc1_out"});
    opGraph.nodes.push_back(fc1Node);

    qwen::ActivationType act_type = param.MLPActivationType;
    qwen::CreateActivateOperation(&activationNode.operation, act_type);
    activationNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_fc1_out"});
    activationNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_activation_out"});
    opGraph.nodes.push_back(activationNode);

    // fc2 Linear
    if (isTp) {
      CreateLinearParallelOperation(param, &fc2Node.operation);
      fc2Node.inTensorIds = atb_speed::common::GetTensorIdxList(
          tensorMap, {"intermediate_activation_out", "in_linear_fc2_weight"});
    } else {
      qwen::CreateLinearOperation(&fc2Node.operation);
      fc2Node.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap,
        {"intermediate_activation_out",
          "in_linear_fc2_weight",
          "in_linear_fc2_bias"});
    }
    fc2Node.outTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_fc2_out"});
    opGraph.nodes.push_back(fc2Node);
  }
  
    // Tp Bias Add
  if (isTp) {
    qwen::CreateAddOperation(&fc2_add_biasNode.operation);
    fc2_add_biasNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_fc2_out", "in_linear_fc2_bias"});
    fc2_add_biasNode.outTensorIds =
        atb_speed::common::GetTensorIdxList(tensorMap, {"fc2_add_bias_out"});
    opGraph.nodes.push_back(fc2_add_biasNode);
  }

  // MLP Residual Add
  std::string downInput =
      (quantized_mlp || isTp) ? "fc2_add_bias_out" : "intermediate_fc2_out";
  qwen::CreateAddOperation(&mlpResidualAddNode.operation);
  mlpResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {downInput, "intermediate_add_out"});
  mlpResidualAddNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"out"});
  opGraph.nodes.push_back(mlpResidualAddNode);

  return atb::NO_ERROR;
}

atb::Status KimiK25_EncoderLayer(const VisionEncoderLayerParam& param,
                                 atb::Operation** operation) {
  atb::GraphParam opGraph;
  opGraph.name = "KimiK25_Vision_Encoder_layer";
  bool isTp = (param.worldSize > 1);

  // tensor names
  std::map<std::string, uint32_t> tensorMap =
      ConstructTensorMap(param,
                         opGraph.inTensorNum,
                         opGraph.outTensorNum,
                         opGraph.internalTensorNum);

  // Attention
  BuildKimiK25AttentionBlock(param, tensorMap, opGraph, isTp);
  // MLP
  BuildKimiK25MLPBlock(param, tensorMap, opGraph, isTp);
  opGraph.inferShapeFunc =
      [=](const atb::SVector<atb::TensorDesc>& inTensorDescs,
          atb::SVector<atb::TensorDesc>& outTensorDescs) {
        if (!inTensorDescs.empty() && !outTensorDescs.empty()) {
          outTensorDescs.at(0) = inTensorDescs.at(0);
        }
        return atb::NO_ERROR;
      };
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
  return atb::NO_ERROR;
}

}  // namespace kimi
}  // namespace atb_speed
