#ifndef ATB_SPEED_MODELS_KIMIK25_ENCODER_H
#define ATB_SPEED_MODELS_KIMIK25_ENCODER_H

#include <vector>

#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"
#include "nlohmann/json.hpp"
#include "models/base/param/mapping.h"
#include "models/qwen2_5/vision_encoder/encoder_layer.h"
#include "operations/fusion/utils.h"

namespace atb_speed {
namespace kimi {

struct VisionEncoderLayerParam : public qwen::VisionEncoderLayerParam {
  atb_speed::base::Mapping mapping;
  int MlpQuantType = atb_speed::common::LinearQuantType::NO_QUANT;
};

atb::Status KimiK25_EncoderLayer(const VisionEncoderLayerParam& param,
                                 atb::Operation** operation);

}  // namespace kimi
}  // namespace atb_speed
#endif
