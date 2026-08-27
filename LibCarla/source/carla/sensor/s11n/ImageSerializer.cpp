// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "carla/sensor/s11n/ImageSerializer.h"

#include "carla/sensor/data/Image.h"

namespace carla {
namespace sensor {
namespace s11n {

  SharedPtr<SensorData> ImageSerializer::Deserialize(RawData &&data) {
    const auto pixel_format = static_cast<data::EPixelFormat>(DeserializeHeader(data).pixel_format);
    auto image = SharedPtr<data::Image>(new data::Image{std::move(data)});
    // Set alpha of each pixel in the buffer to max to make it 100% opaque.
    // Only valid for the legacy 4-byte-per-pixel BGRA_U8 layout: this walks
    // the buffer as an array of 4-byte Color structs and stomps the 4th byte
    // of each group. For the HDR/raw payloads (RGB_U16/RGB_F32/BAYER_RGGB_U16)
    // that 4th byte is not an alpha channel — for RGB_F32 in particular it is
    // the sign+top-exponent byte of a float, so doing this would corrupt
    // nearly every value.
    if (pixel_format == data::EPixelFormat::BGRA_U8) {
      for (auto &pixel : *image) {
        pixel.a = 255u;
      }
    }
    return image;
  }

} // namespace s11n
} // namespace sensor
} // namespace carla
