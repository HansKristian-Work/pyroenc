# PyroEnc

PyroEnc is a simple and standalone library for encoding video with Vulkan video,
suitable for direct integration in a Vulkan application.
It should also be useful as a sample of how to use Vulkan video encoding.

## Partnership

Development of this Vulkan video encoding library has been made in partnership with
[3dverse technologies inc](https://3dverse.com/).

## Device support

As of May 2024, it only works on NVIDIA. The goal is to support every correct Vulkan video
implementation, but RADV and AMD Windows drivers are known to crash.
PyroEnc is actively tested against Vulkan validation layers.

TODO: Current status of non-NVIDIA drivers is unknown.

## Features

Currently, H.264 encode with a simple I and P GOP structure is supported.
H.265 is also supported, but it is highly experimental.
The end goal is to support:

- H.264
- H.265 (8-bit / 10-bit / HDR)
- AV1 (if/when that happens)

The focus of this library is real-time, low-latency streaming, suitable for game streaming.
This focus may extend to high-quality, high-latency encoding as well should
the need arise.

### Currently supported

- Take RGB as input (YCbCr conversion happens on GPU)
- Adjust rate control dynamically
- Insert IDR frames on-demand

### Future features when drivers mature

- Intra-refresh

## API

NOTE: The API is not frozen yet and will change as the implementation matures.

### Creating Encoder

To create an encoder, various information must be provided, such as:

- VkInstance
  - Must enable Vulkan 1.3 or above in apiVersion
- VkPhysicalDevice
- VkDevice
  - Must support Vulkan 1.3 and have `VK_KHR_push_descriptor` extension enabled, or Vulkan 1.4 with pushDescriptor feature enabled
  - Must have video extensions and queues enabled 
    - Potentially, PyroEnc can provide an interface to help doing this, but currently does not
- vkGetInstanceProcAddr callback
- VkQueue + queue family index for conversion queue and encode queue
  - Conversion queue can be either graphics or compute. PyroEnc only uses compute shaders
- Video encoding parameters
  - Width
  - Height
  - Encoding profile
  - Frame rate
  - Tuning parameters
  - Quality level. 0.0f maps to qualityLevel = 0 and 1.0f maps to maxQualityLevels - 1.

```
#include "pyroenc.hpp"
using namespace PyroEnc;

Encoder encoder;
EncoderCreateInfo info = {};
encoder.init_encoder(info);
```

### Updating rate control

This can be called at any time and takes effect from next `send_frame`.
VBR, CBR and ConstantQP mode are supported which directly maps to Vulkan video rate control.
It should be called once before the first frame is sent to encoder.

```
RateControlInfo info = {};
encoder.set_rate_control_info(info);
```

### Sending frame to be encoded

Currently, only RGB input is supported.

```
FrameInfo info = {};
info.view = imageView; // Must be UNORM format with usage VK_IMAGE_USAGE_SAMPLED_BIT.
info.width = width; // Must match the encoder. Scaling is currently not supported.
info.height = height; // Must match the encoder. Scaling is currently not supported.
info.pts = pts++; // Not semantically important to PyroEnc, but it is passed back in EncodedFrame.
info.force_idr = ...; // If true, forces IDR frame to be generated, restarting the GOP.
encoder.send_frame(info);
```

`Encoder::send_frame()` will call `vkQueueSubmit2`, and the application must ensure
that `Encoder::send_frame()` is not called concurrently with any other uses
of either the conversion VkQueue or encoding VkQueue. It is invalid to
concurrently submit to the same VkQueue in Vulkan.

For synchronization, the image view's memory must be visible and owned by the queue.
Generally, this will be `VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT` and `VK_ACCESS_2_SHADER_SAMPLED_READ_BIT`,
with image layout `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`. `Encoder::get_conversion_dst_stage()` and friends
can be called to determine this. For example, if you were rendering to the image,
and then want to send it to encode.

```
VkImageMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
barrier.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
barrier.newLayout = encoder.get_conversion_image_layout();
barrier.dstStageMask = encoder.get_conversion_dst_stage();
barrier.dstAccessMask = encoder.get_conversion_dst_access();
// Fill in the rest
```

To avoid a write-after-read hazard, wait for `encoder.get_conversion_stage_stage()` to complete
execution. The only operation done on the image is to convert color space to YCbCr, so
it should complete very quickly and not stall the graphics queue for long.

### Receiving frame

After sending a frame for encode, you can receive 0 or more frames.
In the current implementation without B-frames, there is a 1:1 correspondence between
sending a frame and receiving it. This API style is similar to FFmpeg.

```
EncodedFrame frame;
while (encoder.receive_encoded_frame(frame) == Result::Success)
{
    // The API is async. Wait for frame to complete encode before we can grab bitstream.
    frame.wait();
    if (frame.get_status() == VK_QUERY_RESULT_STATUS_COMPLETE_KHR)
    {
        // For IDR frames, make sure we write out SPS/PPS as well.
        // To avoid additional memory allocations to pack the payloads together,
        // let application handle it.
        if (frame.is_idr())
            write_packet(encoder.get_encoded_parameters(), encoder.get_encoded_parameters_size());
        write_packet(frame.get_pts(), frame.get_dts(), frame.get_payload(), frame.get_size());
    }
}
```

`EncodedFrame` holds ownership of an entry in the encoder frame pool.
If there are too many `EncodedFrame` objects in flight when calling `send_frame()`,
it may return `Result::NotReady`.
Generally, there is little need to go beyond double-buffering the encoded frame.
All `EncodedFrame` objects must be destroyed before destroying the parent `Encoder`.

## Building

The project is just a simple single C++ file and single header.
The intended use is a static library.

```
add_subdirectory(pyroenc)
target_link_libraries(my-project PRIVATE pyroenc)
```

Alternatively, you can just add `pyroenc.cpp` to your build system.
The only include path required is Vulkan headers.

## Sample

As a sample demonstrating how to use the API, see the `test/` folder.
The samples are written using Granite to avoid a ton of boilerplate.
Run `checkout_granite.sh` from top folder before invoking CMake. In this case, tests are enabled.
This checkout is not done automatically to avoid bloating the simple nature of this library.

```
# Decode 10 seconds of 1080p test video as raw RGBA
ffmpeg -i mytest.mkv -vf scale=1920:1080 -an -t 10 -pix_fmt rgba test.rgb

# Encode raw H.264 at 1920x1080, 60 fps, 15 mbits. The test assumes low-latency CBR encoding.
./pyroenc-test test.h264 test.rgb 1920 1080 60 15000

# Playback
ffplay test.h264
```
