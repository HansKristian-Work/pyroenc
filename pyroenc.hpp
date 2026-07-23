// Copyright (c) 2024 Arntzen Software AS
// SPDX-License-Identifier: MIT

#pragma once

#include <vulkan/vulkan.h>
#include <memory>

namespace PyroEnc
{
struct QueueInfo
{
	VkQueue queue;
	uint32_t family_index;
};

enum class Profile
{
	H264_High,
	H265_Main,
	H265_Main10
};

struct EncoderDirectYCbCrInfo
{
	VkSamplerYcbcrModelConversion ycbcr_conversion;
	VkSamplerYcbcrRange ycbcr_range;
	VkChromaLocation chroma_location_x;
	VkChromaLocation chroma_location_y;
	// Should be SRGB_NONLINEAR or HDR10.
	VkColorSpaceKHR color_space;
};

struct EncoderDirectYCbCrImageInfo
{
	// The image will be created with flags = MUTABLE | EXTENDED_USAGE,
	// usage = VIDEO | STORAGE | TRANSFER_DST.
	// Client is responsible for synchronizing with the encode queue using semaphores
	// and transitioning into ENCODE_SRC layout.
	// The image received in "image" is idle on the GPU and there is no need to consider write-after-read hazards.
	// Client ensures that these usage flags and creation flags are supported.
	// The image handle is only valid until next call to send_frame.
	// The proxy_image_view must be passed into view when encoding.
	// The proxy_image_view is not necessarily a valid VkImageView handle and must not be accessed.
	// The image is CONCURRENT shared with encode queue and conversion queue as passed into encoder.
	VkImage image;
	VkImageView proxy_image_view;
	VkFormat image_format;
	VkFormat plane_formats[3];
	uint32_t num_planes;
	uint32_t active_width;
	uint32_t active_height;

	// The actual resolution of the image. The full image should be filled with information,
	// usually clamp_to_edge behavior or something like that, since it will participate in compression,
	// it just gets cropped away.
	uint32_t padded_width;
	uint32_t padded_height;
};

struct EncoderCreateInfo
{
	PFN_vkGetInstanceProcAddr get_instance_proc_addr;
	VkInstance instance;
	VkPhysicalDevice gpu;
	VkDevice device;

	QueueInfo conversion_queue;
	QueueInfo encode_queue;
	uint32_t width;
	uint32_t height;
	Profile profile;

	uint32_t frame_rate_num;
	uint32_t frame_rate_den;
	float quality_level;

	// Can be set to non-zero if VK_KHR_video_encode_intra_refresh is supported.
	// Intra refresh will be enabled if supported.
	// The intra refresh period will complete over
	// this number of frames. The period may be lowered
	// depending on implementation support.
	uint32_t intra_refresh_period;

	// If not nullptr, the encoder is set to direct YCbCr mode where
	// the user is responsible for converting to YCbCr with padding
	// on edge of the image to align to codec expectations.
	// The chroma siting (left or center) and YCbCr range is encoded in the VUI of the bitstream.
	// If the encoder is created with this option, the image view passed into FrameInfo must be equal to the proxy image view
	// obtained from EncoderDirectYCbCrImageInfo::proxy_image_view.
	const EncoderDirectYCbCrInfo *direct_ycbcr_info;

	struct
	{
		VkVideoEncodeTuningModeKHR tuning;
		VkVideoEncodeContentFlagsKHR content;
		VkVideoEncodeUsageFlagsKHR usage;
	} hints;
};

struct FrameInfo
{
	VkImageView view;
	uint32_t width;
	uint32_t height;
	int64_t pts;
	bool force_idr;
};

enum class RateControlMode
{
	VBR,
	CBR,
	ConstantQP
};

struct RateControlInfo
{
	uint32_t bitrate_kbits;
	uint32_t max_bitrate_kbits;
	int32_t constant_qp;
	// If UINT32_MAX, open/infinite GOP is used. Need force_idr to force a new IDR frame.
	// When intra refresh is used, open GOP is forced.
	uint32_t gop_frames;
	RateControlMode mode;
};

struct EncodedFrame
{
	EncodedFrame();
	~EncodedFrame();

	int64_t get_pts() const;
	int64_t get_dts() const;

	bool wait(uint64_t timeout = UINT64_MAX) const;

	size_t get_size() const;
	const void *get_payload() const;
	VkQueryResultStatusKHR get_status() const;
	bool is_idr() const;

	// In seconds. How long time the encoding operation took.
	// If negative, the implementation does not support timestamps in video encode queue.
	double get_encoding_overhead() const;

	EncodedFrame(EncodedFrame &&other) noexcept;
	EncodedFrame &operator=(EncodedFrame &&other) noexcept;

	struct Impl;
	Impl &get_impl();

private:
	std::unique_ptr<Impl> impl;
};

enum class Severity
{
	Debug,
	Warn,
	Error
};

enum class Result : int
{
	Success = 1,
	NotReady = 0,
	Error = -1
};

struct LogCallback
{
	virtual ~LogCallback() = default;
	virtual void log(Severity severity, const char *msg) = 0;
};

class Encoder
{
public:
	Encoder();
	~Encoder();

	void set_log_callback(LogCallback *cb);

	Result init_encoder(const EncoderCreateInfo &info);
	bool set_rate_control_info(const RateControlInfo &info);

	Result send_frame(const FrameInfo &info);
	Result send_eof();
	Result receive_encoded_frame(EncodedFrame &frame);

	VkPipelineStageFlags2 get_conversion_dst_stage() const;
	VkAccessFlags2 get_conversion_dst_access() const;
	VkImageLayout get_conversion_image_layout() const;

	const void *get_encoded_parameters() const;
	size_t get_encoded_parameters_size() const;

	// If enabled and supported.
	bool intra_refresh_enabled() const;

	// Allocates a frame slot equivalent to send_frame, may return NotReady.
	Result allocate_direct_ycbcr_image_info(EncoderDirectYCbCrImageInfo &info);
	// If a frame slot is allocated but will not be submitted (e.g. EOF happens), it must be passed back again.
	void discard_direct_ycbcr_image_info(EncoderDirectYCbCrImageInfo &info);

	struct Impl;
private:
	std::unique_ptr<Impl> impl;
};
}