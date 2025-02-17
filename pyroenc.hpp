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

	struct Impl;
private:
	std::unique_ptr<Impl> impl;
};
}