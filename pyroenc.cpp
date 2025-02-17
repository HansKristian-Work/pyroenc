// Copyright (c) 2024 Arntzen Software AS
// SPDX-License-Identifier: MIT

#include "pyroenc.hpp"
#include <atomic>
#include <utility>
#include <vector>
#include <queue>
#include <assert.h>
#include <stdio.h>

#define VK_CALL(x) table.x

namespace PyroEnc
{
struct VkTable
{
#define INSTANCE_FUNCTION(fun) PFN_vk##fun vk##fun
#define DEVICE_FUNCTION(fun) PFN_vk##fun vk##fun
#define DEVICE_FUNCTION_FALLBACK(fun, fall) PFN_vk##fun vk##fun
#include "pyroenc_vk_table.inl"
#undef INSTANCE_FUNCTION
#undef DEVICE_FUNCTION
#undef DEVICE_FUNCTION_FALLBACK
};

struct ConversionRegisters
{
	uint32_t width;
	uint32_t height;
	float dither_strength;
};

struct Timeline
{
	VkSemaphore timeline;
	// Represents the highest value submitted to Vulkan.
	uint64_t value;
};

struct Pipeline
{
	VkPipeline pipeline;
	VkPipelineLayout layout;
	VkDescriptorSetLayout set_layout;
};

struct Memory
{
	VkDeviceMemory memory;
	void *mapped;
};

struct Image
{
	VkImage image;
	VkImageView view;
	Memory memory;
};

struct Buffer
{
	VkBuffer buffer;
	Memory memory;
};

struct Frame
{
	uint64_t compute;
	uint64_t encode;

	VkCommandBuffer convert_cmd;
	VkCommandBuffer encode_cmd;

	Image image_ycbcr;
	Buffer payload;
};

struct VideoProfile
{
	VkVideoProfileInfoKHR profile_info = { VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR };
	VkVideoProfileListInfoKHR profile_list = { VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR };
	VkVideoEncodeUsageInfoKHR usage_info = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_USAGE_INFO_KHR };

	struct Format
	{
		VkFormat format = VK_FORMAT_UNDEFINED;
		VkFormat luma_format = VK_FORMAT_UNDEFINED;
		VkFormat chroma_format = VK_FORMAT_UNDEFINED;
		uint32_t subsample_log2[2] = {};
		VkImageFormatProperties format_properties = {};
	} input, dpb;

	union
	{
		struct
		{
			VkVideoEncodeH264ProfileInfoKHR profile;
		} h264;

		struct
		{
			VkVideoEncodeH265ProfileInfoKHR profile;
		} h265;
	};

	bool setup(Encoder::Impl &impl, Profile profile);
	Format get_format_info(Encoder::Impl &impl, VkImageUsageFlags usage);
};

struct VideoEncoderCaps
{
	VkVideoCapabilitiesKHR video_caps = { VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR };
	VkVideoEncodeCapabilitiesKHR encode_caps = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_CAPABILITIES_KHR };

	union
	{
		struct
		{
			VkVideoEncodeH264CapabilitiesKHR caps;
		} h264;

		struct
		{
			VkVideoEncodeH265CapabilitiesKHR caps;
		} h265;
	};

	bool setup(Encoder::Impl &impl);

	uint32_t get_aligned_width(uint32_t width) const;
	uint32_t get_aligned_height(uint32_t height) const;
};

struct VideoSession
{
	std::vector<Memory> memory;
	VkVideoSessionKHR session = VK_NULL_HANDLE;

	bool init(Encoder::Impl &impl);
	void destroy(Encoder::Impl &impl);
};

struct VideoSessionParameters
{
	VkVideoSessionParametersKHR params = VK_NULL_HANDLE;
	VkVideoEncodeQualityLevelInfoKHR quality_level =
		{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_QUALITY_LEVEL_INFO_KHR };
	VkVideoEncodeQualityLevelPropertiesKHR quality_level_props =
		{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_QUALITY_LEVEL_PROPERTIES_KHR };

	union
	{
		struct
		{
			StdVideoH264SequenceParameterSet sps;
			StdVideoH264PictureParameterSet pps;
			StdVideoH264SequenceParameterSetVui vui;
			VkVideoEncodeH264QualityLevelPropertiesKHR quality_level_props;
		} h264;

		struct
		{
			StdVideoH265SequenceParameterSet sps;
			StdVideoH265PictureParameterSet pps;
			StdVideoH265VideoParameterSet vps;
			StdVideoH265SequenceParameterSetVui vui;
			VkVideoEncodeH265QualityLevelPropertiesKHR quality_level_props;
		} h265;
	};

	bool init(Encoder::Impl &impl);
	void destroy(Encoder::Impl &impl);
	std::vector<uint8_t> encoded_parameters;

	bool init_h264(Encoder::Impl &impl);
	bool init_h265(Encoder::Impl &impl);
};

struct RateControl
{
	RateControl();
	VkVideoCodingControlInfoKHR ctrl_info =
		{ VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR };
	VkVideoEncodeRateControlInfoKHR rate_info =
		{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_INFO_KHR };
	VkVideoEncodeRateControlLayerInfoKHR layer =
		{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_LAYER_INFO_KHR };

	union
	{
		struct
		{
			VkVideoEncodeH264RateControlInfoKHR rate_control;
			VkVideoEncodeH264RateControlLayerInfoKHR layer;
		} h264;

		struct
		{
			VkVideoEncodeH265RateControlInfoKHR rate_control;
			VkVideoEncodeH265RateControlLayerInfoKHR layer;
		} h265;
	};

	bool init(Encoder::Impl &impl);

	uint64_t frame_index = 0;
	uint64_t gop_frame_index = 0;
	uint32_t idr_pic_id = 0;
	bool needs_reset = true;

	RateControlInfo info;
};

// This will limit how many EncodedFrames we can have in flight.
// More than or 2 or 3 seems highly unusual.
static constexpr uint32_t FramePoolSize = 4;
// TODO: For B-frames, we might need 3.
static constexpr uint32_t DPBSize = 2;
// TODO: For B-frames, we might need 2.
static constexpr uint32_t MaxActiveReferencePictures = DPBSize - 1;
// Should be enough for everyone.
static constexpr uint32_t MaxPayloadSize = 4 * 1024 * 1024;

struct Encoder::Impl
{
	Impl();
	~Impl();
	bool init_encoder(const EncoderCreateInfo &info);
	bool wait(uint64_t compute_timeline, uint64_t encode_timeline, uint64_t timeout) const;
	bool wait_frame_pool_index(uint32_t index, uint64_t timeout) const;
	void release_frame_pool_index(uint32_t index);
	uint32_t allocate_frame_pool_index();

	bool submit_conversion(const FrameInfo &input, Frame &frame);
	bool submit_encode(Frame &frame, bool &is_idr);
	void transition_conversion_dst_images(VkCommandBuffer cmd);
	void copy_to_ycbcr(VkCommandBuffer cmd, Frame &frame);
	bool record_and_submit_encode(VkCommandBuffer cmd, Frame &frame, bool &is_idr);
	bool submit_encode_command_buffer(VkCommandBuffer cmd);
	void record_rate_control(VkCommandBuffer cmd);
	void record_host_barrier(VkCommandBuffer cmd, Frame &frame);
	void record_acquire_barrier(VkCommandBuffer cmd, const Frame &frame);
	void record_dpb_barrier(VkCommandBuffer cmd);

	Result send_frame(const FrameInfo &input);
	Result send_eof();
	Result receive_encoded_frame(EncodedFrame &frame);
	std::queue<EncodedFrame> encoded_queue;

	Timeline compute_timeline = {};
	Timeline encode_timeline = {};
	Pipeline conversion_pipeline = {};

	VkCommandPool convert_cmd_pool = VK_NULL_HANDLE;
	VkCommandPool encode_cmd_pool = VK_NULL_HANDLE;
	VkQueryPool query_pool = VK_NULL_HANDLE;
	VkQueryPool query_pool_timestamp = VK_NULL_HANDLE;
	uint32_t encode_timestamp_bits = 0;

	VkTable table = {};
	bool func_table_is_valid = false;
	Frame frame_pool[FramePoolSize] = {};
	std::atomic_uint32_t active_frame_pool_indices;

	struct Dpb
	{
		Image dpb[DPBSize];
		Image array_dpb;
		Image luma;
		Image chroma;
		bool dpb_inited = false;
	} dpb = {};

	EncoderCreateInfo info = {};
	VkPhysicalDeviceMemoryProperties mem_props = {};
	VkPhysicalDeviceProperties vk_props = {};
	VkPhysicalDeviceVulkan12Properties vk12_props = {};

	LogCallback *cb = nullptr;

	template <typename... Ts>
	void log(Severity severity, const char *fmt, Ts&&... ts);

	bool allocate_memory(Memory &memory, VkMemoryPropertyFlags props, const VkMemoryRequirements &reqs);
	bool create_buffer(Buffer &buffer, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
	                   const void *pNext);
	bool create_image(Image &image, uint32_t width, uint32_t height, uint32_t layers,
	                  VkFormat format, VkImageUsageFlags usage,
	                  const void *pNext);
	bool create_timeline_semaphore(Timeline &timeline);
	bool create_compute_pipeline(Pipeline &pipeline, const uint32_t *code, size_t code_size,
	                             const VkDescriptorSetLayoutBinding *binding, uint32_t num_bindings,
	                             const VkPushConstantRange *push_ranges, uint32_t num_push_ranges);

	void destroy_image(Image &image);
	void destroy_buffer(Buffer &buffer);
	void destroy_frame_resources(Frame &frame);
	void destroy_dpb();
	void destroy_pipeline(Pipeline &pipeline);
	void destroy_timeline(Timeline &timeline);
	void free_memory(Memory &memory);

	bool init_func_table();
	bool init_frame_resources();
	bool init_command_pools();
	bool init_query_pool();
	bool init_frame_resource(Frame &frame);
	bool init_dpb_resources();
	bool init_pipelines();

	VideoProfile profile;
	VideoEncoderCaps caps;
	VideoSession session;
	VideoSessionParameters session_params;
	RateControl rate;
};

bool Encoder::Impl::create_image(Image &image, uint32_t width, uint32_t height, uint32_t layers, VkFormat format,
                                 VkImageUsageFlags usage, const void *pNext)
{
	image = {};

	VkImageCreateInfo image_info = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
	image_info.format = format;
	image_info.extent = { width, height, 1 };
	image_info.imageType = VK_IMAGE_TYPE_2D;
	image_info.usage = usage;
	image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
	image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	image_info.mipLevels = 1;
	image_info.arrayLayers = layers;
	image_info.samples = VK_SAMPLE_COUNT_1_BIT;
	image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	image_info.pNext = pNext;

	if (VK_CALL(vkCreateImage(info.device, &image_info, nullptr, &image.image)) != VK_SUCCESS)
		return false;

	VkMemoryRequirements reqs;
	VK_CALL(vkGetImageMemoryRequirements(info.device, image.image, &reqs));

	if (!allocate_memory(image.memory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, reqs) &&
	    !allocate_memory(image.memory, 0, reqs))
	{
		return false;
	}

	if (VK_CALL(vkBindImageMemory(info.device, image.image, image.memory.memory, 0)) != VK_SUCCESS)
		return false;

	VkImageViewCreateInfo view_info = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
	view_info.format = format;
	view_info.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS };
	view_info.viewType = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
	view_info.image = image.image;

	if (VK_CALL(vkCreateImageView(info.device, &view_info, nullptr, &image.view)) != VK_SUCCESS)
		return false;

	return true;
}

bool Encoder::Impl::create_buffer(Buffer &buffer, VkDeviceSize size, VkBufferUsageFlags usage,
                                  VkMemoryPropertyFlags props, const void *pNext)
{
	buffer = {};

	VkBufferCreateInfo buffer_info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	buffer_info.size = size;
	buffer_info.usage = usage;
	buffer_info.pNext = pNext;

	const uint32_t family_indices[] = { info.conversion_queue.family_index, info.encode_queue.family_index };
	if (family_indices[0] != family_indices[1])
	{
		buffer_info.sharingMode = VK_SHARING_MODE_CONCURRENT;
		buffer_info.queueFamilyIndexCount = 2;
		buffer_info.pQueueFamilyIndices = family_indices;
	}
	else
		buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	if (VK_CALL(vkCreateBuffer(info.device, &buffer_info, nullptr, &buffer.buffer)) != VK_SUCCESS)
		return false;

	VkMemoryRequirements reqs = {};
	VK_CALL(vkGetBufferMemoryRequirements(info.device, buffer.buffer, &reqs));

	if (!allocate_memory(buffer.memory, props, reqs) ||
	    VK_CALL(vkBindBufferMemory(info.device, buffer.buffer, buffer.memory.memory, 0)) != VK_SUCCESS)
	{
		destroy_buffer(buffer);
		return false;
	}

	return true;
}

bool Encoder::Impl::create_compute_pipeline(Pipeline &pipeline, const uint32_t *code, size_t code_size,
                                            const VkDescriptorSetLayoutBinding *binding, uint32_t num_bindings,
                                            const VkPushConstantRange *push_ranges, uint32_t num_push_ranges)
{
	pipeline = {};

	VkShaderModuleCreateInfo module_info = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
	module_info.pCode = code;
	module_info.codeSize = code_size;

	VkDescriptorSetLayoutCreateInfo set_layout_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
	set_layout_info.bindingCount = num_bindings;
	set_layout_info.pBindings = binding;
	set_layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
	if (VK_CALL(vkCreateDescriptorSetLayout(info.device, &set_layout_info, nullptr, &pipeline.set_layout)) != VK_SUCCESS)
		return false;

	VkPipelineLayoutCreateInfo layout_info = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	layout_info.pPushConstantRanges = push_ranges;
	layout_info.pushConstantRangeCount = num_push_ranges;
	layout_info.setLayoutCount = 1;
	layout_info.pSetLayouts = &pipeline.set_layout;
	if (VK_CALL(vkCreatePipelineLayout(info.device, &layout_info, nullptr, &pipeline.layout)) != VK_SUCCESS)
	{
		destroy_pipeline(pipeline);
		return false;
	}

	VkShaderModule module = VK_NULL_HANDLE;
	if (VK_CALL(vkCreateShaderModule(info.device, &module_info, nullptr, &module)) != VK_SUCCESS)
	{
		destroy_pipeline(pipeline);
		return false;
	}

	VkComputePipelineCreateInfo compute_info = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	compute_info.layout = pipeline.layout;
	compute_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	compute_info.stage.module = module;
	compute_info.stage.pName = "main";
	compute_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;

	VkResult vr = VK_CALL(vkCreateComputePipelines(info.device, VK_NULL_HANDLE,
												   1, &compute_info, nullptr, &pipeline.pipeline));
	VK_CALL(vkDestroyShaderModule(info.device, module, nullptr));
	if (vr != VK_SUCCESS)
		destroy_pipeline(pipeline);
	return vr == VK_SUCCESS;
}

bool Encoder::Impl::create_timeline_semaphore(Timeline &timeline)
{
	VkSemaphoreTypeCreateInfo type_info = { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
	VkSemaphoreCreateInfo sem_info = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
	sem_info.pNext = &type_info;
	type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;

	timeline = {};
	return VK_CALL(vkCreateSemaphore(info.device, &sem_info, nullptr, &timeline.timeline)) == VK_SUCCESS;
}

bool Encoder::Impl::allocate_memory(Memory &memory, VkMemoryPropertyFlags props, const VkMemoryRequirements &reqs)
{
	for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++)
	{
		if ((reqs.memoryTypeBits & (1u << i)) == 0)
			continue;

		if ((props & mem_props.memoryTypes[i].propertyFlags) == props)
		{
			VkMemoryAllocateInfo alloc = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
			alloc.memoryTypeIndex = i;
			alloc.allocationSize = reqs.size;
			if (VK_CALL(vkAllocateMemory(info.device, &alloc, nullptr, &memory.memory)) != VK_SUCCESS)
				continue;

			if ((props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0)
			{
				if (VK_CALL(vkMapMemory(info.device, memory.memory, 0, VK_WHOLE_SIZE, 0, &memory.mapped)) != VK_SUCCESS)
				{
					VK_CALL(vkFreeMemory(info.device, memory.memory, nullptr));
					memory.memory = VK_NULL_HANDLE;
					continue;
				}
			}

			return true;
		}
	}

	memory = {};
	return false;
}

void Encoder::Impl::free_memory(Memory &memory)
{
	// Host memory is automatically unmapped if it is mapped.
	VK_CALL(vkFreeMemory(info.device, memory.memory, nullptr));
	memory = {};
}

void Encoder::Impl::destroy_image(Image &image)
{
	VK_CALL(vkDestroyImageView(info.device, image.view, nullptr));
	VK_CALL(vkDestroyImage(info.device, image.image, nullptr));
	free_memory(image.memory);
	image = {};
}

void Encoder::Impl::destroy_buffer(Buffer &buffer)
{
	VK_CALL(vkDestroyBuffer(info.device, buffer.buffer, nullptr));
	free_memory(buffer.memory);
}

void Encoder::Impl::destroy_frame_resources(Frame &frame)
{
	destroy_image(frame.image_ycbcr);
	destroy_buffer(frame.payload);
	frame = {};
}

void Encoder::Impl::destroy_pipeline(Pipeline &pipeline)
{
	VK_CALL(vkDestroyPipeline(info.device, pipeline.pipeline, nullptr));
	VK_CALL(vkDestroyPipelineLayout(info.device, pipeline.layout, nullptr));
	VK_CALL(vkDestroyDescriptorSetLayout(info.device, pipeline.set_layout, nullptr));
	pipeline = {};
}

void Encoder::Impl::destroy_timeline(Timeline &timeline)
{
	VK_CALL(vkDestroySemaphore(info.device, timeline.timeline, nullptr));
	timeline = {};
}

struct EncodedFrame::Impl
{
	~Impl();

	int64_t pts = 0;
	int64_t dts = 0;
	bool is_idr = false;

	const void *payload = nullptr;
	uint32_t frame_pool_index = UINT32_MAX;
	Encoder::Impl *encoder = nullptr;

	struct Query
	{
		uint32_t offset;
		uint32_t size;
		VkQueryResultStatusKHR status;
	};
	bool get_query(Query &query_data) const;
	double get_encoding_overhead() const;
};

EncodedFrame::EncodedFrame()
{
	impl.reset(new Impl);
}

EncodedFrame::EncodedFrame(PyroEnc::EncodedFrame &&other) noexcept
{
	*this = std::move(other);
}

EncodedFrame &EncodedFrame::operator=(EncodedFrame &&other) noexcept
{
	if (this != &other)
		impl = std::move(other.impl);
	return *this;
}

EncodedFrame::Impl &EncodedFrame::get_impl()
{
	return *impl;
}

EncodedFrame::~EncodedFrame()
{
}

EncodedFrame::Impl::~Impl()
{
	if (encoder)
		encoder->release_frame_pool_index(frame_pool_index);
}

int64_t EncodedFrame::get_pts() const
{
	return impl->pts;
}

int64_t EncodedFrame::get_dts() const
{
	return impl->dts;
}

bool EncodedFrame::Impl::get_query(Query &query_data) const
{
	auto &enc = *encoder;
	auto &table = enc.table;

	if (VK_CALL(vkGetQueryPoolResults(enc.info.device, enc.query_pool, frame_pool_index, 1, sizeof(query_data),
	                                  &query_data, sizeof(query_data),
	                                  VK_QUERY_RESULT_WITH_STATUS_BIT_KHR)) != VK_SUCCESS)
		return false;

	return query_data.status == VK_QUERY_RESULT_STATUS_COMPLETE_KHR;
}

double EncodedFrame::Impl::get_encoding_overhead() const
{
	auto &enc = *encoder;
	auto &table = enc.table;

	if (!enc.query_pool_timestamp)
		return -1.0;

	uint64_t ts[2];

	if (VK_CALL(vkGetQueryPoolResults(enc.info.device, enc.query_pool_timestamp, 2 * frame_pool_index, 2, sizeof(ts),
	                                  ts, sizeof(ts[0]), VK_QUERY_RESULT_64_BIT)) != VK_SUCCESS)
		return -1.0;

	uint64_t tick_offset = (ts[1] - ts[0]) & (((1ull << enc.encode_timestamp_bits) - 1) >> 1);
	return double(tick_offset) * enc.vk_props.limits.timestampPeriod * 1e-9;
}

size_t EncodedFrame::get_size() const
{
	Impl::Query query = {};
	if (!impl->get_query(query))
		return 0;
	else
		return query.size;
}

double EncodedFrame::get_encoding_overhead() const
{
	return impl->get_encoding_overhead();
}

VkQueryResultStatusKHR EncodedFrame::get_status() const
{
	Impl::Query query = {};
	if (!impl->get_query(query))
		return VK_QUERY_RESULT_STATUS_NOT_READY_KHR;
	else
		return query.status;
}

bool EncodedFrame::is_idr() const
{
	return impl->is_idr;
}

bool EncodedFrame::wait(uint64_t timeout) const
{
	return impl->encoder->wait_frame_pool_index(impl->frame_pool_index, timeout);
}

const void *EncodedFrame::get_payload() const
{
	Impl::Query query = {};
	if (!impl->get_query(query))
		return nullptr;
	else
		return static_cast<const uint8_t *>(impl->payload) + query.offset;
}

bool Encoder::Impl::wait(uint64_t compute, uint64_t encode, uint64_t timeout) const
{
	if (!compute_timeline.timeline || !encode_timeline.timeline)
		return false;

	const VkSemaphore sems[] = { compute_timeline.timeline, encode_timeline.timeline };
	const uint64_t values[] = { compute, encode };

	VkSemaphoreWaitInfo wait_info = { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
	wait_info.semaphoreCount = 2;
	wait_info.pSemaphores = sems;
	wait_info.pValues = values;
	return VK_CALL(vkWaitSemaphores(info.device, &wait_info, timeout)) == VK_SUCCESS;
}

bool Encoder::Impl::wait_frame_pool_index(uint32_t index, uint64_t timeout) const
{
	auto &frame = frame_pool[index];
	return wait(frame.compute, frame.encode, timeout);
}

void Encoder::Impl::release_frame_pool_index(uint32_t index)
{
	uint32_t mask = 1u << index;
	uint32_t old = active_frame_pool_indices.fetch_and(~mask, std::memory_order_release);
	(void)old;
	assert((old & mask) != 0);
}

uint32_t Encoder::Impl::allocate_frame_pool_index()
{
	uint32_t bits = active_frame_pool_indices.load(std::memory_order_acquire);
	for (uint32_t i = 0; i < FramePoolSize; i++)
	{
		uint32_t mask = 1u << i;
		if ((bits & mask) == 0)
		{
			uint32_t old = active_frame_pool_indices.fetch_or(mask, std::memory_order_relaxed);
			(void)old;
			assert((old & mask) == 0);

			return i;
		}
	}

	return UINT32_MAX;
}

Encoder::Impl::Impl()
{
	active_frame_pool_indices.store(0, std::memory_order_relaxed);
}

Encoder::Impl::~Impl()
{
	// Relevant if we never initialized the encoder.
	if (!func_table_is_valid)
		return;

	while (!encoded_queue.empty())
		encoded_queue.pop();

	// Hard assertion.
	if (active_frame_pool_indices.load() != 0)
		std::terminate();

	for (uint32_t i = 0; i < FramePoolSize; i++)
		wait_frame_pool_index(i, UINT64_MAX);

	destroy_dpb();
	for (auto &frame : frame_pool)
		destroy_frame_resources(frame);
	destroy_pipeline(conversion_pipeline);
	destroy_timeline(compute_timeline);
	destroy_timeline(encode_timeline);

	VK_CALL(vkDestroyCommandPool(info.device, convert_cmd_pool, nullptr));
	VK_CALL(vkDestroyCommandPool(info.device, encode_cmd_pool, nullptr));
	VK_CALL(vkDestroyQueryPool(info.device, query_pool, nullptr));
	VK_CALL(vkDestroyQueryPool(info.device, query_pool_timestamp, nullptr));

	session_params.destroy(*this);
	session.destroy(*this);
}

bool Encoder::Impl::init_func_table()
{
#define INSTANCE_FUNCTION(fun) \
	table.vk##fun = (PFN_vk##fun)info.get_instance_proc_addr(info.instance, "vk" #fun); \
	if (!table.vk##fun) \
		return false
#define DEVICE_FUNCTION(fun)
#define DEVICE_FUNCTION_FALLBACK(fun, fall)
#include "pyroenc_vk_table.inl"
#undef INSTANCE_FUNCTION
#undef DEVICE_FUNCTION
#undef DEVICE_FUNCTION_FALLBACK

#define INSTANCE_FUNCTION(fun)
#define DEVICE_FUNCTION(fun) \
	table.vk##fun = (PFN_vk##fun)table.vkGetDeviceProcAddr(info.device, "vk" #fun); \
	if (!table.vk##fun) \
		return false
#define DEVICE_FUNCTION_FALLBACK(fun, fall) \
	table.vk##fun = (PFN_vk##fun)table.vkGetDeviceProcAddr(info.device, "vk" #fun); \
	if (!table.vk##fun) \
		table.vk##fun = (PFN_vk##fun)table.vkGetDeviceProcAddr(info.device, "vk" #fall); \
	if (!table.vk##fun) \
		return false
#include "pyroenc_vk_table.inl"
#undef INSTANCE_FUNCTION
#undef DEVICE_FUNCTION
#undef DEVICE_FUNCTION_FALLBACK

	func_table_is_valid = true;
	return true;
}

bool Encoder::Impl::init_frame_resources()
{
	for (auto &frame : frame_pool)
		if (!init_frame_resource(frame))
			return false;

	return true;
}

bool Encoder::Impl::init_dpb_resources()
{
	uint32_t aligned_width = caps.get_aligned_width(info.width);
	uint32_t aligned_height = caps.get_aligned_height(info.height);

	if ((caps.video_caps.flags & VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR) != 0)
	{
		// Could just use the array DPB formulation everywhere, but NV drivers are still bugged as of 570 series :(
		for (auto &img: dpb.dpb)
		{
			if (!create_image(img, aligned_width, aligned_height, 1, profile.dpb.format,
			                  VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR,
			                  &profile.profile_list))
				return false;
		}
	}
	else
	{
		if (!create_image(dpb.array_dpb, aligned_width, aligned_height, DPBSize, profile.dpb.format,
		                  VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR,
		                  &profile.profile_list))
			return false;
	}

	if (!create_image(dpb.luma, aligned_width, aligned_height, 1, profile.input.luma_format,
					  VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT, nullptr))
		return false;

	if (!create_image(dpb.chroma,
	                  aligned_width >> profile.input.subsample_log2[0],
	                  aligned_height >> profile.input.subsample_log2[1],
	                  1,
	                  profile.input.chroma_format,
	                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT, nullptr))
		return false;

	return true;
}

bool Encoder::Impl::init_pipelines()
{
	static const uint32_t code[] =
#include "shaders/rgb_to_yuv.inc"
		;

	VkDescriptorSetLayoutBinding bindings[3] = {};
	VkPushConstantRange push = {};

	bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	bindings[0].binding = 0;
	bindings[0].descriptorCount = 1;
	bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;

	bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	bindings[1].binding = 1;
	bindings[1].descriptorCount = 1;
	bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;

	bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	bindings[2].binding = 2;
	bindings[2].descriptorCount = 1;
	bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;

	push.size = sizeof(ConversionRegisters);
	push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	if (!create_compute_pipeline(conversion_pipeline, code, sizeof(code), bindings, 3, &push, 1))
		return false;

	return true;
}

void Encoder::Impl::transition_conversion_dst_images(VkCommandBuffer cmd)
{
	VkDependencyInfo deps = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
	VkImageMemoryBarrier2 image_barrier[2];

	for (auto &barrier : image_barrier)
	{
		barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
		barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		// Wait for previous frame's copy to YCbCr image to complete.
		barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
		barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
	}

	image_barrier[0].image = dpb.luma.image;
	image_barrier[1].image = dpb.chroma.image;

	deps.imageMemoryBarrierCount = 2;
	deps.pImageMemoryBarriers = image_barrier;

	VK_CALL(vkCmdPipelineBarrier2(cmd, &deps));
}

void Encoder::Impl::copy_to_ycbcr(VkCommandBuffer cmd, Frame &frame)
{
	// This is somewhat unfortunate, but NVIDIA does not expose imageCreateFlags with MUTABLE or EXTENDED_USAGE,
	// so we cannot create YCbCr video image with STORAGE + MUTABLE and take per-plane views.
	// Just copy into the YCbCr planes instead, which isn't ideal, but could be worse.

	VkDependencyInfo deps = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
	VkImageMemoryBarrier2 image_barrier[3];

	for (uint32_t i = 0; i < 3; i++)
	{
		auto &barrier = image_barrier[i];
		barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
		barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

		if (i < 2)
		{
			barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
			barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
			barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
			barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
		}
		else
		{
			barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			// The YCbCr image is idle since we waited for semaphores.
			barrier.srcStageMask = VK_PIPELINE_STAGE_NONE;
			barrier.srcAccessMask = VK_ACCESS_2_NONE;
			barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
			barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
		}
	}

	image_barrier[0].image = dpb.luma.image;
	image_barrier[1].image = dpb.chroma.image;
	image_barrier[2].image = frame.image_ycbcr.image;

	deps.imageMemoryBarrierCount = 3;
	deps.pImageMemoryBarriers = image_barrier;

	VK_CALL(vkCmdPipelineBarrier2(cmd, &deps));

	VkImageCopy regions[2] = {};
	regions[0].extent = { caps.get_aligned_width(info.width), caps.get_aligned_height(info.height), 1 };
	regions[0].srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
	regions[0].dstSubresource = { VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1 };

	regions[1].extent = {
		caps.get_aligned_width(info.width) >> profile.input.subsample_log2[0],
		caps.get_aligned_height(info.height) >> profile.input.subsample_log2[1],
		1,
	};
	regions[1].srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
	regions[1].dstSubresource = { VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 0, 1 };

	VK_CALL(vkCmdCopyImage(cmd, dpb.luma.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	                       frame.image_ycbcr.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	                       1, &regions[0]));

	VK_CALL(vkCmdCopyImage(cmd, dpb.chroma.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	                       frame.image_ycbcr.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	                       1, &regions[1]));

	// Transfer ownership to video queue.
	auto &barrier = image_barrier[2];
	barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
	barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
	barrier.dstStageMask = VK_PIPELINE_STAGE_NONE;
	barrier.dstAccessMask = VK_ACCESS_NONE;
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR;

	if (info.conversion_queue.family_index != info.encode_queue.family_index)
	{
		barrier.srcQueueFamilyIndex = info.conversion_queue.family_index;
		barrier.dstQueueFamilyIndex = info.encode_queue.family_index;
	}

	deps.imageMemoryBarrierCount = 1;
	deps.pImageMemoryBarriers = &barrier;
	VK_CALL(vkCmdPipelineBarrier2(cmd, &deps));
}

bool Encoder::Impl::submit_conversion(const FrameInfo &input, Frame &frame)
{
	VkCommandBuffer cmd = frame.convert_cmd;
	VK_CALL(vkResetCommandBuffer(cmd, 0));
	VkCommandBufferBeginInfo begin_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	VK_CALL(vkBeginCommandBuffer(cmd, &begin_info));

	transition_conversion_dst_images(cmd);

	ConversionRegisters params = {};
	params.width = input.width;
	params.height = input.height;

	const float dither_strength = info.profile == Profile::H265_Main10 ? 1.0f / 1023.0f : 1.0f / 255.0f;
	params.dither_strength = dither_strength;

	VK_CALL(vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, conversion_pipeline.pipeline));
	VK_CALL(vkCmdPushConstants(cmd, conversion_pipeline.layout,
							   VK_SHADER_STAGE_COMPUTE_BIT,
							   0, sizeof(params), &params));

	VkDescriptorImageInfo image_info[3] = {};
	image_info[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	image_info[0].imageView = input.view;
	image_info[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	image_info[1].imageView = dpb.luma.view;
	image_info[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	image_info[2].imageView = dpb.chroma.view;

	VkWriteDescriptorSet writes[3] = {};
	for (uint32_t i = 0; i < 3; i++)
	{
		writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].descriptorType = i != 0 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
		writes[i].dstBinding = i;
		writes[i].descriptorCount = 1;
		writes[i].pImageInfo = &image_info[i];
	}

	VK_CALL(vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
	                                  conversion_pipeline.layout, 0,
	                                  3, writes));

	// This should always align, but be pedantic.
	// Map one workgroup to one macroblock.
	uint32_t wg_x = (caps.get_aligned_width(info.width) + 15) / 16;
	uint32_t wg_y = (caps.get_aligned_height(info.height) + 15) / 16;
	VK_CALL(vkCmdDispatch(cmd, wg_x, wg_y, 1));

	copy_to_ycbcr(cmd, frame);

	if (VK_CALL(vkEndCommandBuffer(cmd)) != VK_SUCCESS)
		return false;

	VkSubmitInfo2 submit_info = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
	VkCommandBufferSubmitInfo cmd_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
	VkSemaphoreSubmitInfo signal = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };

	cmd_info.commandBuffer = cmd;
	signal.semaphore = compute_timeline.timeline;
	signal.value = ++compute_timeline.value;
	signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

	submit_info.pCommandBufferInfos = &cmd_info;
	submit_info.commandBufferInfoCount = 1;
	submit_info.pSignalSemaphoreInfos = &signal;
	submit_info.signalSemaphoreInfoCount = 1;

	return VK_CALL(vkQueueSubmit2(info.conversion_queue.queue, 1, &submit_info, VK_NULL_HANDLE)) == VK_SUCCESS;
}

void Encoder::Impl::record_host_barrier(VkCommandBuffer cmd, Frame &frame)
{
	VkDependencyInfo deps = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
	VkBufferMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
	deps.bufferMemoryBarrierCount = 1;
	deps.pBufferMemoryBarriers = &barrier;

	barrier.buffer = frame.payload.buffer;
	barrier.srcStageMask = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR;
	barrier.srcAccessMask = VK_ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR;
	barrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
	barrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
	barrier.size = VK_WHOLE_SIZE;

	VK_CALL(vkCmdPipelineBarrier2(cmd, &deps));
}

struct H265EncodeInfo
{
	VkVideoEncodeH265PictureInfoKHR h265_src_info = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_PICTURE_INFO_KHR };
	VkVideoEncodeH265NaluSliceSegmentInfoKHR slice = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_NALU_SLICE_SEGMENT_INFO_KHR };
	StdVideoEncodeH265SliceSegmentHeader slice_header = {};
	StdVideoEncodeH265PictureInfo pic = {};
	StdVideoEncodeH265ReferenceListsInfo ref_lists = {};

	VkVideoEncodeH265DpbSlotInfoKHR h265_reconstructed_dpb_slot = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_DPB_SLOT_INFO_KHR };
	StdVideoEncodeH265ReferenceInfo h265_reconstructed_ref = {};

	VkVideoEncodeH265DpbSlotInfoKHR h265_prev_ref_slot = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_DPB_SLOT_INFO_KHR };
	StdVideoEncodeH265ReferenceInfo h265_prev_ref = {};
	StdVideoH265ShortTermRefPicSet short_term_ref_pic_set = {};

	VkVideoEncodeH265GopRemainingFrameInfoKHR gop_remaining =
		{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_GOP_REMAINING_FRAME_INFO_KHR };

	void setup(const VideoEncoderCaps &caps, const VideoSessionParameters &params, RateControl &rate,
	           VkVideoBeginCodingInfoKHR &begin_info, VkVideoEncodeInfoKHR &info,
	           VkVideoEncodeTuningModeKHR tuning);
};

struct H264EncodeInfo
{
	VkVideoEncodeH264PictureInfoKHR h264_src_info = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PICTURE_INFO_KHR };
	VkVideoEncodeH264NaluSliceInfoKHR slice = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_NALU_SLICE_INFO_KHR };
	StdVideoEncodeH264SliceHeader slice_header = {};
	StdVideoEncodeH264PictureInfo pic = {};
	StdVideoEncodeH264ReferenceListsInfo ref_lists = {};

	VkVideoEncodeH264DpbSlotInfoKHR h264_reconstructed_dpb_slot = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_DPB_SLOT_INFO_KHR };
	StdVideoEncodeH264ReferenceInfo h264_reconstructed_ref = {};

	VkVideoEncodeH264DpbSlotInfoKHR h264_prev_ref_slot = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_DPB_SLOT_INFO_KHR };
	StdVideoEncodeH264ReferenceInfo h264_prev_ref = {};

	VkVideoEncodeH264GopRemainingFrameInfoKHR gop_remaining =
		{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_GOP_REMAINING_FRAME_INFO_KHR };

	void setup(const VideoEncoderCaps &caps, const VideoSessionParameters &params, RateControl &rate,
	           VkVideoBeginCodingInfoKHR &begin_info, VkVideoEncodeInfoKHR &info);
};

void H264EncodeInfo::setup(
		const VideoEncoderCaps &caps,
		const VideoSessionParameters &params, RateControl &rate,
		VkVideoBeginCodingInfoKHR &begin_info,
		VkVideoEncodeInfoKHR &info)
{
	bool is_idr = rate.gop_frame_index == 0;

	for (uint32_t i = 0; i < STD_VIDEO_H264_MAX_NUM_LIST_REF; i++)
	{
		ref_lists.RefPicList0[i] = i || is_idr ? STD_VIDEO_H264_NO_REFERENCE_PICTURE : ((rate.gop_frame_index - 1) & 1);
		ref_lists.RefPicList1[i] = STD_VIDEO_H264_NO_REFERENCE_PICTURE;
	}

	pic.flags.IdrPicFlag = is_idr ? 1 : 0;
	pic.flags.is_reference = 1;
	if (is_idr)
		pic.idr_pic_id = rate.idr_pic_id++;
	pic.pRefLists = &ref_lists;

	slice.pStdSliceHeader = &slice_header;
	slice_header.cabac_init_idc = STD_VIDEO_H264_CABAC_INIT_IDC_0;
	slice_header.slice_type = is_idr ? STD_VIDEO_H264_SLICE_TYPE_I : STD_VIDEO_H264_SLICE_TYPE_P;
	if (rate.rate_info.rateControlMode == VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR)
	{
		slice.constantQp = rate.info.constant_qp;
		if (slice.constantQp < caps.h264.caps.minQp)
			slice.constantQp = caps.h264.caps.minQp;
		else if (slice.constantQp > caps.h264.caps.maxQp)
			slice.constantQp = caps.h264.caps.maxQp;
	}

	h264_src_info.naluSliceEntryCount = 1;
	h264_src_info.pNaluSliceEntries = &slice;
	h264_src_info.pStdPictureInfo = &pic;
	h264_src_info.pNext = info.pNext;
	info.pNext = &h264_src_info;

	const uint32_t FrameNumMask =
			((1u << (params.h264.sps.log2_max_frame_num_minus4 + 4)) - 1u);

	// Apply stride 2 later.
	const uint32_t PicOrderCntMask =
			((1u << (params.h264.sps.log2_max_pic_order_cnt_lsb_minus4 + 3)) - 1u);

	h264_reconstructed_dpb_slot.pStdReferenceInfo = &h264_reconstructed_ref;
	h264_reconstructed_ref.FrameNum = rate.gop_frame_index & FrameNumMask;
	// Using a POC stride of 2 seems to be common for progressive scan H.264.
	// Seems to work okay with stride of 1 too, but there might be some nasal demons out there.
	h264_reconstructed_ref.PicOrderCnt = int(rate.gop_frame_index & PicOrderCntMask) * 2;
	const_cast<VkVideoReferenceSlotInfoKHR *>(info.pSetupReferenceSlot)->pNext = &h264_reconstructed_dpb_slot;

	pic.frame_num = h264_reconstructed_ref.FrameNum;
	pic.PicOrderCnt = h264_reconstructed_ref.PicOrderCnt;

	auto pict_type = is_idr ? STD_VIDEO_H264_PICTURE_TYPE_IDR : STD_VIDEO_H264_PICTURE_TYPE_P;
	h264_reconstructed_ref.primary_pic_type = pict_type;
	pic.primary_pic_type = pict_type;

	if (!is_idr)
	{
		assert(info.referenceSlotCount == 1);
		const_cast<VkVideoReferenceSlotInfoKHR &>(info.pReferenceSlots[0]).pNext = &h264_prev_ref_slot;
		h264_prev_ref_slot.pStdReferenceInfo = &h264_prev_ref;

		h264_prev_ref.FrameNum = (rate.gop_frame_index - 1) & FrameNumMask;
		// Using a POC stride of 2 seems to be common for progressive scan H.264.
		// Seems to work okay with stride of 1 too, but there might be some nasal demons out there.
		h264_prev_ref.PicOrderCnt = int((rate.gop_frame_index - 1) & PicOrderCntMask) * 2;

		// Does this matter?
		if (rate.gop_frame_index == 1)
			h264_prev_ref.primary_pic_type = STD_VIDEO_H264_PICTURE_TYPE_IDR;
		else
			h264_prev_ref.primary_pic_type = STD_VIDEO_H264_PICTURE_TYPE_P;
	}

	if (rate.info.gop_frames != UINT32_MAX)
	{
		// This struct may be required by implementation. Providing it does not hurt.
		gop_remaining.useGopRemainingFrames = VK_TRUE;
		gop_remaining.gopRemainingB = 0; // TODO
		gop_remaining.gopRemainingI = is_idr ? 1 : 0;
		gop_remaining.gopRemainingP = rate.info.gop_frames - rate.gop_frame_index - gop_remaining.gopRemainingI;
		gop_remaining.pNext = begin_info.pNext;
		begin_info.pNext = &gop_remaining;
	}
}

void H265EncodeInfo::setup(
		const VideoEncoderCaps &caps,
		const VideoSessionParameters &params, RateControl &rate,
		VkVideoBeginCodingInfoKHR &begin_info,
		VkVideoEncodeInfoKHR &info,
		VkVideoEncodeTuningModeKHR tuning)
{
	// Mostly based on nvpro sample.
	// I don't really know what I'm doing here, but seems to work on NVIDIA at least.

	bool is_idr = rate.gop_frame_index == 0;

	for (uint32_t i = 0; i < STD_VIDEO_H265_MAX_NUM_LIST_REF; i++)
	{
		ref_lists.RefPicList0[i] = i || is_idr ? STD_VIDEO_H265_NO_REFERENCE_PICTURE : ((rate.gop_frame_index - 1) & 1);
		ref_lists.RefPicList1[i] = STD_VIDEO_H265_NO_REFERENCE_PICTURE;
	}

	pic.flags.IrapPicFlag = is_idr ? 1 : 0;
	pic.flags.is_reference = 1;
	pic.pRefLists = &ref_lists;

	short_term_ref_pic_set.num_negative_pics = 1;
	short_term_ref_pic_set.use_delta_flag = 1;
	short_term_ref_pic_set.used_by_curr_pic_s0_flag = 1;
	short_term_ref_pic_set.used_by_curr_pic_flag = 1;
	pic.pShortTermRefPicSet = &short_term_ref_pic_set;

	if (is_idr)
		rate.idr_pic_id++;

	slice.pStdSliceSegmentHeader = &slice_header;
	slice_header.slice_type = is_idr ? STD_VIDEO_H265_SLICE_TYPE_I : STD_VIDEO_H265_SLICE_TYPE_P;
	slice_header.MaxNumMergeCand = 5;
	slice_header.flags.first_slice_segment_in_pic_flag = 1;

	if (tuning != VK_VIDEO_ENCODE_TUNING_MODE_LOW_LATENCY_KHR &&
	    tuning != VK_VIDEO_ENCODE_TUNING_MODE_ULTRA_LOW_LATENCY_KHR)
	{
		if ((caps.h265.caps.stdSyntaxFlags &
		     VK_VIDEO_ENCODE_H265_STD_SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG_SET_BIT_KHR) != 0)
		{
			slice_header.flags.slice_sao_chroma_flag = 1;
			slice_header.flags.slice_sao_luma_flag = 1;
		}
	}

	if (rate.rate_info.rateControlMode == VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR)
	{
		slice.constantQp = rate.info.constant_qp;
		if (slice.constantQp < caps.h265.caps.minQp)
			slice.constantQp = caps.h265.caps.minQp;
		else if (slice.constantQp > caps.h265.caps.maxQp)
			slice.constantQp = caps.h265.caps.maxQp;
	}

	h265_src_info.naluSliceSegmentEntryCount = 1;
	h265_src_info.pNaluSliceSegmentEntries = &slice;
	h265_src_info.pStdPictureInfo = &pic;
	h265_src_info.pNext = info.pNext;
	info.pNext = &h265_src_info;

	const uint32_t PicOrderCntMask =
		((1u << (params.h265.sps.log2_max_pic_order_cnt_lsb_minus4 + 4)) - 1u);

	h265_reconstructed_dpb_slot.pStdReferenceInfo = &h265_reconstructed_ref;
	h265_reconstructed_ref.PicOrderCntVal = int(rate.gop_frame_index & PicOrderCntMask);
	const_cast<VkVideoReferenceSlotInfoKHR *>(info.pSetupReferenceSlot)->pNext = &h265_reconstructed_dpb_slot;

	pic.PicOrderCntVal = h265_reconstructed_ref.PicOrderCntVal;

	auto pict_type = is_idr ? STD_VIDEO_H265_PICTURE_TYPE_IDR : STD_VIDEO_H265_PICTURE_TYPE_P;
	h265_reconstructed_ref.pic_type = pict_type;
	pic.pic_type = pict_type;

	if (!is_idr)
	{
		assert(info.referenceSlotCount == 1);
		const_cast<VkVideoReferenceSlotInfoKHR &>(info.pReferenceSlots[0]).pNext = &h265_prev_ref_slot;
		h265_prev_ref_slot.pStdReferenceInfo = &h265_prev_ref;

		h265_prev_ref.PicOrderCntVal = int((rate.gop_frame_index - 1) & PicOrderCntMask);

		// Does this matter?
		if (rate.gop_frame_index == 1)
			h265_prev_ref.pic_type = STD_VIDEO_H265_PICTURE_TYPE_IDR;
		else
			h265_prev_ref.pic_type = STD_VIDEO_H265_PICTURE_TYPE_P;
	}

	if (rate.info.gop_frames != UINT32_MAX)
	{
		// This struct may be required by implementation. Providing it does not hurt.
		gop_remaining.useGopRemainingFrames = VK_TRUE;
		gop_remaining.gopRemainingB = 0; // TODO
		gop_remaining.gopRemainingI = is_idr ? 1 : 0;
		gop_remaining.gopRemainingP = rate.info.gop_frames - rate.gop_frame_index - gop_remaining.gopRemainingI;
		gop_remaining.pNext = begin_info.pNext;
		begin_info.pNext = &gop_remaining;
	}
}

bool Encoder::Impl::record_and_submit_encode(VkCommandBuffer cmd, Frame &frame, bool &is_idr)
{
	VkVideoBeginCodingInfoKHR video_coding_info = { VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR };
	VkVideoEndCodingInfoKHR end_coding_info = { VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR };
	video_coding_info.videoSession = session.session;
	video_coding_info.videoSessionParameters = session_params.params;
	video_coding_info.pNext = &rate.rate_info;

	if ((rate.info.gop_frames != UINT32_MAX && rate.gop_frame_index == rate.info.gop_frames) || is_idr)
		rate.gop_frame_index = 0;
	is_idr = rate.gop_frame_index == 0;

	const VkExtent2D coded_extent = { caps.get_aligned_width(info.width), caps.get_aligned_height(info.height) };
	uint32_t dpb_index_reconstructed = rate.gop_frame_index & 1;
	uint32_t dpb_index_reference = 1 - dpb_index_reconstructed;

	// We have two reference slots.
	// One where we will write the new reconstructed image,
	// and one for previous frame (if encoding a P frame).
	VkVideoPictureResourceInfoKHR reconstructed_slot_pic = { VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR };
	VkVideoPictureResourceInfoKHR reference_slot_pic = { VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR };
	VkVideoReferenceSlotInfoKHR init_slots[2] = {};

	H264EncodeInfo h264;
	H265EncodeInfo h265;

	if (dpb.array_dpb.image)
	{
		reconstructed_slot_pic.imageViewBinding = dpb.array_dpb.view;
		reference_slot_pic.imageViewBinding = dpb.array_dpb.view;
		reconstructed_slot_pic.baseArrayLayer = dpb_index_reconstructed;
		reference_slot_pic.baseArrayLayer = dpb_index_reference;
	}
	else
	{
		reconstructed_slot_pic.imageViewBinding = dpb.dpb[dpb_index_reconstructed].view;
		reference_slot_pic.imageViewBinding = dpb.dpb[dpb_index_reference].view;
	}

	reconstructed_slot_pic.codedExtent = coded_extent;
	reference_slot_pic.codedExtent = coded_extent;

	init_slots[0].sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
	init_slots[0].slotIndex = -1;
	init_slots[0].pPictureResource = &reconstructed_slot_pic;
	init_slots[1].sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
	init_slots[1].slotIndex = int(dpb_index_reference);
	init_slots[1].pPictureResource = &reference_slot_pic;

	video_coding_info.referenceSlotCount = is_idr ? 1 : 2;
	video_coding_info.pReferenceSlots = init_slots;

	VkVideoEncodeInfoKHR encode_info = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_INFO_KHR };
	encode_info.srcPictureResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
	encode_info.srcPictureResource.codedExtent = coded_extent;
	encode_info.srcPictureResource.imageViewBinding = frame.image_ycbcr.view;

	// Write reconstructed image to slot and initialize it.
	VkVideoReferenceSlotInfoKHR reconstructed_setup_slot = { VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR };
	VkVideoReferenceSlotInfoKHR prev_ref_slot = { VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR };

	reconstructed_setup_slot.slotIndex = int32_t(dpb_index_reconstructed);
	reconstructed_setup_slot.pPictureResource = &reconstructed_slot_pic;
	encode_info.pSetupReferenceSlot = &reconstructed_setup_slot;

	// If not IDR, pass down the reference frame.
	if (!is_idr)
	{
		prev_ref_slot.pPictureResource = &reference_slot_pic;
		prev_ref_slot.slotIndex = int(dpb_index_reference);
		encode_info.pReferenceSlots = &prev_ref_slot;
		encode_info.referenceSlotCount = 1;
	}

	encode_info.dstBuffer = frame.payload.buffer;
	encode_info.dstBufferOffset = 0;
	encode_info.dstBufferRange = MaxPayloadSize;

	switch (info.profile)
	{
	case Profile::H264_High:
		h264.setup(caps, session_params, rate, video_coding_info, encode_info);
		break;

	case Profile::H265_Main:
	case Profile::H265_Main10:
		h265.setup(caps, session_params, rate, video_coding_info, encode_info, info.hints.tuning);
		break;

	default:
		return false;
	}

	auto query_index = uint32_t(&frame - frame_pool);
	VK_CALL(vkCmdResetQueryPool(cmd, query_pool, query_index, 1));

	if (query_pool_timestamp)
	{
		VK_CALL(vkCmdResetQueryPool(cmd, query_pool_timestamp, query_index * 2 + 0, 2));
		VK_CALL(vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR,
		                             query_pool_timestamp, query_index * 2 + 0));
	}

	VK_CALL(vkCmdBeginVideoCodingKHR(cmd, &video_coding_info));
	VK_CALL(vkCmdBeginQuery(cmd, query_pool, query_index, 0));
	VK_CALL(vkCmdEncodeVideoKHR(cmd, &encode_info));
	VK_CALL(vkCmdEndQuery(cmd, query_pool, query_index));
	VK_CALL(vkCmdEndVideoCodingKHR(cmd, &end_coding_info));

	if (query_pool_timestamp)
	{
		VK_CALL(vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR,
		                             query_pool_timestamp, query_index * 2 + 1));
	}

	rate.gop_frame_index++;
	rate.frame_index++;

	record_host_barrier(cmd, frame);
	if (VK_CALL(vkEndCommandBuffer(cmd)) != VK_SUCCESS)
		return false;

	// Calling submit here is a workaround for an NVIDIA driver bug where the H.264 reference lists
	// are accessed in vkQueueSubmit instead of vkCmdEncodeVideoKHR.
	// This avoids a random GPU hang where bogus resources are accessed on GPU.
	return submit_encode_command_buffer(cmd);
}

void Encoder::Impl::record_dpb_barrier(VkCommandBuffer cmd)
{
	VkDependencyInfo deps = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
	VkImageMemoryBarrier2 barriers[DPBSize];
	deps.pImageMemoryBarriers = barriers;

	if (dpb.array_dpb.image)
	{
		deps.imageMemoryBarrierCount = 1;

		auto &barrier = barriers[0];
		barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
		barrier.image = dpb.array_dpb.image;
		barrier.srcStageMask = VK_PIPELINE_STAGE_NONE;
		barrier.srcAccessMask = VK_ACCESS_NONE;
		barrier.dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR;
		barrier.dstAccessMask = VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR |
		                        VK_ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR;
		barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		barrier.newLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR;
		barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS };
	}
	else
	{
		deps.imageMemoryBarrierCount = DPBSize;

		for (uint32_t i = 0; i < DPBSize; i++)
		{
			auto &barrier = barriers[i];
			barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
			barrier.image = dpb.dpb[i].image;
			barrier.srcStageMask = VK_PIPELINE_STAGE_NONE;
			barrier.srcAccessMask = VK_ACCESS_NONE;
			barrier.dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR;
			barrier.dstAccessMask = VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR |
			                        VK_ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR;
			barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			barrier.newLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR;
			barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		}
	}

	VK_CALL(vkCmdPipelineBarrier2(cmd, &deps));
}

void Encoder::Impl::record_acquire_barrier(VkCommandBuffer cmd, const Frame &frame)
{
	VkDependencyInfo deps = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
	VkImageMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
	deps.imageMemoryBarrierCount = 1;
	deps.pImageMemoryBarriers = &barrier;

	barrier.image = frame.image_ycbcr.image;
	barrier.srcStageMask = VK_PIPELINE_STAGE_NONE;
	barrier.srcAccessMask = VK_ACCESS_NONE;
	barrier.dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR;
	barrier.dstAccessMask = VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR;
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR;
	barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
	barrier.srcQueueFamilyIndex = info.conversion_queue.family_index;
	barrier.dstQueueFamilyIndex = info.encode_queue.family_index;

	VK_CALL(vkCmdPipelineBarrier2(cmd, &deps));
}

void Encoder::Impl::record_rate_control(VkCommandBuffer cmd)
{
	VkVideoBeginCodingInfoKHR video_coding_info = { VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR };
	VkVideoEndCodingInfoKHR end_coding_info = { VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR };
	video_coding_info.videoSession = session.session;
	video_coding_info.videoSessionParameters = session_params.params;

	VK_CALL(vkCmdBeginVideoCodingKHR(cmd, &video_coding_info));
	VK_CALL(vkCmdControlVideoCodingKHR(cmd, &rate.ctrl_info));
	VK_CALL(vkCmdEndVideoCodingKHR(cmd, &end_coding_info));
}

bool Encoder::Impl::submit_encode(Frame &frame, bool &is_idr)
{
	VkCommandBuffer cmd = frame.encode_cmd;
	VK_CALL(vkResetCommandBuffer(cmd, 0));
	VkCommandBufferBeginInfo begin_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	VK_CALL(vkBeginCommandBuffer(cmd, &begin_info));

	if (!dpb.dpb_inited)
	{
		record_dpb_barrier(cmd);
		dpb.dpb_inited = true;
	}

	if (rate.needs_reset)
	{
		record_rate_control(cmd);
		rate.needs_reset = false;
	}

	if (info.conversion_queue.family_index != info.encode_queue.family_index)
		record_acquire_barrier(cmd, frame);

	return record_and_submit_encode(cmd, frame, is_idr);
}

bool Encoder::Impl::submit_encode_command_buffer(VkCommandBuffer cmd)
{
	VkSubmitInfo2 submit_info = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
	VkCommandBufferSubmitInfo cmd_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
	VkSemaphoreSubmitInfo signal = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
	VkSemaphoreSubmitInfo waits[2] = {};

	cmd_info.commandBuffer = cmd;

	waits[0] = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
	waits[1] = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };

	waits[0].semaphore = compute_timeline.timeline;
	waits[0].value = compute_timeline.value;
	waits[0].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

	// Wait for previous frame's encode to complete.
	waits[1].semaphore = encode_timeline.timeline;
	waits[1].value = encode_timeline.value;
	waits[1].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

	signal.semaphore = encode_timeline.timeline;
	signal.value = ++encode_timeline.value;
	signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

	submit_info.pCommandBufferInfos = &cmd_info;
	submit_info.commandBufferInfoCount = 1;
	submit_info.pWaitSemaphoreInfos = waits;
	submit_info.waitSemaphoreInfoCount = waits[1].value != 0 ? 2 : 1;
	submit_info.pSignalSemaphoreInfos = &signal;
	submit_info.signalSemaphoreInfoCount = 1;

	return VK_CALL(vkQueueSubmit2(info.encode_queue.queue, 1, &submit_info, VK_NULL_HANDLE)) == VK_SUCCESS;
}

Result Encoder::Impl::send_frame(const FrameInfo &input)
{
	// Scaling not supported.
	if (input.width != info.width || input.height != info.height)
		return Result::Error;

	uint32_t frame_index = allocate_frame_pool_index();

	// If application hasn't been freeing or consuming EncodedFrame objects, we'll exhaust the pool.
	if (frame_index == UINT32_MAX)
		return Result::NotReady;

	EncodedFrame output;
	auto &frame_impl = output.get_impl();
	frame_impl.frame_pool_index = frame_index;
	frame_impl.encoder = this;

	// If application did not write for EncodedFrame, we'll wait here.
	wait_frame_pool_index(frame_index, UINT64_MAX);

	auto &frame = frame_pool[frame_index];

	// Convert RGB to YCbCr.
	if (!submit_conversion(input, frame))
		return Result::Error;
	frame.compute = compute_timeline.value;

	// Submit encode to Vulkan.
	bool is_idr = input.force_idr;
	if (!submit_encode(frame, is_idr))
		return Result::Error;
	frame.encode = encode_timeline.value;

	// Push a new encoded frame.
	frame_impl.payload = frame.payload.memory.mapped;
	frame_impl.is_idr = is_idr;
	frame_impl.pts = input.pts;
	// TODO: For B-frames, this will have to change.
	frame_impl.dts = input.pts;

	encoded_queue.push(std::move(output));
	return Result::Success;
}

Result Encoder::Impl::send_eof()
{
	// This will be relevant for B-frames.
	return Result::Success;
}

Result Encoder::Impl::receive_encoded_frame(EncodedFrame &frame)
{
	if (encoded_queue.empty())
		return Result::NotReady;

	frame = std::move(encoded_queue.front());
	encoded_queue.pop();
	return Result::Success;
}

void Encoder::Impl::destroy_dpb()
{
	for (auto &img : dpb.dpb)
		destroy_image(img);
	destroy_image(dpb.array_dpb);
	destroy_image(dpb.luma);
	destroy_image(dpb.chroma);
}

bool Encoder::Impl::init_frame_resource(Frame &frame)
{
	if (!create_buffer(frame.payload, MaxPayloadSize, VK_BUFFER_USAGE_VIDEO_ENCODE_DST_BIT_KHR,
	                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
	                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
	                   VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
	                   &profile.profile_list))
	{
		return false;
	}

	uint32_t aligned_width = caps.get_aligned_width(info.width);
	uint32_t aligned_height = caps.get_aligned_height(info.height);

	if (!create_image(frame.image_ycbcr, aligned_width, aligned_height, 1, profile.input.format,
	                  VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR,
					  &profile.profile_list))
		return false;

	VkCommandBufferAllocateInfo alloc_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	alloc_info.commandBufferCount = 1;

	alloc_info.commandPool = encode_cmd_pool;
	if (VK_CALL(vkAllocateCommandBuffers(info.device, &alloc_info, &frame.encode_cmd)) != VK_SUCCESS)
		return false;

	alloc_info.commandPool = convert_cmd_pool;
	if (VK_CALL(vkAllocateCommandBuffers(info.device, &alloc_info, &frame.convert_cmd)) != VK_SUCCESS)
		return false;

	return true;
}

bool Encoder::Impl::init_command_pools()
{
	VkCommandPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
	pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

	pool_info.queueFamilyIndex = info.conversion_queue.family_index;
	if (VK_CALL(vkCreateCommandPool(info.device, &pool_info, nullptr, &convert_cmd_pool)) != VK_SUCCESS)
		return false;

	pool_info.queueFamilyIndex = info.encode_queue.family_index;
	if (VK_CALL(vkCreateCommandPool(info.device, &pool_info, nullptr, &encode_cmd_pool)) != VK_SUCCESS)
		return false;

	return true;
}

bool Encoder::Impl::init_query_pool()
{
	VkQueryPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
	VkQueryPoolVideoEncodeFeedbackCreateInfoKHR feedback_pool_info =
		{ VK_STRUCTURE_TYPE_QUERY_POOL_VIDEO_ENCODE_FEEDBACK_CREATE_INFO_KHR };
	feedback_pool_info.encodeFeedbackFlags = VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BUFFER_OFFSET_BIT_KHR |
	                                         VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BYTES_WRITTEN_BIT_KHR;
	feedback_pool_info.pNext = &profile.profile_info;
	pool_info.queryType = VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR;
	pool_info.queryCount = FramePoolSize;
	pool_info.pNext = &feedback_pool_info;
	if (VK_CALL(vkCreateQueryPool(info.device, &pool_info, nullptr, &query_pool)) != VK_SUCCESS)
		return false;


	uint32_t queue_count;
	VK_CALL(vkGetPhysicalDeviceQueueFamilyProperties(info.gpu, &queue_count, nullptr));
	std::vector<VkQueueFamilyProperties> props(queue_count);
	VK_CALL(vkGetPhysicalDeviceQueueFamilyProperties(info.gpu, &queue_count, props.data()));

	if (info.encode_queue.family_index < props.size())
		encode_timestamp_bits = props[info.encode_queue.family_index].timestampValidBits;

	if (encode_timestamp_bits)
	{
		pool_info.queryCount = FramePoolSize * 2;
		pool_info.pNext = nullptr;
		pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
		if (VK_CALL(vkCreateQueryPool(info.device, &pool_info, nullptr, &query_pool_timestamp)) != VK_SUCCESS)
			return false;
	}

	return true;
}

bool Encoder::Impl::init_encoder(const EncoderCreateInfo &info_)
{
	info = info_;
	if (!init_func_table())
		return false;

	VK_CALL(vkGetPhysicalDeviceMemoryProperties(info.gpu, &mem_props));
	vk12_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
	VkPhysicalDeviceProperties2 props2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &vk12_props };
	VK_CALL(vkGetPhysicalDeviceProperties2(info.gpu, &props2));
	vk_props = props2.properties;

	if (!create_timeline_semaphore(compute_timeline))
		return false;
	if (!create_timeline_semaphore(encode_timeline))
		return false;

	if (!profile.setup(*this, info.profile))
		return false;
	if (!caps.setup(*this))
		return false;
	if (!session.init(*this))
		return false;
	if (!session_params.init(*this))
		return false;
	if (!rate.init(*this))
		return false;

	if (!init_command_pools())
		return false;
	if (!init_query_pool())
		return false;

	if (!init_frame_resources())
		return false;
	if (!init_dpb_resources())
		return false;

	if (!init_pipelines())
		return false;

	return true;
}

template <typename... Ts>
void Encoder::Impl::log(Severity severity, const char *fmt, Ts&&... ts)
{
	if (cb)
	{
		char buffer[4096];
		snprintf(buffer, sizeof(buffer), fmt, std::forward<Ts>(ts)...);
		cb->log(severity, buffer);
	}
}

bool Encoder::set_rate_control_info(const RateControlInfo &info)
{
	impl->rate.info = info;
	impl->rate.needs_reset = true;
	return impl->rate.init(*impl);
}

void Encoder::set_log_callback(LogCallback *cb)
{
	impl->cb = cb;
}

Result Encoder::init_encoder(const EncoderCreateInfo &info)
{
	return impl->init_encoder(info) ? Result::Success : Result::Error;
}

Encoder::Encoder()
{
	impl.reset(new Impl);
}

Encoder::~Encoder()
{
}

Result Encoder::send_frame(const FrameInfo &info)
{
	return impl->send_frame(info);
}

Result Encoder::send_eof()
{
	return impl->send_eof();
}

Result Encoder::receive_encoded_frame(EncodedFrame &frame)
{
	return impl->receive_encoded_frame(frame);
}

VkAccessFlags2 Encoder::get_conversion_dst_access() const
{
	return VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
}

VkPipelineStageFlags2 Encoder::get_conversion_dst_stage() const
{
	return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
}

VkImageLayout Encoder::get_conversion_image_layout() const
{
	return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

const void *Encoder::get_encoded_parameters() const
{
	return impl->session_params.encoded_parameters.data();
}

size_t Encoder::get_encoded_parameters_size() const
{
	return impl->session_params.encoded_parameters.size();
}

// Only consider formats we understand how to deal with.
// Every known GPU is 2-plane chroma, so don't bother with 3-plane chroma encode unless we are forced to care.
static bool get_planar_formats(VkFormat ycbcr_format,
                               VkFormat &luma_format, VkFormat &chroma_format)
{
	switch (ycbcr_format)
	{
	case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:
	case VK_FORMAT_G8_B8R8_2PLANE_422_UNORM:
	case VK_FORMAT_G8_B8R8_2PLANE_444_UNORM:
		luma_format = VK_FORMAT_R8_UNORM;
		chroma_format = VK_FORMAT_R8G8_UNORM;
		break;

	case VK_FORMAT_G16_B16R16_2PLANE_420_UNORM:
	case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16:
	case VK_FORMAT_G16_B16R16_2PLANE_422_UNORM:
	case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16:
	case VK_FORMAT_G16_B16R16_2PLANE_444_UNORM:
	case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16:
		luma_format = VK_FORMAT_R16_UNORM;
		chroma_format = VK_FORMAT_R16G16_UNORM;
		break;

	default:
		return false;
	}

	return true;
}

static bool get_planar_subsample(VkFormat ycbcr_format, uint32_t *subsample_log2)
{
	switch (ycbcr_format)
	{
	case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:
	case VK_FORMAT_G16_B16R16_2PLANE_420_UNORM:
	case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16:
		subsample_log2[0] = 1;
		subsample_log2[1] = 1;
		break;

	case VK_FORMAT_G8_B8R8_2PLANE_422_UNORM:
	case VK_FORMAT_G16_B16R16_2PLANE_422_UNORM:
	case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16:
		subsample_log2[0] = 1;
		subsample_log2[1] = 0;
		break;

	case VK_FORMAT_G8_B8R8_2PLANE_444_UNORM:
	case VK_FORMAT_G16_B16R16_2PLANE_444_UNORM:
	case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16:
		subsample_log2[0] = 0;
		subsample_log2[1] = 0;
		break;

	default:
		return false;
	}

	return true;
}

VideoProfile::Format VideoProfile::get_format_info(Encoder::Impl &impl, VkImageUsageFlags usage)
{
	auto &table = impl.table;
	Format format;

	// Query supported formats.
	VkPhysicalDeviceVideoFormatInfoKHR format_info = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_FORMAT_INFO_KHR };
	format_info.imageUsage = usage;
	format_info.pNext = &profile_list;

	uint32_t count = 0;
	if (VK_CALL(vkGetPhysicalDeviceVideoFormatPropertiesKHR(impl.info.gpu, &format_info, &count, nullptr)) != VK_SUCCESS)
		return {};

	if (count == 0)
		return {};

	std::vector<VkVideoFormatPropertiesKHR> props(count);
	for (auto &p : props)
		p.sType = VK_STRUCTURE_TYPE_VIDEO_FORMAT_PROPERTIES_KHR;

	if (VK_CALL(vkGetPhysicalDeviceVideoFormatPropertiesKHR(impl.info.gpu, &format_info, &count, props.data())) != VK_SUCCESS)
		return {};

	for (auto &prop : props)
	{
		VkFormat fmt = prop.format;

		if (!get_planar_formats(fmt, format.luma_format, format.chroma_format))
			continue;
		if (!get_planar_subsample(fmt, format.subsample_log2))
			continue;

		// Sanity check the format before we accept it.
		if (prop.imageType != VK_IMAGE_TYPE_2D)
			continue;

		if ((prop.imageUsageFlags & usage) != usage)
			continue;

		// If imageCreateFlags has MUTABLE and EXTENDED we could avoid the intermediate copy for YCbCr conversion,
		// but neither NV nor RADV advertise it, so take the "safe" path with copy.

		if (prop.componentMapping.r != VK_COMPONENT_SWIZZLE_IDENTITY ||
		    prop.componentMapping.g != VK_COMPONENT_SWIZZLE_IDENTITY ||
		    prop.componentMapping.b != VK_COMPONENT_SWIZZLE_IDENTITY)
		{
			// NV12 vs NV21? No GPU I know of cares ...
			continue;
		}

		{
			VkImageFormatProperties2 props2 = { VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2 };
			VkPhysicalDeviceImageFormatInfo2 fmt_info = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2 };
			fmt_info.format = fmt;
			fmt_info.tiling = VK_IMAGE_TILING_OPTIMAL;
			fmt_info.type = VK_IMAGE_TYPE_2D;
			fmt_info.usage = usage;
			fmt_info.pNext = &profile_list;
			if (VK_CALL(vkGetPhysicalDeviceImageFormatProperties2(impl.info.gpu, &fmt_info, &props2)) != VK_SUCCESS)
				continue;

			if (props2.imageFormatProperties.maxExtent.width < impl.info.width ||
			    props2.imageFormatProperties.maxExtent.height < impl.info.height)
			{
				continue;
			}

			format.format_properties = props2.imageFormatProperties;
			format.format = fmt;
		}

		break;
	}

	return format;
}

bool VideoProfile::setup(Encoder::Impl &impl, Profile profile)
{
	switch (profile)
	{
	case Profile::H264_High:
		h264.profile = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PROFILE_INFO_KHR };
		profile_info.chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR;
		profile_info.chromaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
		profile_info.lumaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
		profile_info.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR;
		h264.profile.stdProfileIdc = STD_VIDEO_H264_PROFILE_IDC_HIGH;
		profile_info.pNext = &h264.profile;
		break;

	case Profile::H265_Main:
		h265.profile = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_PROFILE_INFO_KHR };
		profile_info.chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR;
		profile_info.chromaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
		profile_info.lumaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
		profile_info.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR;
		h265.profile.stdProfileIdc = STD_VIDEO_H265_PROFILE_IDC_MAIN;
		profile_info.pNext = &h265.profile;
		break;

	case Profile::H265_Main10:
		h265.profile = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_PROFILE_INFO_KHR };
		profile_info.chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR;
		profile_info.chromaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_10_BIT_KHR;
		profile_info.lumaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_10_BIT_KHR;
		profile_info.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR;
		h265.profile.stdProfileIdc = STD_VIDEO_H265_PROFILE_IDC_MAIN_10;
		profile_info.pNext = &h265.profile;
		break;

	default:
		return false;
	}

	usage_info.tuningMode = impl.info.hints.tuning;
	usage_info.videoContentHints = impl.info.hints.content;
	usage_info.videoUsageHints = impl.info.hints.usage;
	usage_info.pNext = profile_info.pNext;
	profile_info.pNext = &usage_info;

	profile_list.pProfiles = &profile_info;
	profile_list.profileCount = 1;

	dpb = get_format_info(impl, VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR);
	input = get_format_info(impl, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR);
	return input.format != VK_FORMAT_UNDEFINED && dpb.format != VK_FORMAT_UNDEFINED;
}

bool VideoEncoderCaps::setup(Encoder::Impl &impl)
{
	auto &table = impl.table;
	video_caps.pNext = &encode_caps;

	switch (impl.info.profile)
	{
	case Profile::H264_High:
		h264.caps = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_CAPABILITIES_KHR };
		encode_caps.pNext = &h264.caps;
		break;

	case Profile::H265_Main:
	case Profile::H265_Main10:
		h265.caps = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_CAPABILITIES_KHR };
		encode_caps.pNext = &h265.caps;
		break;

	default:
		return false;
	}

	if (VK_CALL(vkGetPhysicalDeviceVideoCapabilitiesKHR(impl.info.gpu, &impl.profile.profile_info, &video_caps)) != VK_SUCCESS)
		return false;

	if (impl.info.width > video_caps.maxCodedExtent.width ||
	    impl.info.height > video_caps.maxCodedExtent.height ||
	    impl.info.width < video_caps.minCodedExtent.width ||
	    impl.info.height < video_caps.minCodedExtent.height)
	{
		return false;
	}

	return true;
}

uint32_t VideoEncoderCaps::get_aligned_width(uint32_t width) const
{
	return (width + video_caps.pictureAccessGranularity.width - 1) &
	       ~(video_caps.pictureAccessGranularity.width - 1);
}

uint32_t VideoEncoderCaps::get_aligned_height(uint32_t height) const
{
	return (height + video_caps.pictureAccessGranularity.height - 1) &
	       ~(video_caps.pictureAccessGranularity.height - 1);
}

bool VideoSession::init(Encoder::Impl &impl)
{
	auto &table = impl.table;

	VkVideoSessionCreateInfoKHR session_info = { VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR };
	session_info.maxActiveReferencePictures = MaxActiveReferencePictures;
	session_info.maxCodedExtent.width = impl.caps.get_aligned_width(impl.info.width);
	session_info.maxCodedExtent.height = impl.caps.get_aligned_height(impl.info.height);
	session_info.maxDpbSlots = DPBSize;
	session_info.pVideoProfile = &impl.profile.profile_info;
	session_info.queueFamilyIndex = impl.info.encode_queue.family_index;
	session_info.pictureFormat = impl.profile.input.format;
	session_info.referencePictureFormat = impl.profile.input.format;
	session_info.pStdHeaderVersion = &impl.caps.video_caps.stdHeaderVersion;

	if (VK_CALL(vkCreateVideoSessionKHR(impl.info.device, &session_info, nullptr, &session)) != VK_SUCCESS)
		return false;

	uint32_t count;
	if (VK_CALL(vkGetVideoSessionMemoryRequirementsKHR(impl.info.device, session, &count, nullptr)) != VK_SUCCESS)
		return false;
	std::vector<VkVideoSessionMemoryRequirementsKHR> session_reqs(count);
	for (auto &req : session_reqs)
		req.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR;
	if (VK_CALL(vkGetVideoSessionMemoryRequirementsKHR(impl.info.device, session, &count, session_reqs.data())) != VK_SUCCESS)
		return false;

	std::vector<VkBindVideoSessionMemoryInfoKHR> binds;
	binds.reserve(count);

	for (auto &req : session_reqs)
	{
		VkBindVideoSessionMemoryInfoKHR bind_info = { VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR };

		Memory mem = {};
		if (!impl.allocate_memory(mem, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, req.memoryRequirements) &&
		    !impl.allocate_memory(mem, 0, req.memoryRequirements))
		{
			return false;
		}

		memory.push_back(mem);
		bind_info.memoryBindIndex = req.memoryBindIndex;
		bind_info.memory = mem.memory;
		bind_info.memoryOffset = 0;
		bind_info.memorySize = req.memoryRequirements.size;
		binds.push_back(bind_info);
	}

	if (VK_CALL(vkBindVideoSessionMemoryKHR(impl.info.device, session, count, binds.data())) != VK_SUCCESS)
		return false;

	return true;
}

void VideoSession::destroy(Encoder::Impl &impl)
{
	auto &table = impl.table;
	VK_CALL(vkDestroyVideoSessionKHR(impl.info.device, session, nullptr));
	for (auto &mem : memory)
		impl.free_memory(mem);

	memory.clear();
	session = VK_NULL_HANDLE;
}

bool VideoSessionParameters::init(Encoder::Impl &impl)
{
	switch (impl.info.profile)
	{
	case Profile::H264_High:
		return init_h264(impl);

	case Profile::H265_Main:
	case Profile::H265_Main10:
		return init_h265(impl);

	default:
		return false;
	}
}

static float saturate(float value)
{
	if (value > 1.0f)
		return 1.0f;
	else if (value < 0.0f)
		return 0.0f;
	else
		return value;
}

// Could trivially use compiler intrinsics, but this isn't perf critical, so *shrug*.
static uint32_t find_lsb(uint32_t v)
{
	for (int i = 0; i < 32; i++)
		if (v & (1u << i))
			return i;

	return UINT32_MAX;
}

static uint32_t find_msb(uint32_t v)
{
	for (int i = 31; i >= 0; i--)
		if (v & (1u << i))
			return i;

	return UINT32_MAX;
}

bool VideoSessionParameters::init_h265(Encoder::Impl &impl)
{
	// Mostly adapted from nvpro-samples VkEncoderConfigH265.cpp.
	// Most of this is HEVC gibberish. I don't really know what I'm doing here :)

	VkVideoSessionParametersCreateInfoKHR session_param_info =
		{ VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR };
	VkVideoEncodeH265SessionParametersCreateInfoKHR h265_session_param_info =
		{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR };
	VkVideoEncodeH265SessionParametersAddInfoKHR add_info =
		{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR };

	h265_session_param_info.maxStdPPSCount = 1;
	h265_session_param_info.maxStdSPSCount = 1;
	h265_session_param_info.maxStdVPSCount = 1;

	auto &sps = h265.sps;
	auto &pps = h265.pps;
	auto &vps = h265.vps;
	auto &vui = h265.vui;

	pps = {};
	sps = {};
	vps = {};
	vui = {};

	// VUI
	vui.flags.aspect_ratio_info_present_flag = 1;
	// Aspect ratio of pixels (SAR), not actual aspect ratio. Confusing, I know.
	vui.aspect_ratio_idc = STD_VIDEO_H265_ASPECT_RATIO_IDC_SQUARE;

	vui.flags.chroma_loc_info_present_flag = 1;
	vui.chroma_sample_loc_type_bottom_field = 0;
	vui.chroma_sample_loc_type_top_field = 0;

	vui.flags.video_signal_type_present_flag = 1;
	vui.flags.video_full_range_flag = 0;
	vui.flags.colour_description_present_flag = 1;

	vui.flags.vui_timing_info_present_flag = 1;
	vui.vui_num_units_in_tick = impl.info.frame_rate_den;
	vui.vui_time_scale = impl.info.frame_rate_num;

	vui.video_format = 5; // Unspecified. The specified ones cover legacy PAL/NTSC, etc.
	vui.colour_primaries = 1; // BT.709
	vui.transfer_characteristics = 1; // BT.709
	vui.matrix_coeffs = 1; // BT.709

	sps.flags.vui_parameters_present_flag = 1;
	sps.pSequenceParameterSetVui = &vui;

	if (impl.profile.profile_info.chromaSubsampling == VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR)
		sps.chroma_format_idc = STD_VIDEO_H265_CHROMA_FORMAT_IDC_420;
	else if (impl.profile.profile_info.chromaSubsampling == VK_VIDEO_CHROMA_SUBSAMPLING_422_BIT_KHR)
		sps.chroma_format_idc = STD_VIDEO_H265_CHROMA_FORMAT_IDC_422;
	else
		sps.chroma_format_idc = STD_VIDEO_H265_CHROMA_FORMAT_IDC_444;

	bool is_10bit = impl.info.profile == Profile::H265_Main10;
	if (is_10bit)
	{
		sps.bit_depth_luma_minus8 = 2;
		sps.bit_depth_chroma_minus8 = 2;
	}

	auto &caps = impl.caps.h265.caps;

	StdVideoH265ProfileTierLevel level = {};
	level.general_level_idc = std::min<StdVideoH265LevelIdc>(caps.maxLevelIdc, STD_VIDEO_H265_LEVEL_IDC_6_1);
	level.general_profile_idc = impl.profile.h265.profile.stdProfileIdc;
	sps.pProfileTierLevel = &level;
	level.flags.general_progressive_source_flag = 1;
	level.flags.general_frame_only_constraint_flag = 1;
	level.flags.general_tier_flag = level.general_level_idc >= STD_VIDEO_H265_LEVEL_IDC_5_0;

	auto syntax_flags = caps.stdSyntaxFlags;

	if (impl.info.hints.tuning != VK_VIDEO_ENCODE_TUNING_MODE_LOW_LATENCY_KHR &&
	    impl.info.hints.tuning != VK_VIDEO_ENCODE_TUNING_MODE_ULTRA_LOW_LATENCY_KHR)
	{
		if ((syntax_flags & VK_VIDEO_ENCODE_H265_STD_SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG_SET_BIT_KHR) != 0)
			sps.flags.sample_adaptive_offset_enabled_flag = 1;
	}

	sps.flags.amp_enabled_flag = 1;
	sps.flags.strong_intra_smoothing_enabled_flag = 1;

	constexpr uint32_t cb_min_size = 8;

	// Align this to PAG, or RADV breaks at least.
	const uint32_t alignment_width = std::max<uint32_t>(cb_min_size, impl.caps.video_caps.pictureAccessGranularity.width);
	const uint32_t alignment_height = std::max<uint32_t>(cb_min_size, impl.caps.video_caps.pictureAccessGranularity.height);
	uint32_t aligned_width = (impl.info.width + alignment_width - 1) & ~(alignment_width - 1);
	uint32_t aligned_height = (impl.info.height + alignment_height - 1) & ~(alignment_height - 1);

	if (aligned_width != impl.info.width || aligned_height != impl.info.height)
	{
		sps.flags.conformance_window_flag = 1;
		sps.conf_win_right_offset = aligned_width - impl.info.width;
		sps.conf_win_bottom_offset = aligned_height - impl.info.height;

		if (sps.chroma_format_idc != STD_VIDEO_H265_CHROMA_FORMAT_IDC_444)
			sps.conf_win_right_offset >>= 1;
		if (sps.chroma_format_idc == STD_VIDEO_H265_CHROMA_FORMAT_IDC_420)
			sps.conf_win_bottom_offset >>= 1;
	}

	sps.pic_width_in_luma_samples = aligned_width;
	sps.pic_height_in_luma_samples = aligned_height;
	// This is arbitrary.
	sps.log2_max_pic_order_cnt_lsb_minus4 = 4;

	// This is *not* the minimum ctbSizes. Force this to 8x8, otherwise stuff breaks since it's the minimum per spec.
	// CB size != CTB size! I spent a full day on debugging this :(
	sps.log2_min_luma_coding_block_size_minus3 = 0;
	sps.log2_diff_max_min_luma_coding_block_size = find_msb(caps.ctbSizes) + 1; // First bit is 16x16.
	sps.log2_min_luma_transform_block_size_minus2 = find_lsb(caps.transformBlockSizes);
	sps.log2_diff_max_min_luma_transform_block_size =
			find_msb(caps.transformBlockSizes) - find_lsb(caps.transformBlockSizes);

	// First bit for transformBlockSize is 4x4.
	uint32_t max_transform_hierarchy = (find_msb(caps.ctbSizes) + 4) - (find_lsb(caps.transformBlockSizes) + 2);
	sps.max_transform_hierarchy_depth_inter = max_transform_hierarchy;
	sps.max_transform_hierarchy_depth_intra = max_transform_hierarchy;

	StdVideoH265DecPicBufMgr dec_pic_buf_mgr = {};

	// Should probably provide pShortTermRefPicSet here I think,
	// but GPU hangs randomly on RADV when using that.
	sps.pDecPicBufMgr = &dec_pic_buf_mgr;

	vps.pProfileTierLevel = &level;
	vps.pDecPicBufMgr = &dec_pic_buf_mgr;

	if ((syntax_flags & VK_VIDEO_ENCODE_H265_STD_TRANSFORM_SKIP_ENABLED_FLAG_SET_BIT_KHR) != 0 ||
	    (syntax_flags & VK_VIDEO_ENCODE_H265_STD_TRANSFORM_SKIP_ENABLED_FLAG_UNSET_BIT_KHR) == 0)
	{
		pps.flags.transform_skip_enabled_flag = 1;
		pps.log2_max_transform_skip_block_size_minus2 = find_msb(caps.transformBlockSizes);
	}

	if ((syntax_flags & VK_VIDEO_ENCODE_H265_STD_TRANSQUANT_BYPASS_ENABLED_FLAG_SET_BIT_KHR) != 0)
		pps.flags.transquant_bypass_enabled_flag = impl.info.hints.tuning == VK_VIDEO_ENCODE_TUNING_MODE_LOSSLESS_KHR ? 1 : 0;
	if ((caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H265_STD_CONSTRAINED_INTRA_PRED_FLAG_SET_BIT_KHR) != 0)
		pps.flags.constrained_intra_pred_flag = 1;

	pps.flags.deblocking_filter_control_present_flag = 1;
	pps.flags.cabac_init_present_flag = 1;

	// This is required or NV output gets really broken.
	pps.flags.cu_qp_delta_enabled_flag = 1;

	// These seems to be required for some reason.
	vps.flags.vps_temporal_id_nesting_flag = 1;
	sps.flags.sps_temporal_id_nesting_flag = 1;

	add_info.pStdPPSs = &pps;
	add_info.pStdSPSs = &sps;
	add_info.pStdVPSs = &vps;
	add_info.stdPPSCount = 1;
	add_info.stdSPSCount = 1;
	add_info.stdVPSCount = 1;

	h265_session_param_info.pParametersAddInfo = &add_info;
	session_param_info.pNext = &h265_session_param_info;
	session_param_info.videoSession = impl.session.session;

	auto &table = impl.table;

	// Simple rounding to nearest quality level.
	quality_level.qualityLevel = uint32_t(saturate(impl.info.quality_level) *
	                                      float(impl.caps.encode_caps.maxQualityLevels - 1) + 0.5f);
	h265_session_param_info.pNext = &quality_level;

	// Query some properties for the quality level we chose.
	VkPhysicalDeviceVideoEncodeQualityLevelInfoKHR quality_level_info =
		{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_ENCODE_QUALITY_LEVEL_INFO_KHR };
	quality_level_info.pVideoProfile = &impl.profile.profile_info;
	quality_level_info.qualityLevel = quality_level.qualityLevel;
	h265.quality_level_props = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_QUALITY_LEVEL_PROPERTIES_KHR };
	quality_level_props.pNext = &h265.quality_level_props;
	VK_CALL(vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR(impl.info.gpu, &quality_level_info,
	                                                                &quality_level_props));

	if (VK_CALL(vkCreateVideoSessionParametersKHR(impl.info.device, &session_param_info,
	                                              nullptr, &params)) != VK_SUCCESS)
	{
		params = VK_NULL_HANDLE;
		return false;
	}

	VkVideoEncodeSessionParametersGetInfoKHR params_get_info =
		{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_SESSION_PARAMETERS_GET_INFO_KHR };
	VkVideoEncodeH265SessionParametersGetInfoKHR h265_params_get_info =
		{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_SESSION_PARAMETERS_GET_INFO_KHR };
	VkVideoEncodeSessionParametersFeedbackInfoKHR feedback_info =
		{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_SESSION_PARAMETERS_FEEDBACK_INFO_KHR };
	VkVideoEncodeH265SessionParametersFeedbackInfoKHR h265_feedback_info =
		{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_SESSION_PARAMETERS_FEEDBACK_INFO_KHR };
	params_get_info.pNext = &h265_params_get_info;
	feedback_info.pNext = &h265_feedback_info;

	params_get_info.videoSessionParameters = params;
	h265_params_get_info.writeStdPPS = VK_TRUE;
	h265_params_get_info.writeStdSPS = VK_TRUE;
	h265_params_get_info.writeStdVPS = VK_TRUE;

	encoded_parameters.resize(256);
	size_t params_size = encoded_parameters.size();
	auto res = VK_CALL(vkGetEncodedVideoSessionParametersKHR(
			impl.info.device, &params_get_info,
			&feedback_info, &params_size, encoded_parameters.data()));

	if (res != VK_SUCCESS)
	{
		VK_CALL(vkDestroyVideoSessionParametersKHR(impl.info.device, params, nullptr));
		params = VK_NULL_HANDLE;
	}

	encoded_parameters.resize(params_size);
	return true;
}

bool VideoSessionParameters::init_h264(Encoder::Impl &impl)
{
	VkVideoSessionParametersCreateInfoKHR session_param_info =
		{ VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR };
	VkVideoEncodeH264SessionParametersCreateInfoKHR h264_session_param_info =
		{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR };
	h264_session_param_info.maxStdPPSCount = 1;
	h264_session_param_info.maxStdSPSCount = 1;

	auto &sps = h264.sps;
	auto &pps = h264.pps;
	auto &vui = h264.vui;

	pps = {};
	sps = {};
	vui = {};

	sps.flags.vui_parameters_present_flag = 1;
	sps.pSequenceParameterSetVui = &vui;

	vui.flags.aspect_ratio_info_present_flag = 1;
	// Aspect ratio of pixels (SAR), not actual aspect ratio. Confusing, I know.
	vui.aspect_ratio_idc = STD_VIDEO_H264_ASPECT_RATIO_IDC_SQUARE;

	// Center chroma siting would be a bit nicer,
	// but NV drivers have a bug where they only write Left siting.
	// Left siting is "standard", so it's more compatible to use that either way.
	vui.flags.chroma_loc_info_present_flag = 1;
	vui.chroma_sample_loc_type_bottom_field = 0;
	vui.chroma_sample_loc_type_top_field = 0;

	vui.flags.timing_info_present_flag = 1;
	vui.flags.fixed_frame_rate_flag = 1;
	// Multiplying by 2 here since H.264 seems to imply "field rate", even for progressive scan.
	// When playing back in ffplay, this makes it work as expected.
	// mpv seems to make up its own timestamps, however.
	// It is possible we may have to emit SEI packets on our own?
	// Most likely our packets will be muxed into some sensible container which has its own timestamp mechanism.
	vui.time_scale = impl.info.frame_rate_num * 2;
	vui.num_units_in_tick = impl.info.frame_rate_den;

	vui.flags.video_signal_type_present_flag = 1;
	vui.flags.video_full_range_flag = 0;
	vui.flags.color_description_present_flag = 1;

	vui.video_format = 5; // Unspecified. The specified ones cover legacy PAL/NTSC, etc.
	vui.colour_primaries = 1; // BT.709
	vui.transfer_characteristics = 1; // BT.709
	vui.matrix_coefficients = 1; // BT.709

	VkVideoEncodeH264SessionParametersAddInfoKHR add_info =
		{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR };

	if (impl.profile.profile_info.chromaSubsampling == VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR)
		sps.chroma_format_idc = STD_VIDEO_H264_CHROMA_FORMAT_IDC_420;
	else if (impl.profile.profile_info.chromaSubsampling == VK_VIDEO_CHROMA_SUBSAMPLING_422_BIT_KHR)
		sps.chroma_format_idc = STD_VIDEO_H264_CHROMA_FORMAT_IDC_422;
	else
		sps.chroma_format_idc = STD_VIDEO_H264_CHROMA_FORMAT_IDC_444;

	sps.profile_idc = impl.profile.h264.profile.stdProfileIdc;
	sps.level_idc = impl.caps.h264.caps.maxLevelIdc;

	uint32_t aligned_width = impl.caps.get_aligned_width(impl.info.width);
	uint32_t aligned_height = impl.caps.get_aligned_height(impl.info.height);

	if (aligned_width != impl.info.width || aligned_height != impl.info.height)
	{
		sps.flags.frame_cropping_flag = 1;
		sps.frame_crop_right_offset = aligned_width - impl.info.width;
		sps.frame_crop_bottom_offset = aligned_height - impl.info.height;

		if (sps.chroma_format_idc != STD_VIDEO_H264_CHROMA_FORMAT_IDC_444)
			sps.frame_crop_right_offset >>= 1;
		if (sps.chroma_format_idc == STD_VIDEO_H264_CHROMA_FORMAT_IDC_420)
			sps.frame_crop_bottom_offset >>= 1;
	}

	// TODO: May need to change for B-frames.
	sps.max_num_ref_frames = MaxActiveReferencePictures;
	// IIRC, this means no interlacing.
	sps.flags.frame_mbs_only_flag = 1;
	// Seen this in samples, but otherwise I don't know.
	sps.flags.direct_8x8_inference_flag = 1;
	// Somewhat arbitrary, but type 0 seems easiest to reason about.
	sps.pic_order_cnt_type = STD_VIDEO_H264_POC_TYPE_0;

	constexpr uint32_t H264MacroBlockSize = 16;
	sps.pic_width_in_mbs_minus1 = aligned_width / H264MacroBlockSize - 1;
	sps.pic_height_in_map_units_minus1 = aligned_height / H264MacroBlockSize - 1;

	// This is arbitrary.
	sps.log2_max_pic_order_cnt_lsb_minus4 = 4;

	if (impl.caps.h264.caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_TRANSFORM_8X8_MODE_FLAG_SET_BIT_KHR)
		pps.flags.transform_8x8_mode_flag = 1;
	if (impl.caps.h264.caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_ENTROPY_CODING_MODE_FLAG_SET_BIT_KHR)
		pps.flags.entropy_coding_mode_flag = 1;

	add_info.pStdPPSs = &pps;
	add_info.pStdSPSs = &sps;
	add_info.stdPPSCount = 1;
	add_info.stdSPSCount = 1;

	h264_session_param_info.pParametersAddInfo = &add_info;
	session_param_info.pNext = &h264_session_param_info;
	session_param_info.videoSession = impl.session.session;

	auto &table = impl.table;

	// Simple rounding to nearest quality level.
	quality_level.qualityLevel = uint32_t(saturate(impl.info.quality_level) *
	                                      float(impl.caps.encode_caps.maxQualityLevels - 1) + 0.5f);
	h264_session_param_info.pNext = &quality_level;

	// Query some properties for the quality level we chose.
	VkPhysicalDeviceVideoEncodeQualityLevelInfoKHR quality_level_info =
			{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_ENCODE_QUALITY_LEVEL_INFO_KHR };
	quality_level_info.pVideoProfile = &impl.profile.profile_info;
	quality_level_info.qualityLevel = quality_level.qualityLevel;
	h264.quality_level_props = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_QUALITY_LEVEL_PROPERTIES_KHR };
	quality_level_props.pNext = &h264.quality_level_props;
	VK_CALL(vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR(impl.info.gpu, &quality_level_info,
	                                                                &quality_level_props));

	// A low quality mode might opt for using CAVLC instead of CABAC.
	if (!h264.quality_level_props.preferredStdEntropyCodingModeFlag &&
	    (impl.caps.h264.caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_ENTROPY_CODING_MODE_FLAG_UNSET_BIT_KHR) != 0)
	{
		pps.flags.entropy_coding_mode_flag = 0;
	}

	if (VK_CALL(vkCreateVideoSessionParametersKHR(impl.info.device, &session_param_info,
	                                              nullptr, &params)) != VK_SUCCESS)
	{
		params = VK_NULL_HANDLE;
		return false;
	}

	VkVideoEncodeSessionParametersGetInfoKHR params_get_info =
		{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_SESSION_PARAMETERS_GET_INFO_KHR };
	VkVideoEncodeH264SessionParametersGetInfoKHR h264_params_get_info =
		{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_GET_INFO_KHR };
	VkVideoEncodeSessionParametersFeedbackInfoKHR feedback_info =
		{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_SESSION_PARAMETERS_FEEDBACK_INFO_KHR };
	VkVideoEncodeH264SessionParametersFeedbackInfoKHR h264_feedback_info =
		{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_FEEDBACK_INFO_KHR };
	params_get_info.pNext = &h264_params_get_info;
	feedback_info.pNext = &h264_feedback_info;

	params_get_info.videoSessionParameters = params;
	h264_params_get_info.writeStdPPS = VK_TRUE;
	h264_params_get_info.writeStdSPS = VK_TRUE;

	encoded_parameters.resize(256);
	size_t params_size = encoded_parameters.size();
	auto res = VK_CALL(vkGetEncodedVideoSessionParametersKHR(
			impl.info.device, &params_get_info,
			&feedback_info, &params_size, encoded_parameters.data()));

	if (res != VK_SUCCESS)
	{
		VK_CALL(vkDestroyVideoSessionParametersKHR(impl.info.device, params, nullptr));
		params = VK_NULL_HANDLE;
	}

	encoded_parameters.resize(params_size);
	return true;
}

void VideoSessionParameters::destroy(Encoder::Impl &impl)
{
	auto &table = impl.table;

	VK_CALL(vkDestroyVideoSessionParametersKHR(impl.info.device, params, nullptr));
	params = VK_NULL_HANDLE;
}

RateControl::RateControl()
{
	info = {};
	info.bitrate_kbits = 5000;
	info.max_bitrate_kbits = 10000;
	info.mode = RateControlMode::VBR;
	info.gop_frames = 120;
}

bool RateControl::init(Encoder::Impl &impl)
{
	ctrl_info.pNext = &rate_info;
	ctrl_info.flags = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR |
	                  VK_VIDEO_CODING_CONTROL_ENCODE_RATE_CONTROL_BIT_KHR;

	rate_info = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_INFO_KHR };

	auto modes = impl.caps.encode_caps.rateControlModes;
	if (info.mode == RateControlMode::VBR && (modes & VK_VIDEO_ENCODE_RATE_CONTROL_MODE_VBR_BIT_KHR) != 0)
		rate_info.rateControlMode = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_VBR_BIT_KHR;
	else if (info.mode == RateControlMode::CBR && (modes & VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR) != 0)
		rate_info.rateControlMode = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR;
	else if (info.mode == RateControlMode::ConstantQP && (modes & VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR))
		rate_info.rateControlMode = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR;
	else
		rate_info.rateControlMode = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DEFAULT_KHR;

	if (rate_info.rateControlMode != VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR &&
	    rate_info.rateControlMode != VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DEFAULT_KHR)
	{
		if (rate_info.rateControlMode == VK_VIDEO_ENCODE_RATE_CONTROL_MODE_VBR_BIT_KHR ||
		    (impl.info.hints.tuning != VK_VIDEO_ENCODE_TUNING_MODE_LOW_LATENCY_KHR &&
		     impl.info.hints.tuning != VK_VIDEO_ENCODE_TUNING_MODE_ULTRA_LOW_LATENCY_KHR))
		{
			// Large virtual buffer.
			rate_info.virtualBufferSizeInMs = 2000;
			rate_info.initialVirtualBufferSizeInMs = 1000;
		}
		else
		{
			// Low-latency. 2 frame's worth of VBV buffer window.
			rate_info.virtualBufferSizeInMs = uint32_t(2000ull * impl.info.frame_rate_den / impl.info.frame_rate_num);
			rate_info.initialVirtualBufferSizeInMs = rate_info.virtualBufferSizeInMs / 2;
		}

		rate_info.layerCount = 1;
		rate_info.pLayers = &layer;

		layer.frameRateNumerator = impl.info.frame_rate_num;
		layer.frameRateDenominator = impl.info.frame_rate_den;

		layer.averageBitrate = info.bitrate_kbits * 1000ull;
		layer.maxBitrate = info.max_bitrate_kbits * 1000ull;

		// RADV reports these as 0 for some reason ...
		if (impl.caps.encode_caps.maxBitrate && layer.averageBitrate > impl.caps.encode_caps.maxBitrate)
			layer.averageBitrate = impl.caps.encode_caps.maxBitrate;
		if (impl.caps.encode_caps.maxBitrate && layer.maxBitrate > impl.caps.encode_caps.maxBitrate)
			layer.maxBitrate = impl.caps.encode_caps.maxBitrate;

		if (rate_info.rateControlMode == VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR)
			layer.maxBitrate = layer.averageBitrate;

		if (impl.info.profile == Profile::H264_High)
		{
			rate_info.pNext = &h264.rate_control;
			layer.pNext = &h264.layer;

			h264.rate_control = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_RATE_CONTROL_INFO_KHR };
			h264.layer = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_RATE_CONTROL_LAYER_INFO_KHR };

			// If GOP is invalid, override it with some sensible defaults.
			if (info.gop_frames == 0)
				info.gop_frames = impl.session_params.h264.quality_level_props.preferredGopFrameCount;
			if (info.gop_frames == 0)
				info.gop_frames = 1;

			h264.rate_control.consecutiveBFrameCount = 0;
			h264.rate_control.idrPeriod = info.gop_frames;
			h264.rate_control.gopFrameCount = info.gop_frames;
			// VUID 07022 only says we have to set this if layerCount > 1, not if it's == 1.
			// Seems to work fine.
			//h264.rate_control.temporalLayerCount = 1;

			// When we start using intra-refresh, we cannot use these.
			if ((impl.session_params.h264.quality_level_props.preferredRateControlFlags &
			     VK_VIDEO_ENCODE_H264_RATE_CONTROL_REFERENCE_PATTERN_FLAT_BIT_KHR) != 0)
			{
				// VUID 02818. If REFERENCE_PATTERN_FLAT is used, REGULAR_GOP must also be set.
				h264.rate_control.flags |= VK_VIDEO_ENCODE_H264_RATE_CONTROL_REGULAR_GOP_BIT_KHR |
				                           VK_VIDEO_ENCODE_H264_RATE_CONTROL_REFERENCE_PATTERN_FLAT_BIT_KHR;
			}
			else if ((impl.session_params.h264.quality_level_props.preferredRateControlFlags &
			          VK_VIDEO_ENCODE_H264_RATE_CONTROL_REGULAR_GOP_BIT_KHR) != 0)
			{
				h264.rate_control.flags |= VK_VIDEO_ENCODE_H264_RATE_CONTROL_REGULAR_GOP_BIT_KHR;
			}

			// We don't consider dyadic patterns here. We don't use B-frames yet.

			// Don't attempt HRD compliance since we're not emitting HRD data in SPS/PPS anyway.
			//if (impl.caps.h264.caps.flags & VK_VIDEO_ENCODE_H264_CAPABILITY_HRD_COMPLIANCE_BIT_KHR)
			//	h264.rate_control.flags |= VK_VIDEO_ENCODE_H264_RATE_CONTROL_ATTEMPT_HRD_COMPLIANCE_BIT_KHR;

			// Might not be enough to ensure our payload buffer does not overflow.
			h264.layer.useMaxFrameSize = VK_TRUE;
			h264.layer.maxFrameSize.frameISize = MaxPayloadSize;
			h264.layer.maxFrameSize.framePSize = MaxPayloadSize;
			h264.layer.maxFrameSize.frameBSize = MaxPayloadSize;
		}
		else if (impl.info.profile == Profile::H265_Main || impl.info.profile == Profile::H265_Main10)
		{
			rate_info.pNext = &h265.rate_control;
			layer.pNext = &h265.layer;

			h265.rate_control = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_RATE_CONTROL_INFO_KHR };
			h265.layer = { VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_RATE_CONTROL_LAYER_INFO_KHR };

			// If GOP is invalid, override it with some sensible defaults.
			if (info.gop_frames == 0)
				info.gop_frames = impl.session_params.h265.quality_level_props.preferredGopFrameCount;
			if (info.gop_frames == 0)
				info.gop_frames = 1;

			h265.rate_control.consecutiveBFrameCount = 0;
			h265.rate_control.idrPeriod = info.gop_frames;
			h265.rate_control.gopFrameCount = info.gop_frames;

			// When we start using intra-refresh, we cannot use these.
			if ((impl.session_params.h265.quality_level_props.preferredRateControlFlags &
			     VK_VIDEO_ENCODE_H265_RATE_CONTROL_REFERENCE_PATTERN_FLAT_BIT_KHR) != 0)
			{
				h265.rate_control.flags |= VK_VIDEO_ENCODE_H265_RATE_CONTROL_REGULAR_GOP_BIT_KHR |
				                           VK_VIDEO_ENCODE_H265_RATE_CONTROL_REFERENCE_PATTERN_FLAT_BIT_KHR;
			}
			else if ((impl.session_params.h265.quality_level_props.preferredRateControlFlags &
			          VK_VIDEO_ENCODE_H265_RATE_CONTROL_REGULAR_GOP_BIT_KHR) != 0)
			{
				h265.rate_control.flags |= VK_VIDEO_ENCODE_H265_RATE_CONTROL_REGULAR_GOP_BIT_KHR;
			}

			h265.layer.useMaxFrameSize = VK_TRUE;
			h265.layer.maxFrameSize.frameISize = MaxPayloadSize;
			h265.layer.maxFrameSize.framePSize = MaxPayloadSize;
			h265.layer.maxFrameSize.frameBSize = MaxPayloadSize;
		}
	}

	impl.session_params.quality_level.pNext = ctrl_info.pNext;
	ctrl_info.pNext = &impl.session_params.quality_level;
	ctrl_info.flags |= VK_VIDEO_CODING_CONTROL_ENCODE_QUALITY_LEVEL_BIT_KHR;

	return true;
}
}
