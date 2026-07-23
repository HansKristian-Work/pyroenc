// Copyright (c) 2024 Arntzen Software AS
// SPDX-License-Identifier: MIT

#include "device.hpp"
#include "context.hpp"
#include "pyroenc.hpp"
#include "logging.hpp"
#include <cmath>

using namespace PyroEnc;
using namespace Vulkan;

// Crude hack to allow testing both paths.
static bool path_is_nv12(const char *path)
{
	auto *last = strrchr(path, '.');
	return last && strcmp(last, ".nv12") == 0;
}

int main(int argc, char *argv[])
{
	if (argc != 7)
	{
		LOGE("Usage: pyroenc-test <path.h264> <path to raw RGBA> <width> <height> <fps> <bitrate-kbits>\n");
		return EXIT_FAILURE;
	}

	LOGI("Opening raw RGBA input: %s\n", argv[2]);
	FILE *input = fopen(argv[2], "rb");
	if (!input)
	{
		LOGE("Failed to open: %s\n", argv[2]);
		return EXIT_FAILURE;
	}

	bool nv12 = path_is_nv12(argv[2]);

	unsigned width = strtoul(argv[3], nullptr, 0);
	unsigned height = strtoul(argv[4], nullptr, 0);
	unsigned fps = strtoul(argv[5], nullptr, 0);
	unsigned kbits = strtoul(argv[6], nullptr, 0);
	if (width == 0 || height == 0 || fps == 0 || kbits == 0)
	{
		LOGE("Width, height or FPS are invalid.\n");
		return EXIT_FAILURE;
	}

	if (!Context::init_loader(nullptr))
	{
		LOGE("Failed to load Vulkan loader.\n");
		return EXIT_FAILURE;
	}

	Context ctx;

	if (!ctx.init_instance_and_device(nullptr, 0, nullptr, 0,
	                                  CONTEXT_CREATION_ENABLE_VIDEO_ENCODE_BIT |
	                                  CONTEXT_CREATION_ENABLE_VIDEO_H264_BIT |
	                                  CONTEXT_CREATION_ENABLE_VIDEO_H265_BIT |
	                                  CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT))
	{
		LOGE("Failed to create context.\n");
		return EXIT_FAILURE;
	}

	Device dev;
	dev.set_context(ctx);

	if (!dev.get_device_features().supports_video_encode_h264)
	{
		LOGE("Device does not support H.264 encode.\n");
		return EXIT_FAILURE;
	}

	if (!dev.get_device_features().supports_video_encode_h265)
	{
		LOGE("Device does not support H.265 encode.\n");
		return EXIT_FAILURE;
	}

	EncoderCreateInfo info = {};
	info.device = dev.get_device();
	info.instance = dev.get_instance();
	info.gpu = dev.get_physical_device();
	info.get_instance_proc_addr = Context::get_instance_proc_addr();
	info.width = width;
	info.height = height;
	info.profile = Profile::H265_Main;
	info.quality_level = 1.0f;

	EncoderDirectYCbCrInfo direct_ycbcr_info = {};
	if (nv12)
	{
		direct_ycbcr_info.ycbcr_conversion = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;
		direct_ycbcr_info.ycbcr_range = VK_SAMPLER_YCBCR_RANGE_ITU_NARROW;
		direct_ycbcr_info.color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
		direct_ycbcr_info.chroma_location_x = VK_CHROMA_LOCATION_COSITED_EVEN;
		direct_ycbcr_info.chroma_location_y = VK_CHROMA_LOCATION_MIDPOINT;
		info.direct_ycbcr_info = &direct_ycbcr_info;
	}

	info.encode_queue.queue =
			dev.get_queue_info().queues[QUEUE_INDEX_VIDEO_ENCODE];
	info.encode_queue.family_index =
			dev.get_queue_info().family_indices[QUEUE_INDEX_VIDEO_ENCODE];
	info.conversion_queue.queue =
			dev.get_queue_info().queues[QUEUE_INDEX_COMPUTE];
	info.conversion_queue.family_index =
			dev.get_queue_info().family_indices[QUEUE_INDEX_COMPUTE];

	info.frame_rate_num = fps;
	info.frame_rate_den = 1;
	info.hints.tuning = VK_VIDEO_ENCODE_TUNING_MODE_LOW_LATENCY_KHR;
	info.hints.content = VK_VIDEO_ENCODE_CONTENT_RENDERED_BIT_KHR;
	info.hints.usage = VK_VIDEO_ENCODE_USAGE_STREAMING_BIT_KHR;

	if (dev.get_device_features().intra_refresh_features.videoEncodeIntraRefresh)
		info.intra_refresh_period = 16;

	Encoder encoder;
	if (encoder.init_encoder(info) != Result::Success)
	{
		LOGE("Failed to init encoder.\n");
		return EXIT_FAILURE;
	}

	if (encoder.intra_refresh_enabled())
		LOGI("Using intra refresh.\n");
	else
		LOGI("Not using intra refresh.\n");

	auto image_info = Vulkan::ImageCreateInfo::render_target(width, height, VK_FORMAT_R8G8B8A8_UNORM);
	image_info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
	image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	ImageHandle rgb_image;

	if (!nv12)
		rgb_image = dev.create_image(image_info);

	FrameInfo frame = {};
	if (rgb_image)
	{
		frame.view = rgb_image->get_view().get_view().view;
		frame.width = rgb_image->get_width();
		frame.height = rgb_image->get_height();
	}

	LOGI("Opening %s for encode output.\n", argv[1]);
	FILE *file = fopen(argv[1], "wb");

	RateControlInfo rate_info = {};
	rate_info.mode = RateControlMode::CBR;
	rate_info.gop_frames = 120;
	rate_info.max_bitrate_kbits = kbits;
	rate_info.bitrate_kbits = kbits;
	if (!encoder.set_rate_control_info(rate_info))
	{
		LOGE("Failed to set rate control.\n");
		return EXIT_FAILURE;
	}

	int64_t pts = 0;
	for (;;)
	{
		if (nv12)
		{
			EncoderDirectYCbCrImageInfo direct = {};
			if (encoder.allocate_direct_ycbcr_image_info(direct) != PyroEnc::Result::Success)
				break;

			frame.view = direct.proxy_image_view;
			frame.width = width;
			frame.height = height;

			auto cmd = dev.request_command_buffer(Vulkan::CommandBuffer::Type::AsyncCompute);

			ImageCreateInfo wrapped_info = {};
			wrapped_info.type = VK_IMAGE_TYPE_2D;
			wrapped_info.format = direct.image_format;
			wrapped_info.misc = IMAGE_MISC_NO_DEFAULT_VIEWS_BIT;
			wrapped_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR;
			wrapped_info.width = direct.padded_width;
			wrapped_info.height = direct.padded_height;
			wrapped_info.depth = 1;
			wrapped_info.layers = 1;
			wrapped_info.levels = 1;
			auto wrapped = dev.wrap_image(wrapped_info, direct.image);

			cmd->image_barrier(*wrapped, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			                   VK_PIPELINE_STAGE_NONE, VK_ACCESS_NONE,
			                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

			void *ptr = cmd->update_image(*wrapped, {}, { width, height, 1 },
				width, height, { VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1 });

			if (fread(ptr, 1, width * height, input) != width * height)
			{
				dev.submit_discard(cmd);
				encoder.discard_direct_ycbcr_image_info(direct);
				break;
			}

			// TODO: We don't really consider padding area here. Just leave it be for purposes of this test.
			// It will likely be zeroed on alloc anyway.
			ptr = cmd->update_image(*wrapped, {}, { width / 2, height / 2, 1 },
			                        width / 2, height / 2, { VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 0, 1 });

			if (fread(ptr, 1, width * height / 2, input) != width * height / 2)
			{
				dev.submit_discard(cmd);
				encoder.discard_direct_ycbcr_image_info(direct);
				break;
			}

			cmd->image_barrier(*wrapped, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR,
			                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
			                   VK_PIPELINE_STAGE_NONE, VK_ACCESS_2_NONE);

			Semaphore sem;
			dev.submit(cmd, nullptr, 1, &sem);
			dev.add_wait_semaphore(CommandBuffer::Type::VideoEncode, std::move(sem), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, true);
		}
		else
		{
			auto cmd = dev.request_command_buffer(Vulkan::CommandBuffer::Type::AsyncCompute);
			cmd->image_barrier(*rgb_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			                   encoder.get_conversion_dst_stage(), VK_ACCESS_NONE,
			                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

			void *ptr = cmd->update_image(*rgb_image);
			if (fread(ptr, sizeof(uint32_t), info.width * info.height, input) != info.width * info.height)
			{
				dev.submit_discard(cmd);
				break;
			}

			// Granite only likes modern sync2 layouts.
			// They are interoperable.
			const auto canonicalize_layout = [](VkImageLayout layout) {
				if (layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
					return VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
				else
					return layout;
			};

			cmd->image_barrier(*rgb_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, canonicalize_layout(encoder.get_conversion_image_layout()),
			                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
			                   encoder.get_conversion_dst_stage(), encoder.get_conversion_dst_access());

			dev.submit(cmd);
		}

		dev.next_frame_context();

		frame.pts = pts++;
		if (encoder.send_frame(frame) != Result::Success)
		{
			LOGE("Failed to send frame for encoding.\n");
			return EXIT_FAILURE;
		}

		// 1:1 since we're using P-frames only.
		EncodedFrame encoded_frame;
		if (encoder.receive_encoded_frame(encoded_frame) != Result::Success)
		{
			LOGE("Failed to receive encoded frame.\n");
			return EXIT_FAILURE;
		}

		if (!encoded_frame.wait())
		{
			LOGE("Failed to wait for encoded frame.\n");
			return EXIT_FAILURE;
		}

		const void *payload = encoded_frame.get_payload();
		auto status = encoded_frame.get_status();
		size_t size = encoded_frame.get_size();

		// Test error recovery in playback (for testing e.g. intra refresh behavior).
		const auto skips_pts = [](uint64_t pts) {
			switch (pts)
			{
#if 0
			case 2:
			case 100:
			case 200:
			case 300:
			case 400:
				return true;
#endif

			default:
				return false;
			}
		};

		if (skips_pts(pts))
		{

		}
		else if (status == VK_QUERY_RESULT_STATUS_COMPLETE_KHR)
		{
			double overhead = encoded_frame.get_encoding_overhead();
			LOGI("PTS = %u, IDR: %s, got frame size: %zu, overhead %.3f ms\n", unsigned(encoded_frame.get_pts()), encoded_frame.is_idr() ? "yes" : "no", size,
			     overhead * 1e3);
			if (encoded_frame.is_idr())
				fwrite(encoder.get_encoded_parameters(), 1, encoder.get_encoded_parameters_size(), file);
			fwrite(payload, 1, size, file);
		}
		else
		{
			LOGE("Failed with status: %d\n", status);
			break;
		}
	}

	fclose(file);
}