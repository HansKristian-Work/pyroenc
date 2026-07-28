// Copyright (c) 2026 Arntzen Software AS
// SPDX-License-Identifier: MIT

#include "vulkan_headers.hpp"
#include "pyroenc.hpp"
#include "cli_parser.hpp"
#include "context.hpp"
#include "device.hpp"
#include "logging.hpp"
#include "yuv4mpeg.hpp"
#include "path_utils.hpp"

using namespace PyroEnc;
using namespace Granite;
using namespace Vulkan;

static void print_help()
{
	LOGE("Usage: pyroenc-offline file.y4m\n"
		"\t[--output out.{h264,h265}]\n"
		"\t[--help]\n"
		"\t[--intra-refresh <phases>]\n"
		"\t[--bitrate-kbits <kbits>]\n"
		"\t[--hdr10]\n"
		"\t[--quality-level <level [0, 1] range>]\n");
}

int main(int argc, char **argv)
{
	Util::CLICallbacks cbs;
	uint32_t intra_refresh = 0;
	std::string y4m_path;
	std::string out_path;
	float quality_level = 0.0f;
	uint32_t bitrate_kbits = 10;
	bool hdr10 = false;

	cbs.add("--help", [&](Util::CLIParser &parser) { parser.end(); });
	cbs.add("--intra-refresh", [&](Util::CLIParser &parser) { intra_refresh = parser.next_uint(); });
	cbs.add("--bitrate-kbits", [&](Util::CLIParser &parser) { bitrate_kbits = parser.next_uint(); });
	cbs.add("--quality-level", [&](Util::CLIParser &parser) { quality_level = float(parser.next_double()); });
	cbs.add("--output", [&](Util::CLIParser &parser) { out_path = parser.next_string(); });
	cbs.add("--hdr10", [&](Util::CLIParser &) { hdr10 = true; });
	cbs.default_handler = [&](const char *path) { y4m_path = path; };

	Util::CLIParser parser(std::move(cbs), argc - 1, argv + 1);
	if (!parser.parse())
	{
		print_help();
		return EXIT_FAILURE;
	}
	else if (parser.is_ended_state())
	{
		print_help();
		return EXIT_FAILURE;
	}

	if (y4m_path.empty())
	{
		LOGE("Need a y4m path.\n");
		return EXIT_FAILURE;
	}

	if (out_path.empty())
	{
		LOGE("Need output path.\n");
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

	YUV4MPEGFile y4m_file;
	if (!y4m_file.open_read(y4m_path))
	{
		LOGE("Failed to y4m file.\n");
		return EXIT_FAILURE;
	}

	if (y4m_file.get_format() == YUV4MPEGFile::Format::YUV444P ||
		y4m_file.get_format() == YUV4MPEGFile::Format::YUV444P16)
	{
		LOGE("4:4:4 is currently not supported by pyroenc.\n");
		return EXIT_FAILURE;
	}

	LOGI("Opening %s for encode output.\n", argv[1]);
	FILE *file = fopen(out_path.c_str(), "wb");
	if (!file)
	{
		LOGE("Failed to open %s for writing.\n", out_path.c_str());
		return EXIT_FAILURE;
	}

	EncoderCreateInfo info = {};
	info.device = dev.get_device();
	info.instance = dev.get_instance();
	info.gpu = dev.get_physical_device();
	info.get_instance_proc_addr = Context::get_instance_proc_addr();
	info.width = y4m_file.get_width();
	info.height = y4m_file.get_height();
	info.quality_level = quality_level;

	if (Path::ext(out_path) == "h264")
	{
		if (y4m_file.get_format() == YUV4MPEGFile::Format::YUV420P16)
		{
			LOGE("H264 encode only supports 8-bit.\n");
			return EXIT_FAILURE;
		}

		if (hdr10)
		{
			LOGE("H264 encode cannot represent PQ transfer function.\n");
			return EXIT_FAILURE;
		}

		info.profile = Profile::H264_High;
	}
	else if (Path::ext(out_path) == "h265")
	{
		if (y4m_file.get_format() == YUV4MPEGFile::Format::YUV420P16)
			info.profile = Profile::H265_Main10;
		else
			info.profile = Profile::H265_Main;
	}
	else
	{
		LOGE("Must use an extension of .h264 or .h265 for --output.\n");
		return EXIT_FAILURE;
	}

	EncoderDirectYCbCrInfo direct_ycbcr_info = {};
	direct_ycbcr_info.ycbcr_conversion =
		hdr10 ? VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020 : VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;
	direct_ycbcr_info.ycbcr_range =
		y4m_file.is_full_range() ? VK_SAMPLER_YCBCR_RANGE_ITU_FULL : VK_SAMPLER_YCBCR_RANGE_ITU_NARROW;
	direct_ycbcr_info.color_space = hdr10 ? VK_COLOR_SPACE_HDR10_ST2084_EXT : VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
	direct_ycbcr_info.chroma_location_x =
		y4m_file.is_center_chroma() ? VK_CHROMA_LOCATION_MIDPOINT : VK_CHROMA_LOCATION_COSITED_EVEN;
	direct_ycbcr_info.chroma_location_y = VK_CHROMA_LOCATION_MIDPOINT;
	info.direct_ycbcr_info = &direct_ycbcr_info;

	info.encode_queue.queue =
			dev.get_queue_info().queues[QUEUE_INDEX_VIDEO_ENCODE];
	info.encode_queue.family_index =
			dev.get_queue_info().family_indices[QUEUE_INDEX_VIDEO_ENCODE];
	info.conversion_queue.queue =
			dev.get_queue_info().queues[QUEUE_INDEX_COMPUTE];
	info.conversion_queue.family_index =
			dev.get_queue_info().family_indices[QUEUE_INDEX_COMPUTE];

	info.frame_rate_num = y4m_file.get_frame_rate_num();
	info.frame_rate_den = y4m_file.get_frame_rate_den();

	// TODO: Could change options here. Designed for streaming output. Lowest possible latency.
	info.hints.tuning = VK_VIDEO_ENCODE_TUNING_MODE_ULTRA_LOW_LATENCY_KHR;
	info.hints.content = VK_VIDEO_ENCODE_CONTENT_RENDERED_BIT_KHR;
	info.hints.usage = VK_VIDEO_ENCODE_USAGE_STREAMING_BIT_KHR;

	if (intra_refresh && dev.get_device_features().intra_refresh_features.videoEncodeIntraRefresh)
		info.intra_refresh_period = intra_refresh;

	Encoder encoder;
	if (encoder.init_encoder(info) != Result::Success)
	{
		LOGE("Failed to init encoder.\n");
		return EXIT_FAILURE;
	}

	if (intra_refresh && !encoder.intra_refresh_enabled())
	{
		LOGE("Requested intra refresh, but it's not supported.\n");
		return EXIT_FAILURE;
	}

	RateControlInfo rate_info = {};
	rate_info.mode = RateControlMode::CBR;
	rate_info.gop_frames = UINT32_MAX;
	rate_info.max_bitrate_kbits = bitrate_kbits;
	rate_info.bitrate_kbits = bitrate_kbits;
	if (!encoder.set_rate_control_info(rate_info))
	{
		LOGE("Failed to set rate control.\n");
		return EXIT_FAILURE;
	}

	FrameInfo frame = {};

	uint32_t width = info.width;
	uint32_t height = info.height;
	bool high_precision = info.profile == Profile::H265_Main10;
	uint32_t num_pixels = width * height;
	uint32_t num_pixels_chroma = num_pixels / 4;

	std::unique_ptr<uint8_t[]> cb{new uint8_t[num_pixels_chroma * (high_precision ? 2 : 1)]};
	std::unique_ptr<uint8_t[]> cr{new uint8_t[num_pixels_chroma * (high_precision ? 2 : 1)]};

	int64_t pts = 0;
	while (y4m_file.begin_frame())
	{
		EncoderDirectYCbCrImageInfo direct = {};
		if (encoder.allocate_direct_ycbcr_image_info(direct) != Result::Success)
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

		// TODO: We don't really consider padding area here. Just leave it be for purposes of this test.
		// It will likely be zeroed on alloc anyway.

		uint8_t *ptr = static_cast<uint8_t *>(
			cmd->update_image(*wrapped, {}, {width, height, 1},
			                  width, height, {VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1}));

		if (!y4m_file.read(ptr, num_pixels * (high_precision ? 2 : 1)) ||
		    !y4m_file.read(cb.get(), num_pixels_chroma * (high_precision ? 2 : 1)) ||
		    !y4m_file.read(cr.get(), num_pixels_chroma * (high_precision ? 2 : 1)))
		{
			dev.submit_discard(cmd);
			encoder.discard_direct_ycbcr_image_info(direct);
			break;
		}

		ptr = static_cast<uint8_t *>(
			cmd->update_image(*wrapped, {}, {width / 2, height / 2, 1},
			                  width / 2, height / 2, {VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 0, 1}));

		if (high_precision)
		{
			for (uint32_t i = 0; i < num_pixels_chroma; i++)
			{
				memcpy(ptr + 4 * i + 0, cb.get() + 2 * i, sizeof(uint16_t));
				memcpy(ptr + 4 * i + 2, cr.get() + 2 * i, sizeof(uint16_t));
			}
		}
		else
		{
			for (uint32_t i = 0; i < num_pixels_chroma; i++)
			{
				ptr[2 * i + 0] = cb[i];
				ptr[2 * i + 1] = cr[i];
			}
		}

		cmd->image_barrier(*wrapped, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR,
						   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
						   VK_PIPELINE_STAGE_NONE, VK_ACCESS_2_NONE);

		Semaphore sem;
		dev.submit(cmd, nullptr, 1, &sem);
		dev.add_wait_semaphore(CommandBuffer::Type::VideoEncode, std::move(sem), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, true);
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

		if (status == VK_QUERY_RESULT_STATUS_COMPLETE_KHR)
		{
			if (encoded_frame.is_idr())
				fwrite(encoder.get_encoded_parameters(), 1, encoder.get_encoded_parameters_size(), file);
			fwrite(payload, 1, size, file);
			LOGI("Encoded frame %lld ...\n", static_cast<long long>(pts));
		}
		else
		{
			LOGE("Failed with status: %d\n", status);
			break;
		}
	}

	fclose(file);
}