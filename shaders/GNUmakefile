all: rgb_to_yuv.inc

rgb_to_yuv.inc: rgb_to_yuv.comp
	glslc -o $@ $< -mfmt=c -Os --target-env=vulkan1.3
