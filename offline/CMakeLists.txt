add_granite_offline_tool(pyroenc-offline pyroenc_offline.cpp yuv4mpeg.cpp yuv4mpeg.hpp)
target_link_libraries(pyroenc-offline PRIVATE pyroenc granite-util)
