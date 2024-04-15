### Copyright (c) 2024 Arntzen Software AS
### SPDX-License-Identifier: MIT

add_executable(pyroenc-test pyroenc_test.cpp)
target_link_libraries(pyroenc-test PRIVATE pyroenc granite-vulkan)
