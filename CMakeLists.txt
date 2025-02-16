### Copyright (c) 2024 Arntzen Software AS
### SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 99)
project(pyroenc LANGUAGES CXX C)

if (CMAKE_COMPILER_IS_GNUCXX OR (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang"))
    set(PYROENC_CXX_FLAGS -Wshadow -Wall -Wextra -Wno-comment -Wno-missing-field-initializers -Wno-empty-body -fvisibility=hidden)
    if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
        set(PYROENC_CXX_FLAGS ${PYROENC_CXX_FLAGS} -Wno-backslash-newline-escape)
    endif()
    if (NOT (${CMAKE_BUILD_TYPE} MATCHES "Release"))
        message("Enabling frame pointer for profiling/debug.")
        set(PYROENC_CXX_FLAGS ${PYROENC_CXX_FLAGS} -fno-omit-frame-pointer)
    endif()
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
        message("Enabling SSE3 support.")
        set(PYROENC_CXX_FLAGS ${PYROENC_CXX_FLAGS} -msse3)
    endif()
elseif (MSVC)
    set(PYROENC_CXX_FLAGS /D_CRT_SECURE_NO_WARNINGS /wd4267 /wd4244 /wd4309 /wd4005 /MP)
endif()

include(GNUInstallDirs)

if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Granite)
    # Stripped down Granite build
    set(GRANITE_RENDERER OFF CACHE BOOL "" FORCE)
    set(GRANITE_VULKAN_SPIRV_CROSS OFF CACHE BOOL "" FORCE)
    set(GRANITE_VULKAN_SHADER_MANAGER_RUNTIME_COMPILER OFF CACHE BOOL "" FORCE)
    set(GRANITE_VULKAN_FOSSILIZE OFF CACHE BOOL "" FORCE)
    set(GRANITE_SHIPPING ON CACHE BOOL "" FORCE)
    set(GRANITE_VULKAN_SYSTEM_HANDLES OFF CACHE BOOL "" FORCE)
    add_subdirectory(Granite EXCLUDE_FROM_ALL)
    message("pyroenc - Found Granite. Enabling test.")
else()
    message("pyroenc - Did not find Granite. Disabling build of tests.")
endif()

if (NOT TARGET Vulkan::Headers)
    add_subdirectory(vulkan-header EXCLUDE_FROM_ALL)
endif()

add_library(pyroenc STATIC pyroenc.cpp pyroenc.hpp pyroenc_vk_table.inl)
target_link_libraries(pyroenc PUBLIC Vulkan::Headers)
target_compile_options(pyroenc PRIVATE ${PYROENC_CXX_FLAGS})
target_include_directories(pyroenc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Granite)
    add_subdirectory(test)
endif()
