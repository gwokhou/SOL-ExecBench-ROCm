# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.10)
project(sol_execbench_amd_isa_helper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(SOL_EXECBENCH_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(VENDOR_ROOT "${SOL_EXECBENCH_ROOT}/_vendor/isa_spec_manager")
set(COMMON_ROOT "${VENDOR_ROOT}/source/common")
set(TINYXML_ROOT "${VENDOR_ROOT}/source/third_party/tinyxml2")

set(COMMON_SOURCES
    "${COMMON_ROOT}/amdisa_utility.cpp"
    "${COMMON_ROOT}/amdisa_structures.cpp"
    "${COMMON_ROOT}/isa_xml_reader.cpp"
    "${TINYXML_ROOT}/tinyxml2.cpp")

add_library(isa_decoder STATIC
    "${VENDOR_ROOT}/source/isa_decoder/isa_decoder.cpp"
    ${COMMON_SOURCES})
target_include_directories(isa_decoder PRIVATE "${COMMON_ROOT}" "${TINYXML_ROOT}")
target_include_directories(isa_decoder PUBLIC "${VENDOR_ROOT}/include")

add_library(isa_explorer STATIC
    "${VENDOR_ROOT}/source/isa_explorer/isa_explorer.cpp"
    ${COMMON_SOURCES})
target_include_directories(isa_explorer PRIVATE "${COMMON_ROOT}" "${TINYXML_ROOT}")
target_include_directories(isa_explorer PUBLIC "${VENDOR_ROOT}/include")

add_executable(sol-execbench-amd-isa-helper bridge.cpp)
target_include_directories(sol-execbench-amd-isa-helper PRIVATE
    "${VENDOR_ROOT}/include"
    "${VENDOR_ROOT}/source/third_party/json")
target_link_libraries(sol-execbench-amd-isa-helper PRIVATE isa_decoder isa_explorer)
set_target_properties(sol-execbench-amd-isa-helper PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
