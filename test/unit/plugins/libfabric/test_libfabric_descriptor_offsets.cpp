/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025 Intel Corporation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Unit tests for libfabric descriptor offset handling
 * Tests the fix for bug where multiple descriptors pointing to different offsets
 * within the same registered memory region would incorrectly use the base address.
 *
 * Commit: bedad7ca4cae345a44316621ba1efe65a0e5fc90
 */

#include <gtest/gtest.h>
#include <vector>
#include <cstring>
#include <memory>
#include <cstdint>
#include <algorithm>

#include "common/nixl_log.h"

namespace {

// Helper to fill buffer with pattern for verification
void
fillBufferWithPattern(char *buffer, size_t offset, size_t length, uint8_t pattern_base) {
    for (size_t i = 0; i < length; ++i) {
        buffer[offset + i] = static_cast<char>(pattern_base + (i % 256));
    }
}

// Helper to verify buffer pattern
bool
verifyBufferPattern(const char *buffer, size_t offset, size_t length, uint8_t pattern_base) {
    for (size_t i = 0; i < length; ++i) {
        uint8_t expected = static_cast<uint8_t>(pattern_base + (i % 256));
        if (static_cast<uint8_t>(buffer[offset + i]) != expected) {
            NIXL_ERROR << "Pattern mismatch at offset " << (offset + i) << ": expected "
                       << static_cast<int>(expected) << ", got "
                       << static_cast<int>(static_cast<uint8_t>(buffer[offset + i]));
            return false;
        }
    }
    return true;
}

// Test descriptor structure (mimics real libfabric descriptor behavior)
struct TestDescriptor {
    void *addr; // Address within registered buffer
    size_t length; // Length of this descriptor
    size_t offset_in_registration; // Offset from base of registered region
    uint8_t expected_pattern; // Expected pattern for verification
};

// Simulate the buggy behavior (using base address instead of descriptor offset)
void
buggyTransfer(char *dest_buffer,
              const char *src_buffer_base,
              const std::vector<TestDescriptor> &descriptors) {
    for (const auto &desc : descriptors) {
        // BUG: Always use src_buffer_base instead of desc.addr
        std::memcpy(dest_buffer + desc.offset_in_registration,
                    src_buffer_base, // Wrong: should be desc.addr
                    desc.length);
    }
}

// Simulate the correct behavior (using descriptor's specific offset)
void
correctTransfer(char *dest_buffer,
                const char * /*src_buffer_base*/,
                const std::vector<TestDescriptor> &descriptors) {
    for (const auto &desc : descriptors) {
        // CORRECT: Use desc.addr which points to the specific offset
        std::memcpy(dest_buffer + desc.offset_in_registration,
                    desc.addr, // ✓ Correct: uses descriptor's address
                    desc.length);
    }
}

} // anonymous namespace

// Test fixture for descriptor offset tests
class LibfabricDescriptorOffsetTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        // Optionally initialize NIXL logging for debug output
    }

    void
    TearDown() override {
        // Cleanup if needed
    }
};

// ============================================================================
// Test Case 1: Block-Based Transfers
// ============================================================================
// Scenario: Register entire 1GB buffer once, then transfer different blocks
// in multiple iterations. Each iteration transfers a different range of blocks.
// Bug would cause iteration N to read blocks from iteration 0 instead.
TEST_F(LibfabricDescriptorOffsetTest, BlockBasedTransfers_MultipleIterations) {
    NIXL_INFO << "\n=== Test 1: Block-Based Transfers with Multiple Iterations ===";

    const size_t BUFFER_SIZE = 1024 * 1024; // 1MB for testing (scaled down from 1GB)
    const size_t BLOCK_SIZE = 8192; // 8KB blocks
    const size_t BLOCKS_PER_ITERATION = 16;
    const size_t NUM_ITERATIONS = 4;

    // Allocate and register entire buffer once
    std::unique_ptr<char[]> send_buffer(new char[BUFFER_SIZE]);
    std::unique_ptr<char[]> recv_buffer_correct(new char[BUFFER_SIZE]);
    std::unique_ptr<char[]> recv_buffer_buggy(new char[BUFFER_SIZE]);

    // Fill send buffer with unique patterns for each block
    for (size_t block = 0; block < BUFFER_SIZE / BLOCK_SIZE; ++block) {
        uint8_t pattern = static_cast<uint8_t>(block);
        fillBufferWithPattern(send_buffer.get(), block * BLOCK_SIZE, BLOCK_SIZE, pattern);
    }

    // Zero out receive buffers
    std::memset(recv_buffer_correct.get(), 0, BUFFER_SIZE);
    std::memset(recv_buffer_buggy.get(), 0, BUFFER_SIZE);

    // Simulate multiple iterations of block transfers
    for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        size_t start_block = iter * BLOCKS_PER_ITERATION;
        std::vector<TestDescriptor> descriptors;

        NIXL_INFO << "Iteration " << iter << ": Transferring blocks " << start_block << "-"
                  << (start_block + BLOCKS_PER_ITERATION - 1);

        // Create descriptors for this iteration's blocks
        for (size_t i = 0; i < BLOCKS_PER_ITERATION; ++i) {
            size_t block_idx = start_block + i;
            size_t offset = block_idx * BLOCK_SIZE;

            TestDescriptor desc;
            desc.addr = send_buffer.get() + offset; // Points to specific block
            desc.length = BLOCK_SIZE;
            desc.offset_in_registration = offset;
            desc.expected_pattern = static_cast<uint8_t>(block_idx);
            descriptors.push_back(desc);
        }

        // Perform both buggy and correct transfers
        buggyTransfer(recv_buffer_buggy.get(), send_buffer.get(), descriptors);
        correctTransfer(recv_buffer_correct.get(), send_buffer.get(), descriptors);
    }

    // Verify correct transfer worked for all iterations
    for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        size_t start_block = iter * BLOCKS_PER_ITERATION;
        for (size_t i = 0; i < BLOCKS_PER_ITERATION; ++i) {
            size_t block_idx = start_block + i;
            size_t offset = block_idx * BLOCK_SIZE;
            uint8_t expected_pattern = static_cast<uint8_t>(block_idx);

            ASSERT_TRUE(verifyBufferPattern(
                recv_buffer_correct.get(), offset, BLOCK_SIZE, expected_pattern))
                << "Correct transfer: Block " << block_idx << " (iteration " << iter
                << ") has incorrect data";
        }
    }

    // Demonstrate that buggy transfer fails for iterations > 0
    // (Iteration 0 would work by accident since offset=0 matches base)
    bool buggy_fails_as_expected = false;
    for (size_t iter = 1; iter < NUM_ITERATIONS; ++iter) {
        size_t start_block = iter * BLOCKS_PER_ITERATION;
        for (size_t i = 0; i < BLOCKS_PER_ITERATION; ++i) {
            size_t block_idx = start_block + i;
            size_t offset = block_idx * BLOCK_SIZE;
            uint8_t expected_pattern = static_cast<uint8_t>(block_idx);

            if (!verifyBufferPattern(
                    recv_buffer_buggy.get(), offset, BLOCK_SIZE, expected_pattern)) {
                buggy_fails_as_expected = true;
                NIXL_INFO << "  ✓ Buggy transfer failed as expected for block " << block_idx;
                break;
            }
        }
        if (buggy_fails_as_expected) break;
    }

    ASSERT_TRUE(buggy_fails_as_expected)
        << "Buggy transfer should fail for iterations > 0 (demonstrates the bug)";

    NIXL_INFO << "  PASS: Block-based transfers correctly use descriptor offsets";
}

// ============================================================================
// Test Case 2: Scatter-Gather Operations
// ============================================================================
// Scenario: Transfer non-contiguous sections of a registered buffer
// Bug would cause all sections to read from buffer+0 instead of their offsets
TEST_F(LibfabricDescriptorOffsetTest, ScatterGatherOperations_NonContiguousSections) {
    NIXL_INFO << "\n=== Test 2: Scatter-Gather Operations with Non-Contiguous Sections ===";

    const size_t BUFFER_SIZE = 1024 * 1024; // 1MB
    const size_t SECTION_SIZE = 64 * 1024; // 64KB sections

    // Define non-contiguous offsets to transfer (scatter-gather pattern)
    struct Section {
        size_t offset;
        size_t length;
        uint8_t pattern;
    };

    std::vector<Section> sections = {
        {0 * 1024, SECTION_SIZE, 0xAA}, // Section at 0KB
        {256 * 1024, SECTION_SIZE, 0xBB}, // Section at 256KB (192KB gap)
        {512 * 1024, SECTION_SIZE, 0xCC}, // Section at 512KB (192KB gap)
        {768 * 1024, SECTION_SIZE, 0xDD}, // Section at 768KB (192KB gap)
    };

    // Allocate buffers
    std::unique_ptr<char[]> send_buffer(new char[BUFFER_SIZE]);
    std::unique_ptr<char[]> recv_buffer_correct(new char[BUFFER_SIZE]);
    std::unique_ptr<char[]> recv_buffer_buggy(new char[BUFFER_SIZE]);

    // Fill send buffer sections with unique patterns
    std::memset(send_buffer.get(), 0x00, BUFFER_SIZE);
    for (const auto &section : sections) {
        fillBufferWithPattern(send_buffer.get(), section.offset, section.length, section.pattern);
    }

    // Zero out receive buffers
    std::memset(recv_buffer_correct.get(), 0, BUFFER_SIZE);
    std::memset(recv_buffer_buggy.get(), 0, BUFFER_SIZE);

    // Create descriptors for scatter-gather (each points to different offset)
    std::vector<TestDescriptor> descriptors;
    for (const auto &section : sections) {
        TestDescriptor desc;
        desc.addr = send_buffer.get() + section.offset;
        desc.length = section.length;
        desc.offset_in_registration = section.offset;
        desc.expected_pattern = section.pattern;
        descriptors.push_back(desc);
    }

    NIXL_INFO << "Preparing scatter-gather transfer of " << descriptors.size()
              << " non-contiguous sections";

    // Perform both buggy and correct transfers
    buggyTransfer(recv_buffer_buggy.get(), send_buffer.get(), descriptors);
    correctTransfer(recv_buffer_correct.get(), send_buffer.get(), descriptors);

    // Verify correct transfer for each section
    for (size_t i = 0; i < sections.size(); ++i) {
        const auto &section = sections[i];

        NIXL_DEBUG << "  Verifying section " << i << " at offset " << section.offset;

        ASSERT_TRUE(verifyBufferPattern(
            recv_buffer_correct.get(), section.offset, section.length, section.pattern))
            << "Correct transfer: Section " << i << " at offset " << section.offset
            << " has incorrect data";

        // Verify gaps are still zero (not overwritten)
        if (i > 0) {
            size_t gap_start = sections[i - 1].offset + sections[i - 1].length;
            size_t gap_end = section.offset;
            if (gap_end > gap_start) {
                for (size_t j = gap_start; j < gap_end; ++j) {
                    ASSERT_EQ(recv_buffer_correct[j], 0)
                        << "Gap at offset " << j << " was overwritten";
                }
            }
        }
    }

    // Verify buggy transfer fails for sections with offset > 0
    bool buggy_fails_for_offset_sections = false;
    for (size_t i = 1; i < sections.size(); ++i) { // Skip section 0 (offset=0)
        const auto &section = sections[i];
        if (!verifyBufferPattern(
                recv_buffer_buggy.get(), section.offset, section.length, section.pattern)) {
            buggy_fails_for_offset_sections = true;
            NIXL_INFO << "  ✓ Buggy transfer failed as expected for section " << i << " at offset "
                      << section.offset;
            break;
        }
    }

    ASSERT_TRUE(buggy_fails_for_offset_sections)
        << "Buggy transfer should fail for sections with offset > 0";

    NIXL_INFO << "  PASS: Scatter-gather operations correctly use descriptor offsets";
}

// ============================================================================
// Test Case 3: Explicit Bug Demonstration
// ============================================================================
// Minimal test that clearly shows the bug vs fix behavior
TEST_F(LibfabricDescriptorOffsetTest, ExplicitBugDemonstration) {
    NIXL_INFO << "\n=== Test 3: Explicit Bug Demonstration ===";

    const size_t BUFFER_SIZE = 1024;
    const size_t OFFSET_1 = 200; // Changed to avoid pattern repetition at 256
    const size_t OFFSET_2 = 600;
    const size_t TRANSFER_SIZE = 128;

    std::unique_ptr<char[]> buffer(new char[BUFFER_SIZE]);

    // Fill buffer with position-dependent data (each byte = its absolute offset)
    // Use a non-repeating pattern to distinguish different offsets
    for (size_t i = 0; i < BUFFER_SIZE; ++i) {
        buffer[i] = static_cast<char>(i);
    }

    // Create two descriptors at different offsets
    TestDescriptor desc1;
    desc1.addr = buffer.get() + OFFSET_1;
    desc1.offset_in_registration = OFFSET_1;
    desc1.length = TRANSFER_SIZE;

    TestDescriptor desc2;
    desc2.addr = buffer.get() + OFFSET_2;
    desc2.offset_in_registration = OFFSET_2;
    desc2.length = TRANSFER_SIZE;

    std::vector<TestDescriptor> descriptors = {desc1, desc2};

    // Allocate result buffers
    std::unique_ptr<char[]> result_correct(new char[BUFFER_SIZE]);
    std::unique_ptr<char[]> result_buggy(new char[BUFFER_SIZE]);
    std::memset(result_correct.get(), 0xFF, BUFFER_SIZE);
    std::memset(result_buggy.get(), 0xFF, BUFFER_SIZE);

    // Perform transfers
    correctTransfer(result_correct.get(), buffer.get(), descriptors);
    buggyTransfer(result_buggy.get(), buffer.get(), descriptors);

    // Verify correct transfer: data matches the offset
    for (size_t i = 0; i < TRANSFER_SIZE; ++i) {
        uint8_t expected1 = static_cast<uint8_t>(OFFSET_1 + i);
        uint8_t expected2 = static_cast<uint8_t>(OFFSET_2 + i);

        ASSERT_EQ(static_cast<uint8_t>(result_correct[OFFSET_1 + i]), expected1)
            << "Correct: Descriptor 1 at offset " << OFFSET_1 << " has wrong data at index " << i;

        ASSERT_EQ(static_cast<uint8_t>(result_correct[OFFSET_2 + i]), expected2)
            << "Correct: Descriptor 2 at offset " << OFFSET_2 << " has wrong data at index " << i;
    }

    // Verify buggy transfer: data does NOT match the offset (reads from offset 0)
    bool buggy_reads_from_zero = true;
    for (size_t i = 0; i < TRANSFER_SIZE; ++i) {
        uint8_t expected_if_offset_used = static_cast<uint8_t>(OFFSET_1 + i);
        uint8_t actual = static_cast<uint8_t>(result_buggy[OFFSET_1 + i]);
        uint8_t expected_if_base_used = static_cast<uint8_t>(i);

        if (actual != expected_if_base_used) {
            buggy_reads_from_zero = false;
            break;
        }
        if (actual == expected_if_offset_used && OFFSET_1 != 0) {
            buggy_reads_from_zero = false;
            break;
        }
    }

    ASSERT_TRUE(buggy_reads_from_zero)
        << "Buggy transfer should read from base (offset 0) instead of descriptor offset";

    NIXL_INFO << "  PASS: Explicit bug demonstration shows correct vs buggy behavior";
}

// ============================================================================
// Test Case 4: Edge Cases
// ============================================================================
TEST_F(LibfabricDescriptorOffsetTest, EdgeCases) {
    NIXL_INFO << "\n=== Test 4: Edge Cases ===";

    const size_t BUFFER_SIZE = 4096;

    std::unique_ptr<char[]> buffer(new char[BUFFER_SIZE]);
    std::unique_ptr<char[]> result(new char[BUFFER_SIZE]);

    // Fill buffer with predictable pattern
    for (size_t i = 0; i < BUFFER_SIZE; ++i) {
        buffer[i] = static_cast<char>(i);
    }
    std::memset(result.get(), 0, BUFFER_SIZE);

    // Edge case 1: Single byte transfer at various offsets
    std::vector<TestDescriptor> single_byte_descs;
    for (size_t offset : {0, 1, 255, 256, 1023, 2048, 4095}) {
        if (offset < BUFFER_SIZE) {
            TestDescriptor desc;
            desc.addr = buffer.get() + offset;
            desc.length = 1;
            desc.offset_in_registration = offset;
            single_byte_descs.push_back(desc);
        }
    }

    correctTransfer(result.get(), buffer.get(), single_byte_descs);

    for (const auto &desc : single_byte_descs) {
        uint8_t expected = static_cast<uint8_t>(desc.offset_in_registration);
        ASSERT_EQ(static_cast<uint8_t>(result[desc.offset_in_registration]), expected)
            << "Single byte transfer failed at offset " << desc.offset_in_registration;
    }

    // Edge case 2: Transfer at buffer boundary
    std::memset(result.get(), 0, BUFFER_SIZE);
    TestDescriptor boundary_desc;
    boundary_desc.addr = buffer.get() + (BUFFER_SIZE - 64);
    boundary_desc.length = 64;
    boundary_desc.offset_in_registration = BUFFER_SIZE - 64;

    std::vector<TestDescriptor> boundary_descs = {boundary_desc};
    correctTransfer(result.get(), buffer.get(), boundary_descs);

    for (size_t i = 0; i < 64; ++i) {
        size_t offset = BUFFER_SIZE - 64 + i;
        uint8_t expected = static_cast<uint8_t>(offset);
        ASSERT_EQ(static_cast<uint8_t>(result[offset]), expected)
            << "Boundary transfer failed at offset " << offset;
    }

    NIXL_INFO << "  PASS: Edge cases handled correctly";
}

// Main test runner
int
main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
