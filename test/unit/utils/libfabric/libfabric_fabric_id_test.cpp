/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Intel Corporation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Tests for the rail fabric ids and the peer rail choice built on them: fabricIdFromEpName(),
 * resolveLocalFabricMask() on synthetic getifaddrs() lists, sameFabric() and pickPeerRail().
 */

#include "libfabric/libfabric_common.h"
#include "common/nixl_log.h"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <cstring>
#include <string>
#include <vector>

#include <rdma/fabric.h>

namespace {

using LibfabricUtils::nixlLibfabricFabricId;

int failures = 0;

void
check(bool cond, const std::string &what) {
    if (cond) {
        NIXL_INFO << "  PASS: " << what;
    } else {
        NIXL_ERROR << "  FAIL: " << what;
        ++failures;
    }
}

struct sockaddr_in
sin4(const char *ip) {
    struct sockaddr_in sin = {};
    sin.sin_family = AF_INET;
    inet_pton(AF_INET, ip, &sin.sin_addr);
    return sin;
}

std::array<char, LF_EP_NAME_MAX_LEN>
epName(const char *ip) {
    const struct sockaddr_in sin = sin4(ip);
    std::array<char, LF_EP_NAME_MAX_LEN> buf = {};
    memcpy(buf.data(), &sin, sizeof(sin));
    return buf;
}

uint32_t
mask(const char *m) {
    uint32_t out = 0;
    inet_pton(AF_INET, m, &out);
    return out;
}

/** A local rail's id: address and netmask. */
nixlLibfabricFabricId
localId(const char *ip, const char *m) {
    nixlLibfabricFabricId id;
    id.valid = true;
    inet_pton(AF_INET, ip, &id.addr);
    id.mask = mask(m);
    return id;
}

/** A peer rail's id, read the way peerFabricIds() reads it. */
nixlLibfabricFabricId
peerId(const char *ip) {
    const auto ep = epName(ip);
    return LibfabricUtils::fabricIdFromEpName(FI_SOCKADDR, ep.data(), ep.size());
}

void
testFabricIdFromEpName() {
    NIXL_INFO << "Testing fabricIdFromEpName";

    const auto a = epName("192.168.10.6");
    const auto id = LibfabricUtils::fabricIdFromEpName(FI_SOCKADDR_IN, a.data(), a.size());
    check(id.valid && id.addr == sin4("192.168.10.6").sin_addr.s_addr,
          "an IPv4 endpoint name gives its address");
    check(id.mask == 0, "a name carries no netmask");
    check(LibfabricUtils::fabricIdFromEpName(FI_SOCKADDR, a.data(), a.size()).valid,
          "the generic sockaddr format reads an IPv4 name");
    check(LibfabricUtils::fabricIdFromEpName(FI_SOCKADDR_IN6, a.data(), a.size()).valid,
          "the family, not the rail's sockaddr format, decides");
    check(!LibfabricUtils::fabricIdFromEpName(FI_ADDR_EFA, a.data(), a.size()).valid,
          "a non-sockaddr format is never read, even if its bytes look like AF_INET");
    check(!LibfabricUtils::fabricIdFromEpName(FI_SOCKADDR_IN, a.data(), 4).valid,
          "a short name gives an invalid id");
    check(!LibfabricUtils::fabricIdFromEpName(FI_SOCKADDR_IN, nullptr, a.size()).valid,
          "a null name gives an invalid id");

    struct sockaddr_in6 sin6 = {};
    sin6.sin6_family = AF_INET6;
    std::array<char, LF_EP_NAME_MAX_LEN> b = {};
    memcpy(b.data(), &sin6, sizeof(sin6));
    check(!LibfabricUtils::fabricIdFromEpName(FI_SOCKADDR, b.data(), b.size()).valid,
          "an IPv6 name gives an invalid id");
}

void
testResolveLocalFabricMask() {
    NIXL_INFO << "Testing resolveLocalFabricMask";

    // lo also holds the rail address as a /32 and comes first; eth0 has it as a /24
    struct sockaddr_in lo_addr = sin4("10.0.0.5"), lo_mask = sin4("255.255.255.255");
    struct sockaddr_in eth_addr = sin4("10.0.0.5"), eth_mask = sin4("255.255.255.0");
    struct sockaddr_in other_addr = sin4("10.0.1.5"), other_mask = sin4("255.255.0.0");
    struct ifaddrs other = {};
    other.ifa_name = const_cast<char *>("eth1");
    other.ifa_addr = reinterpret_cast<struct sockaddr *>(&other_addr);
    other.ifa_netmask = reinterpret_cast<struct sockaddr *>(&other_mask);
    struct ifaddrs eth = {};
    eth.ifa_next = &other;
    eth.ifa_name = const_cast<char *>("eth0");
    eth.ifa_addr = reinterpret_cast<struct sockaddr *>(&eth_addr);
    eth.ifa_netmask = reinterpret_cast<struct sockaddr *>(&eth_mask);
    struct ifaddrs lo = {};
    lo.ifa_next = &eth;
    lo.ifa_name = const_cast<char *>("lo");
    lo.ifa_flags = IFF_LOOPBACK;
    lo.ifa_addr = reinterpret_cast<struct sockaddr *>(&lo_addr);
    lo.ifa_netmask = reinterpret_cast<struct sockaddr *>(&lo_mask);

    auto id = peerId("10.0.0.5");
    LibfabricUtils::resolveLocalFabricMask(id, &lo);
    check(id.mask == mask("255.255.255.0"), "the loopback /32 is skipped for the NIC's /24");

    id = peerId("10.0.1.5");
    LibfabricUtils::resolveLocalFabricMask(id, &lo);
    check(id.mask == mask("255.255.0.0"), "the interface carrying the address gives the mask");

    id = peerId("10.9.9.9");
    LibfabricUtils::resolveLocalFabricMask(id, &lo);
    check(id.valid && id.mask == 0, "an address on no interface keeps mask 0");

    id = peerId("10.0.0.5");
    LibfabricUtils::resolveLocalFabricMask(id, nullptr);
    check(id.mask == 0, "an empty interface list keeps mask 0");

    nixlLibfabricFabricId invalid;
    LibfabricUtils::resolveLocalFabricMask(invalid, &lo);
    check(!invalid.valid && invalid.mask == 0, "an invalid id is left alone");
}

void
testSameFabric() {
    NIXL_INFO << "Testing sameFabric";

    auto local = peerId("192.168.10.6");
    check(LibfabricUtils::sameFabric(local, peerId("192.168.10.6")),
          "without a netmask an exact address matches");
    check(!LibfabricUtils::sameFabric(local, peerId("192.168.10.16")),
          "without a netmask only an exact address matches");
    local.mask = mask("255.255.255.0");
    check(LibfabricUtils::sameFabric(local, peerId("192.168.10.16")),
          "an address on the subnet matches");
    check(!LibfabricUtils::sameFabric(local, peerId("192.168.20.6")),
          "an address off the subnet does not");
    check(!LibfabricUtils::sameFabric(local, nixlLibfabricFabricId{}),
          "an invalid id never matches");
}

/** Peer rail per counter value of the single-rail path, as prepareAndSubmitTransfer() drives
 *  pickPeerRail(): local rail selected_rails[c % L], round c / L. */
std::vector<size_t>
routeSingle(const std::vector<size_t> &selected,
            const std::vector<size_t> &registered,
            const std::vector<nixlLibfabricFabricId> &local_ids,
            const std::vector<nixlLibfabricFabricId> &peer_ids,
            size_t posts) {
    std::vector<size_t> out;
    for (size_t c = 0; c < posts; ++c) {
        out.push_back(LibfabricUtils::pickPeerRail(
            selected, c % selected.size(), c / selected.size(), registered, local_ids, peer_ids));
    }
    return out;
}

/** Which peer rails local rail selected[pos] reaches over the posts of routeSingle(). */
std::vector<size_t>
peersOf(const std::vector<size_t> &route, size_t local_count, size_t pos) {
    std::vector<size_t> peers;
    for (size_t c = pos; c < route.size(); c += local_count) {
        if (std::find(peers.begin(), peers.end(), route[c]) == peers.end()) {
            peers.push_back(route[c]);
        }
    }
    std::sort(peers.begin(), peers.end());
    return peers;
}

void
testPickPeerRail() {
    NIXL_INFO << "Testing pickPeerRail";

    const auto a1 = localId("10.0.0.1", "255.255.255.0");
    const auto a2 = localId("10.0.0.2", "255.255.255.0");
    const auto b1 = localId("10.0.1.1", "255.255.255.0");
    const auto b2 = localId("10.0.1.2", "255.255.255.0");
    const auto pa1 = peerId("10.0.0.11"), pa2 = peerId("10.0.0.12");
    const auto pb1 = peerId("10.0.1.11"), pb2 = peerId("10.0.1.12");
    const std::vector<size_t> four = {0, 1, 2, 3};
    const std::vector<size_t> two = {0, 1};

    // No subnet information: exactly the old registered[c % R] round-robin
    {
        const std::vector<size_t> reg = {1, 3, 4};
        bool same = true;
        for (const auto &peers : {std::vector<nixlLibfabricFabricId>{},
                                  std::vector<nixlLibfabricFabricId>{pa1, pa2, pa1, pa2, pa1}}) {
            const std::vector<nixlLibfabricFabricId> no_local(4);
            const auto &locals =
                peers.empty() ? std::vector<nixlLibfabricFabricId>{a1, a2, a1, a2} : no_local;
            const auto route = routeSingle(four, reg, locals, peers, 40);
            for (size_t c = 0; c < route.size(); ++c) {
                same = same && route[c] == reg[c % reg.size()];
            }
        }
        check(same, "without peer ids or local ids, posts round-robin exactly as before");
    }

    // Flat fabric, any rail counts: still exactly the old round-robin
    {
        bool same = true;
        for (size_t l = 1; l <= 4; ++l) {
            for (size_t r = 1; r <= 4; ++r) {
                std::vector<size_t> sel(l), reg(r);
                std::iota(sel.begin(), sel.end(), 0);
                std::iota(reg.begin(), reg.end(), 0);
                const std::vector<nixlLibfabricFabricId> locals = {a1, a2, a1, a2};
                const std::vector<nixlLibfabricFabricId> peers = {pa1, pa2, pa1, pa2};
                const auto route = routeSingle(sel, reg, locals, peers, 60);
                for (size_t c = 0; c < route.size(); ++c) {
                    same = same && route[c] == c % r;
                }
            }
        }
        check(same, "on one subnet the choice is the old round-robin for every rail count");
    }

    // Interleaved subnets, same on both sides: index pairing, every peer rail used
    {
        const std::vector<nixlLibfabricFabricId> locals = {a1, b1, a2, b2};
        const std::vector<nixlLibfabricFabricId> peers = {pa1, pb1, pa2, pb2};
        const auto route = routeSingle(four, four, locals, peers, 400);
        bool pairs = true;
        for (size_t pos = 0; pos < 4; ++pos) {
            pairs = pairs && peersOf(route, 4, pos) == std::vector<size_t>{pos};
        }
        check(pairs, "interleaved subnets on both sides pair each rail with its own index");
    }

    // Crossed cabling: each local rail always posts to the peer rail on its subnet
    {
        const std::vector<nixlLibfabricFabricId> locals = {a1, b1};
        const std::vector<nixlLibfabricFabricId> peers = {pb1, pa1};
        const auto route = routeSingle(two, two, locals, peers, 40);
        check(peersOf(route, 2, 0) == std::vector<size_t>{1} &&
                  peersOf(route, 2, 1) == std::vector<size_t>{0},
              "crossed cabling sends each local rail to the peer rail on its subnet");
    }

    // Fewer local rails than peer rails: each local rail rotates over its subnet's peer rails
    {
        const std::vector<nixlLibfabricFabricId> locals = {a1, b1};
        const std::vector<nixlLibfabricFabricId> peers = {pa1, pb1, pa2, pb2};
        const auto route = routeSingle(two, four, locals, peers, 400);
        check(peersOf(route, 2, 0) == std::vector<size_t>{0, 2} &&
                  peersOf(route, 2, 1) == std::vector<size_t>{1, 3},
              "a local rail rotates over all peer rails on its subnet");
        const auto n0 = std::count(route.begin(), route.end(), 0);
        const auto n2 = std::count(route.begin(), route.end(), 2);
        check(n0 == n2, "and uses them equally");
    }

    // Two local rails on one subnet with two peer rails there: different ones in each round
    {
        const std::vector<nixlLibfabricFabricId> locals = {a1, a2};
        const std::vector<nixlLibfabricFabricId> peers = {pb1, pa1, pa2};
        const auto route = routeSingle(two, {0, 1, 2}, locals, peers, 40);
        bool distinct = true;
        for (size_t c = 0; c + 1 < route.size(); c += 2) {
            distinct = distinct && route[c] != route[c + 1] && route[c] != 0 && route[c + 1] != 0;
        }
        check(distinct, "same-subnet local rails use different peer rails within a round");
    }

    // Striped post (round 0): stripes on one subnet go to different peer rails
    {
        const std::vector<nixlLibfabricFabricId> locals = {a1, b1, a2, b2};
        const std::vector<nixlLibfabricFabricId> peers = {pa1, pb1, pa2, pb2};
        bool identity = true;
        for (size_t i = 0; i < 4; ++i) {
            identity =
                identity && LibfabricUtils::pickPeerRail(four, i, 0, four, locals, peers) == i;
        }
        check(identity, "stripes on interleaved subnets each get their own peer rail");
    }

    // The peer rail on the subnet does not hold the buffer: old round-robin over the others
    {
        const std::vector<nixlLibfabricFabricId> locals = {a1, b1};
        const std::vector<nixlLibfabricFabricId> peers = {pa1, pb1};
        const auto route = routeSingle(two, {1}, locals, peers, 10);
        check(std::all_of(route.begin(), route.end(), [](size_t p) { return p == 1; }),
              "a buffer only off the subnet still goes to its registered rail");
    }

    // Fully routed fabric: every NIC on its own /31, so nothing matches
    {
        const std::vector<nixlLibfabricFabricId> locals = {localId("10.0.0.1", "255.255.255.254"),
                                                           localId("10.0.0.3", "255.255.255.254")};
        const std::vector<nixlLibfabricFabricId> peers = {peerId("10.0.0.9"), peerId("10.0.0.11")};
        const auto route = routeSingle(two, two, locals, peers, 20);
        bool same = true;
        for (size_t c = 0; c < route.size(); ++c) {
            same = same && route[c] == c % 2;
        }
        check(same, "with no peer rail on any subnet, posts round-robin as before");
    }

    // Malformed input: rail or peer ids shorter than the rail ids, no out-of-range read
    check(LibfabricUtils::pickPeerRail({0, 5}, 1, 0, {0, 5}, {a1}, {pb1}) == 5 &&
              LibfabricUtils::pickPeerRail({0, 1}, 0, 0, {0, 5}, {a1, b1}, {pb1}) == 0,
          "rails without ids are treated as off the subnet");
}

} // namespace

int
main() {
    NIXL_INFO << "=== Testing libfabric fabric ids and peer rail choice ===";

    testFabricIdFromEpName();
    testResolveLocalFabricMask();
    testSameFabric();
    testPickPeerRail();

    if (failures != 0) {
        NIXL_ERROR << "=== " << failures << " check(s) FAILED ===";
        return 1;
    }
    NIXL_INFO << "=== Test completed successfully! ===";
    return 0;
}
