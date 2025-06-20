/*
 * Copyright (c) 2009 IITP RAS
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Based on
 *      NS-2 ARPMEC model developed by the CMU/MONARCH group and optimized and
 *      tuned by Samir Das and Mahesh Marina, University of Cincinnati;
 *
 *      ARPMEC-UU implementation by Erik Nordström of Uppsala University
 *      https://web.archive.org/web/20100527072022/http://core.it.uu.se/core/index.php/ARPMEC-UU
 *
 * Authors: Elena Buchatskaia <borovkovaes@iitp.ru>
 *          Pavel Boyko <boyko@iitp.ru>
 */
#include "arpmec-packet.h"

#include "ns3/address-utils.h"
#include "ns3/packet.h"
#include <cmath> // For std::abs in double comparison

namespace ns3
{
namespace arpmec
{

NS_OBJECT_ENSURE_REGISTERED(TypeHeader);

TypeHeader::TypeHeader(MessageType t)
    : m_type(t),
      m_valid(true)
{
}

TypeId
TypeHeader::GetTypeId()
{
    static TypeId tid = TypeId("ns3::arpmec::TypeHeader")
                            .SetParent<Header>()
                            .SetGroupName("Arpmec")
                            .AddConstructor<TypeHeader>();
    return tid;
}

TypeId
TypeHeader::GetInstanceTypeId() const
{
    return GetTypeId();
}

uint32_t
TypeHeader::GetSerializedSize() const
{
    return 1;
}

void
TypeHeader::Serialize(Buffer::Iterator i) const
{
    i.WriteU8((uint8_t)m_type);
}

uint32_t
TypeHeader::Deserialize(Buffer::Iterator start)
{
    Buffer::Iterator i = start;
    uint8_t type = i.ReadU8();
    m_valid = true;
    switch (type)
    {
    case ARPMECTYPE_RREQ:
    case ARPMECTYPE_RREP:
    case ARPMECTYPE_RERR:
    case ARPMECTYPE_RREP_ACK:
    case ARPMEC_HELLO:
    case ARPMEC_JOIN:
    case ARPMEC_CH_NOTIFICATION:
    case ARPMEC_CLUSTER_LIST:
    case ARPMEC_DATA:
    case ARPMEC_ABDICATE: {
        m_type = (MessageType)type;
        break;
    }
    default:
        m_valid = false;
    }
    uint32_t dist = i.GetDistanceFrom(start);
    NS_ASSERT(dist == GetSerializedSize());
    return dist;
}

void
TypeHeader::Print(std::ostream& os) const
{
    switch (m_type)
    {
    case ARPMECTYPE_RREQ: {
        os << "RREQ";
        break;
    }
    case ARPMECTYPE_RREP: {
        os << "RREP";
        break;
    }
    case ARPMECTYPE_RERR: {
        os << "RERR";
        break;
    }
    case ARPMECTYPE_RREP_ACK: {
        os << "RREP_ACK";
        break;
    }
    case ARPMEC_HELLO: {
        os << "ARPMEC_HELLO";
        break;
    }
    case ARPMEC_JOIN: {
        os << "ARPMEC_JOIN";
        break;
    }
    case ARPMEC_CH_NOTIFICATION: {
        os << "ARPMEC_CH_NOTIFICATION";
        break;
    }
    case ARPMEC_CLUSTER_LIST: {
        os << "ARPMEC_CLUSTER_LIST";
        break;
    }
    case ARPMEC_DATA: {
        os << "ARPMEC_DATA";
        break;
    }
    case ARPMEC_ABDICATE: {
        os << "ARPMEC_ABDICATE";
        break;
    }
    default:
        os << "UNKNOWN_TYPE";
    }
}

bool
TypeHeader::operator==(const TypeHeader& o) const
{
    return (m_type == o.m_type && m_valid == o.m_valid);
}

std::ostream&
operator<<(std::ostream& os, const TypeHeader& h)
{
    h.Print(os);
    return os;
}

//-----------------------------------------------------------------------------
// RREQ
//-----------------------------------------------------------------------------
RreqHeader::RreqHeader(uint8_t flags,
                       uint8_t reserved,
                       uint8_t hopCount,
                       uint32_t requestID,
                       Ipv4Address dst,
                       uint32_t dstSeqNo,
                       Ipv4Address origin,
                       uint32_t originSeqNo)
    : m_flags(flags),
      m_reserved(reserved),
      m_hopCount(hopCount),
      m_requestID(requestID),
      m_dst(dst),
      m_dstSeqNo(dstSeqNo),
      m_origin(origin),
      m_originSeqNo(originSeqNo)
{
}

NS_OBJECT_ENSURE_REGISTERED(RreqHeader);

TypeId
RreqHeader::GetTypeId()
{
    static TypeId tid = TypeId("ns3::arpmec::RreqHeader")
                            .SetParent<Header>()
                            .SetGroupName("Arpmec")
                            .AddConstructor<RreqHeader>();
    return tid;
}

TypeId
RreqHeader::GetInstanceTypeId() const
{
    return GetTypeId();
}

uint32_t
RreqHeader::GetSerializedSize() const
{
    return 23;
}

void
RreqHeader::Serialize(Buffer::Iterator i) const
{
    i.WriteU8(m_flags);
    i.WriteU8(m_reserved);
    i.WriteU8(m_hopCount);
    i.WriteHtonU32(m_requestID);
    WriteTo(i, m_dst);
    i.WriteHtonU32(m_dstSeqNo);
    WriteTo(i, m_origin);
    i.WriteHtonU32(m_originSeqNo);
}

uint32_t
RreqHeader::Deserialize(Buffer::Iterator start)
{
    Buffer::Iterator i = start;
    m_flags = i.ReadU8();
    m_reserved = i.ReadU8();
    m_hopCount = i.ReadU8();
    m_requestID = i.ReadNtohU32();
    ReadFrom(i, m_dst);
    m_dstSeqNo = i.ReadNtohU32();
    ReadFrom(i, m_origin);
    m_originSeqNo = i.ReadNtohU32();

    uint32_t dist = i.GetDistanceFrom(start);
    NS_ASSERT(dist == GetSerializedSize());
    return dist;
}

void
RreqHeader::Print(std::ostream& os) const
{
    os << "RREQ ID " << m_requestID << " destination: ipv4 " << m_dst << " sequence number "
       << m_dstSeqNo << " source: ipv4 " << m_origin << " sequence number " << m_originSeqNo
       << " flags:"
       << " Gratuitous RREP " << (*this).GetGratuitousRrep() << " Destination only "
       << (*this).GetDestinationOnly() << " Unknown sequence number " << (*this).GetUnknownSeqno();
}

std::ostream&
operator<<(std::ostream& os, const RreqHeader& h)
{
    h.Print(os);
    return os;
}

void
RreqHeader::SetGratuitousRrep(bool f)
{
    if (f)
    {
        m_flags |= (1 << 5);
    }
    else
    {
        m_flags &= ~(1 << 5);
    }
}

bool
RreqHeader::GetGratuitousRrep() const
{
    return (m_flags & (1 << 5));
}

void
RreqHeader::SetDestinationOnly(bool f)
{
    if (f)
    {
        m_flags |= (1 << 4);
    }
    else
    {
        m_flags &= ~(1 << 4);
    }
}

bool
RreqHeader::GetDestinationOnly() const
{
    return (m_flags & (1 << 4));
}

void
RreqHeader::SetUnknownSeqno(bool f)
{
    if (f)
    {
        m_flags |= (1 << 3);
    }
    else
    {
        m_flags &= ~(1 << 3);
    }
}

bool
RreqHeader::GetUnknownSeqno() const
{
    return (m_flags & (1 << 3));
}

bool
RreqHeader::operator==(const RreqHeader& o) const
{
    return (m_flags == o.m_flags && m_reserved == o.m_reserved && m_hopCount == o.m_hopCount &&
            m_requestID == o.m_requestID && m_dst == o.m_dst && m_dstSeqNo == o.m_dstSeqNo &&
            m_origin == o.m_origin && m_originSeqNo == o.m_originSeqNo);
}

//-----------------------------------------------------------------------------
// RREP
//-----------------------------------------------------------------------------

RrepHeader::RrepHeader(uint8_t prefixSize,
                       uint8_t hopCount,
                       Ipv4Address dst,
                       uint32_t dstSeqNo,
                       Ipv4Address origin,
                       Time lifeTime)
    : m_flags(0),
      m_prefixSize(prefixSize),
      m_hopCount(hopCount),
      m_dst(dst),
      m_dstSeqNo(dstSeqNo),
      m_origin(origin)
{
    m_lifeTime = uint32_t(lifeTime.GetMilliSeconds());
}

NS_OBJECT_ENSURE_REGISTERED(RrepHeader);

TypeId
RrepHeader::GetTypeId()
{
    static TypeId tid = TypeId("ns3::arpmec::RrepHeader")
                            .SetParent<Header>()
                            .SetGroupName("Arpmec")
                            .AddConstructor<RrepHeader>();
    return tid;
}

TypeId
RrepHeader::GetInstanceTypeId() const
{
    return GetTypeId();
}

uint32_t
RrepHeader::GetSerializedSize() const
{
    return 19;
}

void
RrepHeader::Serialize(Buffer::Iterator i) const
{
    i.WriteU8(m_flags);
    i.WriteU8(m_prefixSize);
    i.WriteU8(m_hopCount);
    WriteTo(i, m_dst);
    i.WriteHtonU32(m_dstSeqNo);
    WriteTo(i, m_origin);
    i.WriteHtonU32(m_lifeTime);
}

uint32_t
RrepHeader::Deserialize(Buffer::Iterator start)
{
    Buffer::Iterator i = start;

    m_flags = i.ReadU8();
    m_prefixSize = i.ReadU8();
    m_hopCount = i.ReadU8();
    ReadFrom(i, m_dst);
    m_dstSeqNo = i.ReadNtohU32();
    ReadFrom(i, m_origin);
    m_lifeTime = i.ReadNtohU32();

    uint32_t dist = i.GetDistanceFrom(start);
    NS_ASSERT(dist == GetSerializedSize());
    return dist;
}

void
RrepHeader::Print(std::ostream& os) const
{
    os << "destination: ipv4 " << m_dst << " sequence number " << m_dstSeqNo;
    if (m_prefixSize != 0)
    {
        os << " prefix size " << m_prefixSize;
    }
    os << " source ipv4 " << m_origin << " lifetime " << m_lifeTime
       << " acknowledgment required flag " << (*this).GetAckRequired();
}

void
RrepHeader::SetLifeTime(Time t)
{
    m_lifeTime = t.GetMilliSeconds();
}

Time
RrepHeader::GetLifeTime() const
{
    Time t(MilliSeconds(m_lifeTime));
    return t;
}

void
RrepHeader::SetAckRequired(bool f)
{
    if (f)
    {
        m_flags |= (1 << 6);
    }
    else
    {
        m_flags &= ~(1 << 6);
    }
}

bool
RrepHeader::GetAckRequired() const
{
    return (m_flags & (1 << 6));
}

void
RrepHeader::SetPrefixSize(uint8_t sz)
{
    m_prefixSize = sz;
}

uint8_t
RrepHeader::GetPrefixSize() const
{
    return m_prefixSize;
}

bool
RrepHeader::operator==(const RrepHeader& o) const
{
    return (m_flags == o.m_flags && m_prefixSize == o.m_prefixSize && m_hopCount == o.m_hopCount &&
            m_dst == o.m_dst && m_dstSeqNo == o.m_dstSeqNo && m_origin == o.m_origin &&
            m_lifeTime == o.m_lifeTime);
}

void
RrepHeader::SetHello(Ipv4Address origin, uint32_t srcSeqNo, Time lifetime)
{
    m_flags = 0;
    m_prefixSize = 0;
    m_hopCount = 0;
    m_dst = origin;
    m_dstSeqNo = srcSeqNo;
    m_origin = origin;
    m_lifeTime = lifetime.GetMilliSeconds();
}

std::ostream&
operator<<(std::ostream& os, const RrepHeader& h)
{
    h.Print(os);
    return os;
}

//-----------------------------------------------------------------------------
// RREP-ACK
//-----------------------------------------------------------------------------

RrepAckHeader::RrepAckHeader()
    : m_reserved(0)
{
}

NS_OBJECT_ENSURE_REGISTERED(RrepAckHeader);

TypeId
RrepAckHeader::GetTypeId()
{
    static TypeId tid = TypeId("ns3::arpmec::RrepAckHeader")
                            .SetParent<Header>()
                            .SetGroupName("Arpmec")
                            .AddConstructor<RrepAckHeader>();
    return tid;
}

TypeId
RrepAckHeader::GetInstanceTypeId() const
{
    return GetTypeId();
}

uint32_t
RrepAckHeader::GetSerializedSize() const
{
    return 1;
}

void
RrepAckHeader::Serialize(Buffer::Iterator i) const
{
    i.WriteU8(m_reserved);
}

uint32_t
RrepAckHeader::Deserialize(Buffer::Iterator start)
{
    Buffer::Iterator i = start;
    m_reserved = i.ReadU8();
    uint32_t dist = i.GetDistanceFrom(start);
    NS_ASSERT(dist == GetSerializedSize());
    return dist;
}

void
RrepAckHeader::Print(std::ostream& os) const
{
}

bool
RrepAckHeader::operator==(const RrepAckHeader& o) const
{
    return m_reserved == o.m_reserved;
}

std::ostream&
operator<<(std::ostream& os, const RrepAckHeader& h)
{
    h.Print(os);
    return os;
}

//-----------------------------------------------------------------------------
// RERR
//-----------------------------------------------------------------------------
RerrHeader::RerrHeader()
    : m_flag(0),
      m_reserved(0)
{
}

NS_OBJECT_ENSURE_REGISTERED(RerrHeader);

TypeId
RerrHeader::GetTypeId()
{
    static TypeId tid = TypeId("ns3::arpmec::RerrHeader")
                            .SetParent<Header>()
                            .SetGroupName("Arpmec")
                            .AddConstructor<RerrHeader>();
    return tid;
}

TypeId
RerrHeader::GetInstanceTypeId() const
{
    return GetTypeId();
}

uint32_t
RerrHeader::GetSerializedSize() const
{
    return (3 + 8 * GetDestCount());
}

void
RerrHeader::Serialize(Buffer::Iterator i) const
{
    i.WriteU8(m_flag);
    i.WriteU8(m_reserved);
    i.WriteU8(GetDestCount());
    for (auto j = m_unreachableDstSeqNo.begin(); j != m_unreachableDstSeqNo.end(); ++j)
    {
        WriteTo(i, (*j).first);
        i.WriteHtonU32((*j).second);
    }
}

uint32_t
RerrHeader::Deserialize(Buffer::Iterator start)
{
    Buffer::Iterator i = start;
    m_flag = i.ReadU8();
    m_reserved = i.ReadU8();
    uint8_t dest = i.ReadU8();
    m_unreachableDstSeqNo.clear();
    Ipv4Address address;
    uint32_t seqNo;
    for (uint8_t k = 0; k < dest; ++k)
    {
        ReadFrom(i, address);
        seqNo = i.ReadNtohU32();
        m_unreachableDstSeqNo.insert(std::make_pair(address, seqNo));
    }

    uint32_t dist = i.GetDistanceFrom(start);
    NS_ASSERT(dist == GetSerializedSize());
    return dist;
}

void
RerrHeader::Print(std::ostream& os) const
{
    os << "Unreachable destination (ipv4 address, seq. number):";
    for (auto j = m_unreachableDstSeqNo.begin(); j != m_unreachableDstSeqNo.end(); ++j)
    {
        os << (*j).first << ", " << (*j).second;
    }
    os << "No delete flag " << (*this).GetNoDelete();
}

void
RerrHeader::SetNoDelete(bool f)
{
    if (f)
    {
        m_flag |= (1 << 0);
    }
    else
    {
        m_flag &= ~(1 << 0);
    }
}

bool
RerrHeader::GetNoDelete() const
{
    return (m_flag & (1 << 0));
}

bool
RerrHeader::AddUnDestination(Ipv4Address dst, uint32_t seqNo)
{
    if (m_unreachableDstSeqNo.find(dst) != m_unreachableDstSeqNo.end())
    {
        return true;
    }

    NS_ASSERT(GetDestCount() < 255); // can't support more than 255 destinations in single RERR
    m_unreachableDstSeqNo.insert(std::make_pair(dst, seqNo));
    return true;
}

bool
RerrHeader::RemoveUnDestination(std::pair<Ipv4Address, uint32_t>& un)
{
    if (m_unreachableDstSeqNo.empty())
    {
        return false;
    }
    auto i = m_unreachableDstSeqNo.begin();
    un = *i;
    m_unreachableDstSeqNo.erase(i);
    return true;
}

void
RerrHeader::Clear()
{
    m_unreachableDstSeqNo.clear();
    m_flag = 0;
    m_reserved = 0;
}

bool
RerrHeader::operator==(const RerrHeader& o) const
{
    if (m_flag != o.m_flag || m_reserved != o.m_reserved || GetDestCount() != o.GetDestCount())
    {
        return false;
    }

    auto j = m_unreachableDstSeqNo.begin();
    auto k = o.m_unreachableDstSeqNo.begin();
    for (uint8_t i = 0; i < GetDestCount(); ++i)
    {
        if ((j->first != k->first) || (j->second != k->second))
        {
            return false;
        }

        j++;
        k++;
    }
    return true;
}

std::ostream&
operator<<(std::ostream& os, const RerrHeader& h)
{
    h.Print(os);
    return os;
}

//-----------------------------------------------------------------------------
// ARPMEC_HELLO
//-----------------------------------------------------------------------------
NS_OBJECT_ENSURE_REGISTERED(ArpmecHelloHeader);

ArpmecHelloHeader::ArpmecHelloHeader() : m_nodeId(0), m_channelId(0), m_sequenceNumber(0), m_rssi(-100.0), m_pdr(0.0), m_timestamp(0) {}

TypeId
ArpmecHelloHeader::GetTypeId()
{
    static TypeId tid = TypeId("ns3::arpmec::ArpmecHelloHeader")
        .SetParent<Header>()
        .SetGroupName("Arpmec")
        .AddConstructor<ArpmecHelloHeader>();
    return tid;
}

TypeId
ArpmecHelloHeader::GetInstanceTypeId() const
{
    return GetTypeId();
}

uint32_t
ArpmecHelloHeader::GetSerializedSize() const
{
    return 33; // 4 bytes nodeId + 1 byte channelId + 4 bytes sequenceNumber + 8 bytes rssi + 8 bytes pdr + 8 bytes timestamp
}

void
ArpmecHelloHeader::Serialize(Buffer::Iterator start) const
{
     start.WriteHtonU32(m_nodeId);
    start.WriteU8(m_channelId);
    start.WriteHtonU32(m_sequenceNumber);

    // Fix: Handle negative RSSI values properly
    int64_t rssiInt = static_cast<int64_t>(m_rssi * 1000000);
    start.WriteHtonU64(static_cast<uint64_t>(rssiInt));

    start.WriteHtonU64(static_cast<uint64_t>(m_pdr * 1000000));
    start.WriteHtonU64(m_timestamp);
}

uint32_t
ArpmecHelloHeader::Deserialize(Buffer::Iterator start)
{
    Buffer::Iterator i = start;
    m_nodeId = i.ReadNtohU32();
    m_channelId = i.ReadU8();
    m_sequenceNumber = i.ReadNtohU32();

    // Fix: Handle negative RSSI values properly
    uint64_t rssiRaw = i.ReadNtohU64();
    int64_t rssiInt = static_cast<int64_t>(rssiRaw);
    m_rssi = static_cast<double>(rssiInt) / 1000000.0;

    m_pdr = static_cast<double>(i.ReadNtohU64()) / 1000000.0;
    m_timestamp = i.ReadNtohU64();

    uint32_t dist = i.GetDistanceFrom(start);
    NS_ASSERT(dist == GetSerializedSize());
    return dist;
}

void
ArpmecHelloHeader::Print(std::ostream& os) const
{
    os << "HELLO from node " << m_nodeId << " on channel "
       << static_cast<int>(m_channelId)
       << " with sequence number " << m_sequenceNumber
       << ", RSSI: " << m_rssi << " dBm"
       << ", PDR: " << m_pdr
       << ", timestamp: " << m_timestamp << " μs";
}

bool
ArpmecHelloHeader::operator==(const ArpmecHelloHeader& o) const
{
    return (m_nodeId == o.m_nodeId &&
            m_channelId == o.m_channelId &&
            m_sequenceNumber == o.m_sequenceNumber &&
            std::abs(m_rssi - o.m_rssi) < 0.001 &&      // Small tolerance for double comparison
            std::abs(m_pdr - o.m_pdr) < 0.001 &&        // Small tolerance for double comparison
            m_timestamp == o.m_timestamp);
}

std::ostream&
operator<<(std::ostream& os, const ArpmecHelloHeader& h)
{
    h.Print(os);
    return os;
}

// Other header implementations (TypeHeader, ArpmecJoinHeader, etc.) remain unchanged
// ... (include implementations as previously provided)

//-----------------------------------------------------------------------------
// ARPMEC_JOIN
//-----------------------------------------------------------------------------

NS_OBJECT_ENSURE_REGISTERED(ArpmecJoinHeader);

ArpmecJoinHeader::ArpmecJoinHeader() : m_nodeId(0), m_chId(0) {}

TypeId
ArpmecJoinHeader::GetTypeId()
{
    static TypeId tid = TypeId("ns3::arpmec::ArpmecJoinHeader")
        .SetParent<Header>()
        .SetGroupName("Arpmec")
        .AddConstructor<ArpmecJoinHeader>();
    return tid;
}

TypeId
ArpmecJoinHeader::GetInstanceTypeId() const
{
    return GetTypeId();
}

uint32_t
ArpmecJoinHeader::GetSerializedSize() const
{
    return 8; // 4 bytes pour nodeId, 4 bytes pour chId
}

void
ArpmecJoinHeader::Serialize(Buffer::Iterator start) const
{
    start.WriteHtonU32(m_nodeId);
    start.WriteHtonU32(m_chId);
}

uint32_t
ArpmecJoinHeader::Deserialize(Buffer::Iterator start)
{
    Buffer::Iterator i = start;
    m_nodeId = i.ReadNtohU32();
    m_chId = i.ReadNtohU32();
    uint32_t dist = i.GetDistanceFrom(start);
    NS_ASSERT(dist == GetSerializedSize());
    return dist;
}

void
ArpmecJoinHeader::Print(std::ostream& os) const
{
    os << "JOIN from node " << m_nodeId << " to CH " << m_chId;
}

bool
ArpmecJoinHeader::operator==(const ArpmecJoinHeader& o) const
{
    return (m_nodeId == o.m_nodeId && m_chId == o.m_chId);
}

std::ostream&
operator<<(std::ostream& os, const ArpmecJoinHeader& h)
{
    h.Print(os);
    return os;
}

//-----------------------------------------------------------------------------
// ARPMEC_CH_NOTIFICATION
//-----------------------------------------------------------------------------
NS_OBJECT_ENSURE_REGISTERED(ArpmecChNotificationHeader);

ArpmecChNotificationHeader::ArpmecChNotificationHeader() : m_chId(0) {}

TypeId
ArpmecChNotificationHeader::GetTypeId()
{
    static TypeId tid = TypeId("ns3::arpmec::ArpmecChNotificationHeader")
        .SetParent<Header>()
        .SetGroupName("Arpmec")
        .AddConstructor<ArpmecChNotificationHeader>();
    return tid;
}

TypeId
ArpmecChNotificationHeader::GetInstanceTypeId() const
{
    return GetTypeId();
}

uint32_t
ArpmecChNotificationHeader::GetSerializedSize() const
{
    return 8 + m_clusterMembers.size() * 4; // 4 bytes pour chId, 4 bytes pour taille de la liste, 4 bytes par membre
}

void
ArpmecChNotificationHeader::Serialize(Buffer::Iterator start) const
{
    start.WriteHtonU32(m_chId);
    start.WriteHtonU32(m_clusterMembers.size());
    for (const auto& member : m_clusterMembers)
    {
        start.WriteHtonU32(member);
    }
}

uint32_t
ArpmecChNotificationHeader::Deserialize(Buffer::Iterator start)
{
    Buffer::Iterator i = start;
    m_chId = i.ReadNtohU32();
    uint32_t size = i.ReadNtohU32();
    m_clusterMembers.clear();
    for (uint32_t j = 0; j < size; ++j)
    {
        m_clusterMembers.push_back(i.ReadNtohU32());
    }
    uint32_t dist = i.GetDistanceFrom(start);
    NS_ASSERT(dist == GetSerializedSize());
    return dist;
}

void
ArpmecChNotificationHeader::Print(std::ostream& os) const
{
    os << "CH_NOTIFICATION from CH " << m_chId << " with members: ";
    for (const auto& member : m_clusterMembers)
    {
        os << member << " ";
    }
}

bool
ArpmecChNotificationHeader::operator==(const ArpmecChNotificationHeader& o) const
{
    return (m_chId == o.m_chId && m_clusterMembers == o.m_clusterMembers);
}

std::ostream&
operator<<(std::ostream& os, const ArpmecChNotificationHeader& h)
{
    h.Print(os);
    return os;
}

//-----------------------------------------------------------------------------
// ARPMEC_CLUSTER_LIST
//-----------------------------------------------------------------------------

NS_OBJECT_ENSURE_REGISTERED(ArpmecClusterListHeader);

ArpmecClusterListHeader::ArpmecClusterListHeader() {}

TypeId
ArpmecClusterListHeader::GetTypeId()
{
    static TypeId tid = TypeId("ns3::arpmec::ArpmecClusterListHeader")
        .SetParent<Header>()
        .SetGroupName("Arpmec")
        .AddConstructor<ArpmecClusterListHeader>();
    return tid;
}

TypeId
ArpmecClusterListHeader::GetInstanceTypeId() const
{
    return GetTypeId();
}

uint32_t
ArpmecClusterListHeader::GetSerializedSize() const
{
    uint32_t size = 4; // 4 bytes pour la taille de la liste des clusters
    for (const auto& cluster : m_clusters)
    {
        size += 8; // 4 bytes pour chId, 4 bytes pour taille des membres
        size += cluster.second.size() * 4; // 4 bytes par membre
    }
    return size;
}

void
ArpmecClusterListHeader::Serialize(Buffer::Iterator start) const
{
    start.WriteHtonU32(m_clusters.size());
    for (const auto& cluster : m_clusters)
    {
        start.WriteHtonU32(cluster.first);
        start.WriteHtonU32(cluster.second.size());
        for (const auto& member : cluster.second)
        {
            start.WriteHtonU32(member);
        }
    }
}

uint32_t
ArpmecClusterListHeader::Deserialize(Buffer::Iterator start)
{
    Buffer::Iterator i = start;
    uint32_t numClusters = i.ReadNtohU32();
    m_clusters.clear();
    for (uint32_t j = 0; j < numClusters; ++j)
    {
        uint32_t chId = i.ReadNtohU32();
        uint32_t numMembers = i.ReadNtohU32();
        std::vector<uint32_t> members;
        for (uint32_t k = 0; k < numMembers; ++k)
        {
            members.push_back(i.ReadNtohU32());
        }
        m_clusters.emplace_back(chId, members);
    }
    uint32_t dist = i.GetDistanceFrom(start);
    NS_ASSERT(dist == GetSerializedSize());
    return dist;
}

void
ArpmecClusterListHeader::Print(std::ostream& os) const
{
    os << "CLUSTER_LIST with " << m_clusters.size() << " clusters: ";
    for (const auto& cluster : m_clusters)
    {
        os << "CH " << cluster.first << " members: ";
        for (const auto& member : cluster.second)
        {
            os << member << " ";
        }
    }
}

bool
ArpmecClusterListHeader::operator==(const ArpmecClusterListHeader& o) const
{
    return (m_clusters == o.m_clusters);
}

std::ostream&
operator<<(std::ostream& os, const ArpmecClusterListHeader& h)
{
    h.Print(os);
    return os;
}

//-----------------------------------------------------------------------------
// ARPMEC_DATA
//-----------------------------------------------------------------------------

NS_OBJECT_ENSURE_REGISTERED(ArpmecDataHeader);

ArpmecDataHeader::ArpmecDataHeader() : m_sourceId(0), m_destId(0), m_chId(0) {}

TypeId
ArpmecDataHeader::GetTypeId()
{
    static TypeId tid = TypeId("ns3::arpmec::ArpmecDataHeader")
        .SetParent<Header>()
        .SetGroupName("Arpmec")
        .AddConstructor<ArpmecDataHeader>();
    return tid;
}

TypeId
ArpmecDataHeader::GetInstanceTypeId() const
{
    return GetTypeId();
}

uint32_t
ArpmecDataHeader::GetSerializedSize() const
{
    return 12; // 4 bytes pour sourceId, 4 bytes pour destId, 4 bytes pour chId
}

void
ArpmecDataHeader::Serialize(Buffer::Iterator start) const
{
    start.WriteHtonU32(m_sourceId);
    start.WriteHtonU32(m_destId);
    start.WriteHtonU32(m_chId);
}

uint32_t
ArpmecDataHeader::Deserialize(Buffer::Iterator start)
{
    Buffer::Iterator i = start;
    m_sourceId = i.ReadNtohU32();
    m_destId = i.ReadNtohU32();
    m_chId = i.ReadNtohU32();
    uint32_t dist = i.GetDistanceFrom(start);
    NS_ASSERT(dist == GetSerializedSize());
    return dist;
}

void
ArpmecDataHeader::Print(std::ostream& os) const
{
    os << "DATA from source " << m_sourceId << " to destination " << m_destId << " via CH " << m_chId;
}

bool
ArpmecDataHeader::operator==(const ArpmecDataHeader& o) const
{
    return (m_sourceId == o.m_sourceId && m_destId == o.m_destId && m_chId == o.m_chId);
}

std::ostream&
operator<<(std::ostream& os, const ArpmecDataHeader& h)
{
    h.Print(os);
    return os;
}
//-----------------------------------------------------------------------------
// ARPMEC_ABDICATE
//-----------------------------------------------------------------------------

NS_OBJECT_ENSURE_REGISTERED(ArpmecAbdicateHeader);

ArpmecAbdicateHeader::ArpmecAbdicateHeader() : m_chId(0) {}

TypeId
ArpmecAbdicateHeader::GetTypeId()
{
    static TypeId tid = TypeId("ns3::arpmec::ArpmecAbdicateHeader")
        .SetParent<Header>()
        .SetGroupName("Arpmec")
        .AddConstructor<ArpmecAbdicateHeader>();
    return tid;
}

TypeId
ArpmecAbdicateHeader::GetInstanceTypeId() const
{
    return GetTypeId();
}

uint32_t
ArpmecAbdicateHeader::GetSerializedSize() const
{
    return 4; // 4 bytes pour chId
}

void
ArpmecAbdicateHeader::Serialize(Buffer::Iterator start) const
{
    start.WriteHtonU32(m_chId);
}

uint32_t
ArpmecAbdicateHeader::Deserialize(Buffer::Iterator start)
{
    Buffer::Iterator i = start;
    m_chId = i.ReadNtohU32();
    uint32_t dist = i.GetDistanceFrom(start);
    NS_ASSERT(dist == GetSerializedSize());
    return dist;
}

void
ArpmecAbdicateHeader::Print(std::ostream& os) const
{
    os << "ABDICATE from CH " << m_chId;
}

bool
ArpmecAbdicateHeader::operator==(const ArpmecAbdicateHeader& o) const
{
    return (m_chId == o.m_chId);
}

std::ostream&
operator<<(std::ostream& os, const ArpmecAbdicateHeader& h)
{
    h.Print(os);
    return os;
}


} // namespace arpmec
} // namespace ns3
