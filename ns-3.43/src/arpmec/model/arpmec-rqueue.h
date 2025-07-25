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
#ifndef ARPMEC_RQUEUE_H
#define ARPMEC_RQUEUE_H

#include "ns3/ipv4-routing-protocol.h"
#include "ns3/simulator.h"

#include <vector>

namespace ns3
{
namespace arpmec
{

/**
 * \ingroup arpmec
 * \brief ARPMEC Queue Entry
 */
class QueueEntry
{
  public:
    /// IPv4 routing unicast forward callback typedef
    typedef Ipv4RoutingProtocol::UnicastForwardCallback UnicastForwardCallback;
    /// IPv4 routing error callback typedef
    typedef Ipv4RoutingProtocol::ErrorCallback ErrorCallback;

    /**
     * constructor
     *
     * \param pa the packet to add to the queue
     * \param h the Ipv4Header
     * \param ucb the UnicastForwardCallback function
     * \param ecb the ErrorCallback function
     * \param exp the expiration time
     */
    QueueEntry(Ptr<const Packet> pa = nullptr,
               const Ipv4Header& h = Ipv4Header(),
               UnicastForwardCallback ucb = UnicastForwardCallback(),
               ErrorCallback ecb = ErrorCallback(),
               Time exp = Simulator::Now())
        : m_packet(pa),
          m_header(h),
          m_ucb(ucb),
          m_ecb(ecb),
          m_expire(exp + Simulator::Now())
    {
    }

    /**
     * \brief Compare queue entries
     * \param o QueueEntry to compare
     * \return true if equal
     */
    bool operator==(const QueueEntry& o) const
    {
        return ((m_packet == o.m_packet) &&
                (m_header.GetDestination() == o.m_header.GetDestination()) &&
                (m_expire == o.m_expire));
    }

    // Fields
    /**
     * Get unicast forward callback
     * \returns unicast callback
     */
    UnicastForwardCallback GetUnicastForwardCallback() const
    {
        return m_ucb;
    }

    /**
     * Set unicast forward callback
     * \param ucb The unicast callback
     */
    void SetUnicastForwardCallback(UnicastForwardCallback ucb)
    {
        m_ucb = ucb;
    }

    /**
     * Get error callback
     * \returns the error callback
     */
    ErrorCallback GetErrorCallback() const
    {
        return m_ecb;
    }

    /**
     * Set error callback
     * \param ecb The error callback
     */
    void SetErrorCallback(ErrorCallback ecb)
    {
        m_ecb = ecb;
    }

    /**
     * Get packet from entry
     * \returns the packet
     */
    Ptr<const Packet> GetPacket() const
    {
        return m_packet;
    }

    /**
     * Set packet in entry
     * \param p The packet
     */
    void SetPacket(Ptr<const Packet> p)
    {
        m_packet = p;
    }

    /**
     * Get IPv4 header
     * \returns the IPv4 header
     */
    Ipv4Header GetIpv4Header() const
    {
        return m_header;
    }

    /**
     * Set IPv4 header
     * \param h the IPv4 header
     */
    void SetIpv4Header(Ipv4Header h)
    {
        m_header = h;
    }

    /**
     * Set expire time
     * \param exp The expiration time
     */
    void SetExpireTime(Time exp)
    {
        m_expire = exp + Simulator::Now();
    }

    /**
     * Get expire time
     * \returns the expiration time
     */
    Time GetExpireTime() const
    {
        return m_expire - Simulator::Now();
    }

  private:
    /// Data packet
    Ptr<const Packet> m_packet;
    /// IP header
    Ipv4Header m_header;
    /// Unicast forward callback
    UnicastForwardCallback m_ucb;
    /// Error callback
    ErrorCallback m_ecb;
    /// Expire time for queue entry
    Time m_expire;
};

/**
 * \ingroup arpmec
 * \brief ARPMEC route request queue
 *
 * Since ARPMEC is an on demand routing we queue requests while looking for route.
 */
class RequestQueue
{
  public:
    /**
     * constructor
     *
     * \param maxLen the maximum length
     * \param routeToQueueTimeout the route to queue timeout
     */
    RequestQueue(uint32_t maxLen, Time routeToQueueTimeout)
        : m_maxLen(maxLen),
          m_queueTimeout(routeToQueueTimeout)
    {
    }

    /**
     * Push entry in queue, if there is no entry with the same packet and destination address in
     * queue.
     * \param entry the queue entry
     * \returns true if the entry is queued
     */
    bool Enqueue(QueueEntry& entry);
    /**
     * Return first found (the earliest) entry for given destination
     *
     * \param dst the destination IP address
     * \param entry the queue entry
     * \returns true if the entry is dequeued
     */
    bool Dequeue(Ipv4Address dst, QueueEntry& entry);
    /**
     * Remove all packets with destination IP address dst
     * \param dst the destination IP address
     */
    void DropPacketWithDst(Ipv4Address dst);
    /**
     * Finds whether a packet with destination dst exists in the queue
     *
     * \param dst the destination IP address
     * \returns true if an entry with the IP address is found
     */
    bool Find(Ipv4Address dst);
    /**
     * \returns the number of entries
     */
    uint32_t GetSize();

    // Fields
    /**
     * Get maximum queue length
     * \returns the maximum queue length
     */
    uint32_t GetMaxQueueLen() const
    {
        return m_maxLen;
    }

    /**
     * Set maximum queue length
     * \param len The maximum queue length
     */
    void SetMaxQueueLen(uint32_t len)
    {
        m_maxLen = len;
    }

    /**
     * Get queue timeout
     * \returns the queue timeout
     */
    Time GetQueueTimeout() const
    {
        return m_queueTimeout;
    }

    /**
     * Set queue timeout
     * \param t The queue timeout
     */
    void SetQueueTimeout(Time t)
    {
        m_queueTimeout = t;
    }

  private:
    /// The queue
    std::vector<QueueEntry> m_queue;
    /// Remove all expired entries
    void Purge();
    /**
     * Notify that packet is dropped from queue by timeout
     * \param en the queue entry to drop
     * \param reason the reason to drop the entry
     */
    void Drop(QueueEntry en, std::string reason);
    /// The maximum number of packets that we allow a routing protocol to buffer.
    uint32_t m_maxLen;
    /// The maximum period of time that a routing protocol is allowed to buffer a packet for,
    /// seconds.
    Time m_queueTimeout;
};

} // namespace arpmec
} // namespace ns3

#endif /* ARPMEC_RQUEUE_H */
