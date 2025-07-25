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

#ifndef ARPMECNEIGHBOR_H
#define ARPMECNEIGHBOR_H

#include "ns3/arp-cache.h"
#include "ns3/callback.h"
#include "ns3/ipv4-address.h"
#include "ns3/simulator.h"
#include "ns3/timer.h"

#include <vector>

namespace ns3
{

class WifiMacHeader;

namespace arpmec
{

class RoutingProtocol;

/**
 * \ingroup arpmec
 * \brief maintain list of active neighbors
 */
class Neighbors
{
  public:
    /**
     * constructor
     * \param delay the delay time for purging the list of neighbors
     */
    Neighbors(Time delay);

    /// Neighbor description
    struct Neighbor
    {
        /// Neighbor IPv4 address
        Ipv4Address m_neighborAddress;
        /// Neighbor MAC address
        Mac48Address m_hardwareAddress;
        /// Neighbor expire time
        Time m_expireTime;
        /// Neighbor close indicator
        bool close;

        /**
         * \brief Neighbor structure constructor
         *
         * \param ip Ipv4Address entry
         * \param mac Mac48Address entry
         * \param t Time expire time
         */
        Neighbor(Ipv4Address ip, Mac48Address mac, Time t)
            : m_neighborAddress(ip),
              m_hardwareAddress(mac),
              m_expireTime(t),
              close(false)
        {
        }
    };

    /**
     * Return expire time for neighbor node with address addr, if exists, else return 0.
     * \param addr the IP address of the neighbor node
     * \returns the expire time for the neighbor node
     */
    Time GetExpireTime(Ipv4Address addr);
    /**
     * Check that node with address addr is neighbor
     * \param addr the IP address to check
     * \returns true if the node with IP address is a neighbor
     */
    bool IsNeighbor(Ipv4Address addr);
    /**
     * Update expire time for entry with address addr, if it exists, else add new entry
     * \param addr the IP address to check
     * \param expire the expire time for the address
     */
    void Update(Ipv4Address addr, Time expire);
    /// Remove all expired entries
    void Purge();
    /// Schedule m_ntimer.
    void ScheduleTimer();

    /// Remove all entries
    void Clear()
    {
        m_nb.clear();
    }

    /**
     * Add ARP cache to be used to allow layer 2 notifications processing
     * \param a pointer to the ARP cache to add
     */
    void AddArpCache(Ptr<ArpCache> a);
    /**
     * Don't use given ARP cache any more (interface is down)
     * \param a pointer to the ARP cache to delete
     */
    void DelArpCache(Ptr<ArpCache> a);

    /**
     * Get callback to ProcessTxError
     * \returns the callback function
     */
    Callback<void, const WifiMacHeader&> GetTxErrorCallback() const
    {
        return m_txErrorCallback;
    }

    /**
     * Set link failure callback
     * \param cb the callback function
     */
    void SetCallback(Callback<void, Ipv4Address> cb)
    {
        m_handleLinkFailure = cb;
    }

    /**
     * Get link failure callback
     * \returns the link failure callback
     */
    Callback<void, Ipv4Address> GetCallback() const
    {
        return m_handleLinkFailure;
    }

  private:
    /// link failure callback
    Callback<void, Ipv4Address> m_handleLinkFailure;
    /// TX error callback
    Callback<void, const WifiMacHeader&> m_txErrorCallback;
    /// Timer for neighbor's list. Schedule Purge().
    Timer m_ntimer;
    /// vector of entries
    std::vector<Neighbor> m_nb;
    /// list of ARP cached to be used for layer 2 notifications processing
    std::vector<Ptr<ArpCache>> m_arp;

    /**
     * Find MAC address by IP using list of ARP caches
     *
     * \param addr the IP address to lookup
     * \returns the MAC address for the IP address
     */
    Mac48Address LookupMacAddress(Ipv4Address addr);
    /**
     * Process layer 2 TX error notification
     * \param hdr header of the packet
     */
    void ProcessTxError(const WifiMacHeader& hdr);
};

} // namespace arpmec
} // namespace ns3

#endif /* ARPMECNEIGHBOR_H */
