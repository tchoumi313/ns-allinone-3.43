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

#ifndef ARPMEC_ID_CACHE_H
#define ARPMEC_ID_CACHE_H

#include "ns3/ipv4-address.h"
#include "ns3/simulator.h"

#include <vector>

namespace ns3
{
namespace arpmec
{
/**
 * \ingroup arpmec
 *
 * \brief Unique packets identification cache used for simple duplicate detection.
 */
class IdCache
{
  public:
    /**
     * constructor
     * \param lifetime the lifetime for added entries
     */
    IdCache(Time lifetime)
        : m_lifetime(lifetime)
    {
    }

    /**
     * Check that entry (addr, id) exists in cache. Add entry, if it doesn't exist.
     * \param addr the IP address
     * \param id the cache entry ID
     * \returns true if the pair exists
     */
    bool IsDuplicate(Ipv4Address addr, uint32_t id);
    /// Remove all expired entries
    void Purge();
    /**
     * \returns number of entries in cache
     */
    uint32_t GetSize();

    /**
     * Set lifetime for future added entries.
     * \param lifetime the lifetime for entries
     */
    void SetLifetime(Time lifetime)
    {
        m_lifetime = lifetime;
    }

    /**
     * Return lifetime for existing entries in cache
     * \returns the lifetime
     */
    Time GetLifeTime() const
    {
        return m_lifetime;
    }

  private:
    /// Unique packet ID
    struct UniqueId
    {
        /// ID is supposed to be unique in single address context (e.g. sender address)
        Ipv4Address m_context;
        /// The id
        uint32_t m_id;
        /// When record will expire
        Time m_expire;
    };

    /**
     * \brief IsExpired structure
     */
    struct IsExpired
    {
        /**
         * \brief Check if the entry is expired
         *
         * \param u UniqueId entry
         * \return true if expired, false otherwise
         */
        bool operator()(const UniqueId& u) const
        {
            return (u.m_expire < Simulator::Now());
        }
    };

    /// Already seen IDs
    std::vector<UniqueId> m_idCache;
    /// Default lifetime for ID records
    Time m_lifetime;
};

} // namespace arpmec
} // namespace ns3

#endif /* ARPMEC_ID_CACHE_H */
