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
#ifndef ARPMECPACKET_H
#define ARPMECPACKET_H

#include "ns3/enum.h"
#include "ns3/header.h"
#include "ns3/ipv4-address.h"
#include "ns3/nstime.h"
#include <utility> // For std::pair


#include <iostream>
#include <map>
#include <vector>
#include "ns3/buffer.h"

namespace ns3
{
namespace arpmec
{

/**
 * \ingroup arpmec
 * \brief MessageType enumeration
 */
enum MessageType
{
    ARPMECTYPE_RREQ = 1,    //!< ARPMECTYPE_RREQ
    ARPMECTYPE_RREP = 2,    //!< ARPMECTYPE_RREP
    ARPMECTYPE_RERR = 3,    //!< ARPMECTYPE_RERR
    ARPMECTYPE_RREP_ACK = 4, //!< ARPMECTYPE_RREP_ACK
    ARPMEC_HELLO = 5,     //!< ARPMEC_HELLO
    ARPMEC_JOIN = 6,      //!< ARPMEC_JOIN
    ARPMEC_CH_NOTIFICATION = 7, //!< ARPMEC_CH_NOTIFICATION
    ARPMEC_CLUSTER_LIST = 8,    //!< ARPMEC_CLUSTER_LIST
    ARPMEC_DATA = 9,            //!< ARPMEC_DATA
    ARPMEC_ABDICATE = 10        //!< ARPMEC_ABDICATE
};

/**
 * \ingroup arpmec
 * \brief ARPMEC types
 */
class TypeHeader : public Header
{
  public:
    /**
     * constructor
     * \param t the ARPMEC RREQ type
     */
    TypeHeader(MessageType t = ARPMECTYPE_RREQ);

    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();
    TypeId GetInstanceTypeId() const override;
    uint32_t GetSerializedSize() const override;
    void Serialize(Buffer::Iterator start) const override;
    uint32_t Deserialize(Buffer::Iterator start) override;
    void Print(std::ostream& os) const override;

    /**
     * \returns the type
     */
    MessageType Get() const
    {
        return m_type;
    }

    /**
     * Check that type if valid
     * \returns true if the type is valid
     */
    bool IsValid() const
    {
        return m_valid;
    }

    /**
     * \brief Comparison operator
     * \param o header to compare
     * \return true if the headers are equal
     */
    bool operator==(const TypeHeader& o) const;

  private:
    MessageType m_type; ///< type of the message
    bool m_valid;       ///< Indicates if the message is valid
};

/**
 * \brief Stream output operator
 * \param os output stream
 * \param h the TypeHeader
 * \return updated stream
 */
std::ostream& operator<<(std::ostream& os, const TypeHeader& h);

/**
* \ingroup arpmec
* \brief   Route Request (RREQ) Message Format
  \verbatim
  0                   1                   2                   3
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |     Type      |J|R|G|D|U|   Reserved          |   Hop Count   |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |                            RREQ ID                            |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |                    Destination IP Address                     |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |                  Destination Sequence Number                  |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |                    Originator IP Address                      |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |                  Originator Sequence Number                   |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  \endverbatim
*/
class RreqHeader : public Header
{
  public:
    /**
     * constructor
     *
     * \param flags the message flags (0)
     * \param reserved the reserved bits (0)
     * \param hopCount the hop count
     * \param requestID the request ID
     * \param dst the destination IP address
     * \param dstSeqNo the destination sequence number
     * \param origin the origin IP address
     * \param originSeqNo the origin sequence number
     */
    RreqHeader(uint8_t flags = 0,
               uint8_t reserved = 0,
               uint8_t hopCount = 0,
               uint32_t requestID = 0,
               Ipv4Address dst = Ipv4Address(),
               uint32_t dstSeqNo = 0,
               Ipv4Address origin = Ipv4Address(),
               uint32_t originSeqNo = 0);

    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();
    TypeId GetInstanceTypeId() const override;
    uint32_t GetSerializedSize() const override;
    void Serialize(Buffer::Iterator start) const override;
    uint32_t Deserialize(Buffer::Iterator start) override;
    void Print(std::ostream& os) const override;

    // Fields
    /**
     * \brief Set the hop count
     * \param count the hop count
     */
    void SetHopCount(uint8_t count)
    {
        m_hopCount = count;
    }

    /**
     * \brief Get the hop count
     * \return the hop count
     */
    uint8_t GetHopCount() const
    {
        return m_hopCount;
    }

    /**
     * \brief Set the request ID
     * \param id the request ID
     */
    void SetId(uint32_t id)
    {
        m_requestID = id;
    }

    /**
     * \brief Get the request ID
     * \return the request ID
     */
    uint32_t GetId() const
    {
        return m_requestID;
    }

    /**
     * \brief Set the destination address
     * \param a the destination address
     */
    void SetDst(Ipv4Address a)
    {
        m_dst = a;
    }

    /**
     * \brief Get the destination address
     * \return the destination address
     */
    Ipv4Address GetDst() const
    {
        return m_dst;
    }

    /**
     * \brief Set the destination sequence number
     * \param s the destination sequence number
     */
    void SetDstSeqno(uint32_t s)
    {
        m_dstSeqNo = s;
    }

    /**
     * \brief Get the destination sequence number
     * \return the destination sequence number
     */
    uint32_t GetDstSeqno() const
    {
        return m_dstSeqNo;
    }

    /**
     * \brief Set the origin address
     * \param a the origin address
     */
    void SetOrigin(Ipv4Address a)
    {
        m_origin = a;
    }

    /**
     * \brief Get the origin address
     * \return the origin address
     */
    Ipv4Address GetOrigin() const
    {
        return m_origin;
    }

    /**
     * \brief Set the origin sequence number
     * \param s the origin sequence number
     */
    void SetOriginSeqno(uint32_t s)
    {
        m_originSeqNo = s;
    }

    /**
     * \brief Get the origin sequence number
     * \return the origin sequence number
     */
    uint32_t GetOriginSeqno() const
    {
        return m_originSeqNo;
    }

    // Flags
    /**
     * \brief Set the gratuitous RREP flag
     * \param f the gratuitous RREP flag
     */
    void SetGratuitousRrep(bool f);
    /**
     * \brief Get the gratuitous RREP flag
     * \return the gratuitous RREP flag
     */
    bool GetGratuitousRrep() const;
    /**
     * \brief Set the Destination only flag
     * \param f the Destination only flag
     */
    void SetDestinationOnly(bool f);
    /**
     * \brief Get the Destination only flag
     * \return the Destination only flag
     */
    bool GetDestinationOnly() const;
    /**
     * \brief Set the unknown sequence number flag
     * \param f the unknown sequence number flag
     */
    void SetUnknownSeqno(bool f);
    /**
     * \brief Get the unknown sequence number flag
     * \return the unknown sequence number flag
     */
    bool GetUnknownSeqno() const;

    /**
     * \brief Comparison operator
     * \param o RREQ header to compare
     * \return true if the RREQ headers are equal
     */
    bool operator==(const RreqHeader& o) const;

  private:
    uint8_t m_flags;        ///< |J|R|G|D|U| bit flags, see RFC
    uint8_t m_reserved;     ///< Not used (must be 0)
    uint8_t m_hopCount;     ///< Hop Count
    uint32_t m_requestID;   ///< RREQ ID
    Ipv4Address m_dst;      ///< Destination IP Address
    uint32_t m_dstSeqNo;    ///< Destination Sequence Number
    Ipv4Address m_origin;   ///< Originator IP Address
    uint32_t m_originSeqNo; ///< Source Sequence Number
};

/**
 * \brief Stream output operator
 * \param os output stream
 * \return updated stream
 */
std::ostream& operator<<(std::ostream& os, const RreqHeader&);

/**
* \ingroup arpmec
* \brief Route Reply (RREP) Message Format
  \verbatim
  0                   1                   2                   3
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |     Type      |R|A|    Reserved     |Prefix Sz|   Hop Count   |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |                     Destination IP address                    |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |                  Destination Sequence Number                  |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |                    Originator IP address                      |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |                           Lifetime                            |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  \endverbatim
*/
class RrepHeader : public Header
{
  public:
    /**
     * constructor
     *
     * \param prefixSize the prefix size (0)
     * \param hopCount the hop count (0)
     * \param dst the destination IP address
     * \param dstSeqNo the destination sequence number
     * \param origin the origin IP address
     * \param lifetime the lifetime
     */
    RrepHeader(uint8_t prefixSize = 0,
               uint8_t hopCount = 0,
               Ipv4Address dst = Ipv4Address(),
               uint32_t dstSeqNo = 0,
               Ipv4Address origin = Ipv4Address(),
               Time lifetime = MilliSeconds(0));
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();
    TypeId GetInstanceTypeId() const override;
    uint32_t GetSerializedSize() const override;
    void Serialize(Buffer::Iterator start) const override;
    uint32_t Deserialize(Buffer::Iterator start) override;
    void Print(std::ostream& os) const override;

    // Fields
    /**
     * \brief Set the hop count
     * \param count the hop count
     */
    void SetHopCount(uint8_t count)
    {
        m_hopCount = count;
    }

    /**
     * \brief Get the hop count
     * \return the hop count
     */
    uint8_t GetHopCount() const
    {
        return m_hopCount;
    }

    /**
     * \brief Set the destination address
     * \param a the destination address
     */
    void SetDst(Ipv4Address a)
    {
        m_dst = a;
    }

    /**
     * \brief Get the destination address
     * \return the destination address
     */
    Ipv4Address GetDst() const
    {
        return m_dst;
    }

    /**
     * \brief Set the destination sequence number
     * \param s the destination sequence number
     */
    void SetDstSeqno(uint32_t s)
    {
        m_dstSeqNo = s;
    }

    /**
     * \brief Get the destination sequence number
     * \return the destination sequence number
     */
    uint32_t GetDstSeqno() const
    {
        return m_dstSeqNo;
    }

    /**
     * \brief Set the origin address
     * \param a the origin address
     */
    void SetOrigin(Ipv4Address a)
    {
        m_origin = a;
    }

    /**
     * \brief Get the origin address
     * \return the origin address
     */
    Ipv4Address GetOrigin() const
    {
        return m_origin;
    }

    /**
     * \brief Set the lifetime
     * \param t the lifetime
     */
    void SetLifeTime(Time t);
    /**
     * \brief Get the lifetime
     * \return the lifetime
     */
    Time GetLifeTime() const;

    // Flags
    /**
     * \brief Set the ack required flag
     * \param f the ack required flag
     */
    void SetAckRequired(bool f);
    /**
     * \brief get the ack required flag
     * \return the ack required flag
     */
    bool GetAckRequired() const;
    /**
     * \brief Set the prefix size
     * \param sz the prefix size
     */
    void SetPrefixSize(uint8_t sz);
    /**
     * \brief Set the prefix size
     * \return the prefix size
     */
    uint8_t GetPrefixSize() const;

    /**
     * Configure RREP to be a Hello message
     *
     * \param src the source IP address
     * \param srcSeqNo the source sequence number
     * \param lifetime the lifetime of the message
     */
    void SetHello(Ipv4Address src, uint32_t srcSeqNo, Time lifetime);

    /**
     * \brief Comparison operator
     * \param o RREP header to compare
     * \return true if the RREP headers are equal
     */
    bool operator==(const RrepHeader& o) const;

  private:
    uint8_t m_flags;      ///< A - acknowledgment required flag
    uint8_t m_prefixSize; ///< Prefix Size
    uint8_t m_hopCount;   ///< Hop Count
    Ipv4Address m_dst;    ///< Destination IP Address
    uint32_t m_dstSeqNo;  ///< Destination Sequence Number
    Ipv4Address m_origin; ///< Source IP Address
    uint32_t m_lifeTime;  ///< Lifetime (in milliseconds)
};

/**
 * \brief Stream output operator
 * \param os output stream
 * \return updated stream
 */
std::ostream& operator<<(std::ostream& os, const RrepHeader&);

/**
* \ingroup arpmec
* \brief Route Reply Acknowledgment (RREP-ACK) Message Format
  \verbatim
  0                   1
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |     Type      |   Reserved    |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  \endverbatim
*/
class RrepAckHeader : public Header
{
  public:
    /// constructor
    RrepAckHeader();

    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();
    TypeId GetInstanceTypeId() const override;
    uint32_t GetSerializedSize() const override;
    void Serialize(Buffer::Iterator start) const override;
    uint32_t Deserialize(Buffer::Iterator start) override;
    void Print(std::ostream& os) const override;

    /**
     * \brief Comparison operator
     * \param o RREP header to compare
     * \return true if the RREQ headers are equal
     */
    bool operator==(const RrepAckHeader& o) const;

  private:
    uint8_t m_reserved; ///< Not used (must be 0)
};

/**
 * \brief Stream output operator
 * \param os output stream
 * \return updated stream
 */
std::ostream& operator<<(std::ostream& os, const RrepAckHeader&);

/**
* \ingroup arpmec
* \brief Route Error (RERR) Message Format
  \verbatim
  0                   1                   2                   3
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |     Type      |N|          Reserved           |   DestCount   |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |            Unreachable Destination IP Address (1)             |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |         Unreachable Destination Sequence Number (1)           |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-|
  |  Additional Unreachable Destination IP Addresses (if needed)  |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |Additional Unreachable Destination Sequence Numbers (if needed)|
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  \endverbatim
*/
class RerrHeader : public Header
{
  public:
    /// constructor
    RerrHeader();

    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();
    TypeId GetInstanceTypeId() const override;
    uint32_t GetSerializedSize() const override;
    void Serialize(Buffer::Iterator i) const override;
    uint32_t Deserialize(Buffer::Iterator start) override;
    void Print(std::ostream& os) const override;

    // No delete flag
    /**
     * \brief Set the no delete flag
     * \param f the no delete flag
     */
    void SetNoDelete(bool f);
    /**
     * \brief Get the no delete flag
     * \return the no delete flag
     */
    bool GetNoDelete() const;

    /**
     * \brief Add unreachable node address and its sequence number in RERR header
     * \param dst unreachable IPv4 address
     * \param seqNo unreachable sequence number
     * \return false if we already added maximum possible number of unreachable destinations
     */
    bool AddUnDestination(Ipv4Address dst, uint32_t seqNo);
    /**
     * \brief Delete pair (address + sequence number) from REER header, if the number of unreachable
     * destinations > 0
     * \param un unreachable pair (address + sequence number)
     * \return true on success
     */
    bool RemoveUnDestination(std::pair<Ipv4Address, uint32_t>& un);
    /// Clear header
    void Clear();

    /**
     * \returns number of unreachable destinations in RERR message
     */
    uint8_t GetDestCount() const
    {
        return (uint8_t)m_unreachableDstSeqNo.size();
    }

    /**
     * \brief Comparison operator
     * \param o RERR header to compare
     * \return true if the RERR headers are equal
     */
    bool operator==(const RerrHeader& o) const;

  private:
    uint8_t m_flag;     ///< No delete flag
    uint8_t m_reserved; ///< Not used (must be 0)

    /// List of Unreachable destination: IP addresses and sequence numbers
    std::map<Ipv4Address, uint32_t> m_unreachableDstSeqNo;
};

/**
 * \brief Stream output operator
 * \param os output stream
 * \return updated stream
 */
std::ostream& operator<<(std::ostream& os, const RerrHeader&);

// Dans aodv-packet.h
/**
 * \ingroup myaodv
 * \brief ARPMEC Hello Header
 *
 * This header is used to send Hello messages in the ARPMEC protocol.
 * It contains the node ID and channel ID.
 */
class ArpmecHelloHeader : public Header
{
  public:
    /**
     * \brief Constructor
     * \param nodeId the node ID (0)
     * \param channelId the channel ID (0)
     */
    ArpmecHelloHeader();

    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    TypeId GetInstanceTypeId() const override;
    /**
     * \brief Get the serialized size of the header
     * \return the serialized size in bytes
     */
    uint32_t GetSerializedSize() const override;
    /**
     * \brief Serialize the header into a buffer
     * \param start the buffer iterator to start serialization
     */
    void Serialize(Buffer::Iterator start) const override;
    /**
     * \brief Deserialize the header from a buffer
     * \param start the buffer iterator to start deserialization
     * \return the number of bytes deserialized
     */
    uint32_t Deserialize(Buffer::Iterator start) override;
    /**
     * \brief Print the header contents
     * \param os the output stream
     */
    void Print(std::ostream& os) const override;

    // Champs
    /**
     * \brief Set the node ID
     * \param id the node ID
     */
    void SetNodeId(uint32_t id) 
    { 
      m_nodeId = id; 
    }
    /**
     * \brief Get the node ID
     * \return the node ID
     */
    uint32_t GetNodeId() const 
    { 
      return m_nodeId; 
    }
    /**
     * \brief Set the channel ID
     * \param cid the channel ID
     */
    void SetChannelId(uint8_t cid) 
    { 
      m_channelId = cid; 
    }
    /**
     * \brief Get the channel ID
     * \return the channel ID
     */
    uint8_t GetChannelId() const 
    { 
      return m_channelId; 
    }
    /**
     * \brief Comparison operator
     * \param o the other ArpmecHelloHeader to compare
     * \return true if the headers are equal
     */

//--------------------------------------------------------
//Step 4
//--------------------------------------------------------
    void SetSequenceNumber(uint32_t seq) { m_sequenceNumber = seq; }
    uint32_t GetSequenceNumber() const { return m_sequenceNumber; }

    bool operator==(const ArpmecHelloHeader& o) const;




  private:
    /**
     * \brief Stream output operator
     * \param os output stream
     * \param h the ArpmecHelloHeader
     * \return updated stream
     */
    uint32_t m_nodeId;    // ID du nœud
    uint8_t m_channelId;  // ID du canal
    uint32_t m_sequenceNumber; // Added for PDR  step 4

};
// Other header classes (ArpmecJoinHeader, etc.) remain unchanged
// ... (include definitions for JOIN, CH_NOTIFICATION, CLUSTER_LIST, DATA, ABDICATE as previously provided)


// Dans aodv-packet.h
/**
 * \ingroup myaodv
 * \brief ARPMEC Join Header
 *
 * This header is used to send Join messages in the ARPMEC protocol.
 * It contains the node ID and channel ID.
 */
class ArpmecJoinHeader : public Header
{
  public:
    /**
     * \brief Constructor
     * \param nodeId the node ID (0)
     * \param channelId the channel ID (0)
     */
    ArpmecJoinHeader();

    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    TypeId GetInstanceTypeId() const override;
    /**
     * \brief Get the serialized size of the header
     * \return the serialized size in bytes
     */
    uint32_t GetSerializedSize() const override;
    /**
     * \brief Serialize the header into a buffer
     * \param start the buffer iterator to start serialization
     */
    void Serialize(Buffer::Iterator start) const override;
    /**
     * \brief Deserialize the header from a buffer
     * \param start the buffer iterator to start deserialization
     * \return the number of bytes deserialized
     */
    uint32_t Deserialize(Buffer::Iterator start) override;
    /**
     * \brief Print the header contents
     * \param os the output stream
     */
    void Print(std::ostream& os) const override;

    // Champs
    /**
     * \brief Set the node ID
     * \param id the node ID
     */
    void SetNodeId(uint32_t id) { m_nodeId = id; }
    /**
     * \brief Get the node ID
     * \return the node ID
     */
    uint32_t GetNodeId() const { return m_nodeId; }
    /**
     * \brief Set the channel ID
     * \param cid the channel ID
     */
    void SetChId(uint32_t cid) { m_chId = cid; }
    /**
     * \brief Get the channel ID
     * \return the channel ID
     */
    uint32_t GetChId() const { return m_chId; }
    /**
     * \brief Comparison operator
     * \param o the other ArpmecJoinHeader to compare
     * \return true if the headers are equal
     */

    bool operator==(const ArpmecJoinHeader& o) const;

  private:
    /**
     * \brief Stream output operator
     * \param os output stream
     * \param h the ArpmecJoinHeader
     * \return updated stream
     */
    uint32_t m_nodeId;    // ID du nœud
    uint32_t m_chId;      // ID du CH
};

// Dans aodv-packet.h
/**
 * \ingroup myaodv
 * \brief ARPMEC Channel Notification Header
 *
 * This header is used to send channel notification messages in the ARPMEC protocol.
 * It contains the channel ID and a list of cluster members.
 */
class ArpmecChNotificationHeader : public Header
{
  public:
    /**
     * \brief Constructor
     * Initializes the channel ID and cluster members.
     */
    ArpmecChNotificationHeader();
    /**
     * \brief Constructor with parameters
     * \param cid the channel ID
     * \param members the list of cluster members
     */
    static TypeId GetTypeId();
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    TypeId GetInstanceTypeId() const override;
    /**
     * \brief Get the serialized size of the header
     * \return the serialized size in bytes
     */
    uint32_t GetSerializedSize() const override;
    /**
     * \brief Serialize the header into a buffer
     * \param start the buffer iterator to start serialization
     */
    void Serialize(Buffer::Iterator start) const override;
    /**
     * \brief Deserialize the header from a buffer
     * \param start the buffer iterator to start deserialization
     * \return the number of bytes deserialized
     */
    uint32_t Deserialize(Buffer::Iterator start) override;
    /**
     * \brief Print the header contents
     * \param os the output stream
     */
    void Print(std::ostream& os) const override;

    // Champs
    /**
     * \brief Set the channel ID
     * \param cid the channel ID
     */
    void SetChId(uint32_t cid) 
    { 
      m_chId = cid; 
    }
    /**
     * \brief Get the channel ID
     * \return the channel ID
     */
    uint32_t GetChId() const 
    { 
      return m_chId; 
    }
    /**
     * \brief Set the cluster members
     * \param members the list of cluster members
     */
    void SetClusterMembers(const std::vector<uint32_t>& members) 
    { 
      m_clusterMembers = members; 
    }
    /**
     * \brief Add a cluster member
     * \param member the cluster member to add
     */
    const std::vector<uint32_t>& GetClusterMembers() const 
    { 
      return m_clusterMembers; 
    }

    /**
     * \brief Add a cluster member
     * \param member the cluster member to add
     */
    bool operator==(const ArpmecChNotificationHeader& o) const;

  private:
    uint32_t m_chId;              // ID du CH
    std::vector<uint32_t> m_clusterMembers; // Liste des membres
};

// Dans aodv-packet.h
/**
 * \ingroup myaodv
 * \brief ARPMEC Cluster List Header
 *
 * This header is used to send cluster list messages in the ARPMEC protocol.
 * It contains a list of clusters, each represented by a pair of cluster ID and a list of member IDs.
 */
class ArpmecClusterListHeader : public Header
{
  public:
    /**
     * \brief Constructor
     * Initializes an empty cluster list.
     */
  public:
    
    ArpmecClusterListHeader();
    /**
     * \brief Constructor with parameters
     * \param clusters the list of clusters, each represented by a pair of cluster ID and a list of member IDs
     */


    static TypeId GetTypeId();
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    TypeId GetInstanceTypeId() const override;
    /**
     * \brief Get the serialized size of the header
     * \return the serialized size in bytes
     */
    uint32_t GetSerializedSize() const override;
    /**
     * \brief Serialize the header into a buffer
     * \param start the buffer iterator to start serialization
     */
    void Serialize(Buffer::Iterator start) const override;
    /**
     * \brief Deserialize the header from a buffer
     * \param start the buffer iterator to start deserialization
     * \return the number of bytes deserialized
     */
    uint32_t Deserialize(Buffer::Iterator start) override;
    /**
     * \brief Print the header contents
     * \param os the output stream
     */
    void Print(std::ostream& os) const override;

    // Champs
    /**
     * \brief Set the clusters
     * \param clusters the list of clusters, each represented by a pair of cluster ID and a list of member IDs
     */
    void SetClusters(const std::vector<std::pair<uint32_t, std::vector<uint32_t>>>& clusters) 
    { 
      m_clusters = clusters; 
    }
    /**
     * \brief Get the clusters
     * \return the list of clusters, each represented by a pair of cluster ID and a list of member IDs
     */
    const std::vector<std::pair<uint32_t, std::vector<uint32_t>>>& GetClusters() const 
    { 
      return m_clusters; 
    }
    /**
     * \brief Add a cluster
     * \param clusterId the cluster ID
     * \param members the list of member IDs in the cluster
     */

    bool operator==(const ArpmecClusterListHeader& o) const;


  private:

    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> m_clusters; // Liste des clusters
};


// Dans aodv-packet.h
/**
 * \ingroup myaodv
 * \brief ARPMEC Data Header
 *
 * This header is used to send data messages in the ARPMEC protocol.
 * It contains source ID, destination ID, and channel ID.
 */
class ArpmecDataHeader : public Header
{
  public:
    /**
     * \brief Constructor
     * Initializes the source ID, destination ID, and channel ID to 0.
     */
    ArpmecDataHeader();
    /**
     * \brief Constructor with parameters
     * \param sourceId the source ID
     * \param destId the destination ID
     * \param chId the channel ID
     */
    static TypeId GetTypeId();
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    TypeId GetInstanceTypeId() const override;
    /**
     * \brief Get the serialized size of the header
     * \return the serialized size in bytes
     */
    uint32_t GetSerializedSize() const override;
    /**
     * \brief Serialize the header into a buffer
     * \param start the buffer iterator to start serialization
     */
    void Serialize(Buffer::Iterator start) const override;
    /**
     * \brief Deserialize the header from a buffer
     * \param start the buffer iterator to start deserialization
     * \return the number of bytes deserialized
     */
    uint32_t Deserialize(Buffer::Iterator start) override;
    /**
     * \brief Print the header contents
     * \param os the output stream
     */
    void Print(std::ostream& os) const override;

    // Champs
    /**
     * \brief Set the source ID
     * \param id the source ID
     */
    void SetSourceId(uint32_t id) 
    { 
      m_sourceId = id; 
    }
    /**
     * \brief Get the source ID
     * \return the source ID
     */
    uint32_t GetSourceId() const 
    { 
      return m_sourceId; 
    }
    /**
     * \brief Set the destination ID
     * \param id the destination ID
     */
    void SetDestId(uint32_t id) 
    { 
      m_destId = id; 
    }
    /**
     * \brief Get the destination ID
     * \return the destination ID
     */
    uint32_t GetDestId() const 
    { 
      return m_destId; 
    }
    /**
     * \brief Set the channel ID
     * \param cid the channel ID
     */
    void SetChId(uint32_t cid) 
    { 
      m_chId = cid; 
    }
    /**
     * \brief Get the channel ID
     * \return the channel ID
     */
    uint32_t GetChId() const 
    { 
      return m_chId; 
    }

    bool operator==(const ArpmecDataHeader& o) const;

  private:
    /**
     * \brief Stream output operator
     * \param os output stream
     * \param h the ArpmecDataHeader
     * \return updated stream
     */
    uint32_t m_sourceId;  // ID de la source
    uint32_t m_destId;    // ID de la destination
    uint32_t m_chId;      // ID du CH
};

// Dans aodv-packet.h
/**
 * \ingroup myaodv
 * \brief ARPMEC Abdicate Header
 *
 * This header is used to send abdication messages in the ARPMEC protocol.
 * It contains the channel ID of the CH that is abdicating.
 */
class ArpmecAbdicateHeader : public Header
{
  public:
    /**
     * \brief Constructor
     * Initializes the channel ID to 0.
     */
  public:

    ArpmecAbdicateHeader();
    /**
     * \brief Constructor with parameters
     * \param cid the channel ID of the CH that is abdicating
     */
    static TypeId GetTypeId();
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    TypeId GetInstanceTypeId() const override;
    /**
     * \brief Get the serialized size of the header
     * \return the serialized size in bytes
     */
    uint32_t GetSerializedSize() const override;
    /**
     * \brief Serialize the header into a buffer
     * \param start the buffer iterator to start serialization
     */
    void Serialize(Buffer::Iterator start) const override;
    /**
     * \brief Deserialize the header from a buffer
     * \param start the buffer iterator to start deserialization
     * \return the number of bytes deserialized
     */
    uint32_t Deserialize(Buffer::Iterator start) override;
    /**
     * \brief Print the header contents
     * \param os the output stream
     */
    void Print(std::ostream& os) const override;

    // Champs
    /**
     * \brief Set the channel ID
     * \param cid the channel ID of the CH that is abdicating
     */
    void SetChId(uint32_t cid) 
    { 
      m_chId = cid; 
    }
    /**
     * \brief Get the channel ID
     * \return the channel ID of the CH that is abdicating
     */
    uint32_t GetChId() const 
    {
       return m_chId; 
      }
    /**
     * \brief Comparison operator
     * \param o the other ArpmecAbdicateHeader to compare
     * \return true if the headers are equal
     */
    bool operator==(const ArpmecAbdicateHeader& o) const;

  private:
    /**
     * \brief Stream output operator
     * \param os output stream
     * \param h the ArpmecAbdicateHeader
     * \return updated stream
     */
    uint32_t m_chId;      // ID du CH
};

// Opérateurs de flux pour l’affichage
std::ostream& operator<<(std::ostream& os, const ArpmecHelloHeader& h);
std::ostream& operator<<(std::ostream& os, const ArpmecJoinHeader& h);
std::ostream& operator<<(std::ostream& os, const ArpmecChNotificationHeader& h);
std::ostream& operator<<(std::ostream& os, const ArpmecClusterListHeader& h);
std::ostream& operator<<(std::ostream& os, const ArpmecDataHeader& h);
std::ostream& operator<<(std::ostream& os, const ArpmecAbdicateHeader& h);


} // namespace arpmec
} // namespace ns3

#endif /* ARPMECPACKET_H */
