"""Phase 8A — Streaming PCAP reader.

Streams packets one at a time from a .pcap file.  The caller is responsible
for accumulating packets into time-windows and flushing.

Design rules:
  - NEVER use scapy.rdpcap() or dpkt.pcap.Reader() with a list comprehension
    that materialises all packets.  Both the 11 GB Tuesday capture and any
    other large file must be read packet-by-packet through a context manager.
  - Primary implementation: scapy.utils.PcapReader (context manager).
  - Fallback implementation: dpkt.pcap / dpkt.pcapng file-handle streaming.
  - The generator yields one PacketRecord namedtuple per packet; memory is
    bounded by the single-packet parse overhead.

PacketRecord fields
-------------------
  timestamp   : float  — Unix epoch seconds (fractional)
  src_ip      : str
  dst_ip      : str
  src_port    : int    — 0 if not TCP/UDP
  dst_port    : int    — 0 if not TCP/UDP
  protocol    : int    — IP protocol number (6=TCP, 17=UDP, etc.)
  ttl         : int
  tcp_window  : int    — 0 if not TCP
  tcp_flags   : int    — raw flags byte; 0 if not TCP
  ip_frag     : bool   — True when MF flag set or fragment offset > 0
  payload_len : int    — L4 payload length in bytes
  pkt_len     : int    — total packet length on wire
"""
from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Generator, NamedTuple, Optional

logger = logging.getLogger(__name__)


class PacketRecord(NamedTuple):
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    ttl: int
    tcp_window: int
    tcp_flags: int
    ip_frag: bool
    payload_len: int
    pkt_len: int


def stream_pcap(
    pcap_path: str | Path,
    max_packets: Optional[int] = None,
    filter_ip: Optional[str] = None,
) -> Generator[PacketRecord, None, None]:
    """Stream packets from a .pcap file one at a time.

    Uses scapy PcapReader (context manager) as the primary approach.
    Falls back to dpkt streaming on import error.

    Args:
        pcap_path: Absolute path to the .pcap file.
        max_packets: Stop after this many packets (None = read all).
        filter_ip: If given, yield only packets where src or dst matches.

    Yields:
        PacketRecord for each valid IP packet found.
    """
    pcap_path = Path(pcap_path)
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP not found: {pcap_path}")

    try:
        yield from _stream_scapy(pcap_path, max_packets, filter_ip)
    except ImportError:
        logger.warning("scapy not available; falling back to dpkt streaming reader.")
        yield from _stream_dpkt(pcap_path, max_packets, filter_ip)


# ── Scapy streaming implementation ───────────────────────────────────────────

def _stream_scapy(
    pcap_path: Path,
    max_packets: Optional[int],
    filter_ip: Optional[str],
) -> Generator[PacketRecord, None, None]:
    from scapy.utils import PcapReader  # noqa: PLC0415
    from scapy.layers.inet import IP, TCP, UDP  # noqa: PLC0415

    count = 0
    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            if not pkt.haslayer(IP):
                continue

            ip = pkt[IP]
            src_ip = ip.src
            dst_ip = ip.dst

            if filter_ip and filter_ip not in (src_ip, dst_ip):
                continue

            proto = ip.proto
            ttl = int(ip.ttl)
            frag_flags = int(ip.flags)
            frag_offset = int(ip.frag)
            ip_frag = bool((frag_flags & 0x1) or frag_offset > 0)  # MF bit or offset
            pkt_len = len(pkt)

            src_port = dst_port = 0
            tcp_window = tcp_flags = 0
            payload_len = 0

            if pkt.haslayer(TCP):
                t = pkt[TCP]
                src_port = int(t.sport)
                dst_port = int(t.dport)
                tcp_window = int(t.window)
                tcp_flags = int(t.flags)
                payload_len = len(bytes(t.payload))
            elif pkt.haslayer(UDP):
                u = pkt[UDP]
                src_port = int(u.sport)
                dst_port = int(u.dport)
                payload_len = len(bytes(u.payload))
            else:
                # Other IP protocol
                payload_len = max(0, pkt_len - (ip.ihl * 4) - 20)

            ts = float(pkt.time)

            yield PacketRecord(
                timestamp=ts,
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=proto,
                ttl=ttl,
                tcp_window=tcp_window,
                tcp_flags=tcp_flags,
                ip_frag=ip_frag,
                payload_len=payload_len,
                pkt_len=pkt_len,
            )

            count += 1
            if max_packets is not None and count >= max_packets:
                break


# ── dpkt fallback streaming implementation ───────────────────────────────────

def _stream_dpkt(
    pcap_path: Path,
    max_packets: Optional[int],
    filter_ip: Optional[str],
) -> Generator[PacketRecord, None, None]:
    import dpkt  # noqa: PLC0415
    import socket  # noqa: PLC0415

    count = 0
    with open(pcap_path, "rb") as fh:
        try:
            cap = dpkt.pcap.Reader(fh)
        except Exception:
            fh.seek(0)
            cap = dpkt.pcapng.Reader(fh)

        for ts, raw in cap:
            try:
                eth = dpkt.ethernet.Ethernet(raw)
                if not isinstance(eth.data, dpkt.ip.IP):
                    continue
                ip = eth.data
            except Exception:
                continue

            src_ip = socket.inet_ntoa(ip.src)
            dst_ip = socket.inet_ntoa(ip.dst)

            if filter_ip and filter_ip not in (src_ip, dst_ip):
                continue

            proto = ip.p
            ttl = ip.ttl
            # IP flags: bit 2 = MF; offset in ip.off bits 12:0
            ip_frag = bool((ip.off & dpkt.ip.IP_MF) or (ip.off & dpkt.ip.IP_OFFMASK))
            pkt_len = len(raw)

            src_port = dst_port = 0
            tcp_window = tcp_flags = 0
            payload_len = 0

            if isinstance(ip.data, dpkt.tcp.TCP):
                t = ip.data
                src_port = t.sport
                dst_port = t.dport
                tcp_window = t.win
                tcp_flags = t.flags
                payload_len = len(t.data)
            elif isinstance(ip.data, dpkt.udp.UDP):
                u = ip.data
                src_port = u.sport
                dst_port = u.dport
                payload_len = len(u.data)
            else:
                payload_len = len(ip.data)

            yield PacketRecord(
                timestamp=float(ts),
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=proto,
                ttl=ttl,
                tcp_window=tcp_window,
                tcp_flags=tcp_flags,
                ip_frag=ip_frag,
                payload_len=payload_len,
                pkt_len=pkt_len,
            )

            count += 1
            if max_packets is not None and count >= max_packets:
                break


# ── Utility: window-aligned batch accumulator ─────────────────────────────────

def stream_windows(
    pcap_path: str | Path,
    window_seconds: float = 10.0,
    max_packets: Optional[int] = None,
    filter_ip: Optional[str] = None,
) -> Generator[tuple[float, list[PacketRecord]], None, None]:
    """Yield (window_start_ts, [PacketRecord, ...]) buckets aligned to wall-clock windows.

    Window boundaries are floor-aligned to window_seconds:
        window_start = floor(timestamp / window_seconds) * window_seconds

    The bucket is yielded when the first packet of the *next* window arrives,
    so all packets within a window are complete before being emitted.

    Args:
        pcap_path: Path to .pcap file.
        window_seconds: Window size in seconds (default 10, matching flow pipeline).
        max_packets: Cap on total packets read.
        filter_ip: If given, only packets from/to this IP.

    Yields:
        (window_start_ts, bucket) where bucket is a non-empty list of PacketRecords.
    """
    import math

    current_bucket: list[PacketRecord] = []
    current_window_start: Optional[float] = None

    for pkt in stream_pcap(pcap_path, max_packets=max_packets, filter_ip=filter_ip):
        w_start = math.floor(pkt.timestamp / window_seconds) * window_seconds

        if current_window_start is None:
            current_window_start = w_start

        if w_start != current_window_start:
            if current_bucket:
                yield current_window_start, current_bucket
            current_bucket = []
            current_window_start = w_start

        current_bucket.append(pkt)

    # Flush the final bucket
    if current_bucket and current_window_start is not None:
        yield current_window_start, current_bucket
