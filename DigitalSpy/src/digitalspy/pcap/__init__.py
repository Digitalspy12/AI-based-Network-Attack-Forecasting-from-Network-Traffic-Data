"""DigitalSpy PCAP ingestion module.

Provides streaming packet-level feature extraction for Phase 8A.
Never loads a full PCAP into memory (no rdpcap / scapy.rdpcap usage).
"""
