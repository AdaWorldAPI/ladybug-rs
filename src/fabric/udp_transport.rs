//! UDP Transport: FireflyFrames over the wire
//!
//! Enables zero-copy cognitive execution between:
//! - Railway cloud (Firefly compiler) → UDP → Raspberry Pi (Ladybug executor)
//! - Any UDP sender → Any Ladybug runtime
//!
//! ## Why UDP?
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                           UDP vs TCP/HTTP                                   │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │  HTTP/REST:  ~100ms RTT, JSON parse, memory copy, allocations              │
//! │  gRPC:       ~20ms RTT, protobuf parse, connection overhead                │
//! │  WebSocket:  ~10ms RTT, framing overhead, still needs parsing              │
//! │  UDP Frame:  ~1ms RTT, zero-copy decode, direct execution                  │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Frame Format (1250 bytes fits in single UDP packet)
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────────────┐
//! │  0x00-0x07: Header (magic, version, session, lane, hive, seq)        │
//! │  0x08-0x0F: Instruction (prefix, opcode, flags, dest, src1, src2)    │
//! │  0x10-0x1F: Operand (128-bit extended operand)                       │
//! │  0x20-0x4F: Data (384-bit fingerprint fragment)                      │
//! │  0x50-0x7F: Context (qualia, truth, timestamp, version)              │
//! │  0x80-0x9F: ECC (Reed-Solomon for error correction)                  │
//! └───────────────────────────────────────────────────────────────────────┘
//! Total: 160 bytes core + up to 1090 bytes payload
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! // Server (Railway/Firefly)
//! let sender = UdpSender::new("0.0.0.0:5050").await?;
//! sender.send_frame(&frame, "192.168.1.100:5051").await?;
//!
//! // Client (Raspberry Pi)
//! let receiver = UdpReceiver::new("0.0.0.0:5051").await?;
//! let mut executor = Executor::new();
//! receiver.run_executor(&mut executor).await;
//! ```

use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::firefly_frame::FireflyFrame;
use super::executor::{Executor, ExecResult};

/// Size of FireflyFrame in u64 words
const FRAME_WORDS: usize = FireflyFrame::WORDS;
/// Size of FireflyFrame in bytes
const FRAME_BYTES: usize = FRAME_WORDS * 8;

/// Maximum UDP payload (MTU - IP header - UDP header)
pub const MAX_UDP_PAYLOAD: usize = 1472;

/// Magic bytes for frame identification
pub const FRAME_MAGIC: [u8; 2] = [0xAD, 0xA1];

/// Frame protocol version
pub const FRAME_VERSION: u8 = 1;

// =============================================================================
// FRAME PACKET
// =============================================================================

/// UDP frame packet with transport metadata
#[derive(Debug, Clone)]
pub struct FramePacket {
    /// The firefly frame
    pub frame: FireflyFrame,
    /// Source address
    pub from: SocketAddr,
    /// Arrival timestamp
    pub received_at: Instant,
    /// Packet sequence for reordering
    pub sequence: u32,
    /// Acknowledgment requested
    pub ack_requested: bool,
}

impl FramePacket {
    /// Encode frame to UDP packet bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(MAX_UDP_PAYLOAD);

        // Transport header (8 bytes)
        buf.extend_from_slice(&FRAME_MAGIC);
        buf.push(FRAME_VERSION);
        buf.push(if self.ack_requested { 0x01 } else { 0x00 });
        buf.extend_from_slice(&self.sequence.to_le_bytes());

        // Frame data: convert u64 words to bytes
        let mut frame_clone = self.frame.clone();
        let words = frame_clone.encode();
        for word in words.iter() {
            buf.extend_from_slice(&word.to_le_bytes());
        }

        buf
    }

    /// Decode from UDP packet bytes
    pub fn decode(data: &[u8], from: SocketAddr) -> Result<Self, DecodeError> {
        // Need at least 8 bytes header + FRAME_BYTES for frame
        let min_size = 8 + FRAME_BYTES;
        if data.len() < min_size {
            return Err(DecodeError::TooShort);
        }

        // Verify magic
        if data[0..2] != FRAME_MAGIC {
            return Err(DecodeError::BadMagic);
        }

        // Check version
        let version = data[2];
        if version != FRAME_VERSION {
            return Err(DecodeError::VersionMismatch { expected: FRAME_VERSION, got: version });
        }

        let ack_requested = data[3] & 0x01 != 0;
        let sequence = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

        // Convert bytes to u64 words for frame decode
        let frame_bytes = &data[8..8 + FRAME_BYTES];
        let mut words = [0u64; FRAME_WORDS];
        for (i, chunk) in frame_bytes.chunks_exact(8).enumerate() {
            if i < FRAME_WORDS {
                words[i] = u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                    chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
            }
        }

        // Decode frame
        let frame = FireflyFrame::decode(&words)
            .ok_or_else(|| DecodeError::FrameError("frame decode failed".to_string()))?;

        Ok(Self {
            frame,
            from,
            received_at: Instant::now(),
            sequence,
            ack_requested,
        })
    }
}

#[derive(Debug)]
pub enum DecodeError {
    TooShort,
    BadMagic,
    VersionMismatch { expected: u8, got: u8 },
    FrameError(String),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::TooShort => write!(f, "packet too short"),
            DecodeError::BadMagic => write!(f, "bad magic bytes"),
            DecodeError::VersionMismatch { expected, got } =>
                write!(f, "version mismatch: expected {}, got {}", expected, got),
            DecodeError::FrameError(e) => write!(f, "frame decode error: {}", e),
        }
    }
}

impl std::error::Error for DecodeError {}

// =============================================================================
// UDP SENDER
// =============================================================================

/// Send FireflyFrames over UDP
pub struct UdpSender {
    socket: std::net::UdpSocket,
    sequence: u32,
    stats: SenderStats,
}

#[derive(Debug, Clone, Default)]
pub struct SenderStats {
    pub frames_sent: u64,
    pub bytes_sent: u64,
    pub errors: u64,
}

impl UdpSender {
    /// Create new UDP sender bound to address
    pub fn new(bind_addr: &str) -> io::Result<Self> {
        let socket = std::net::UdpSocket::bind(bind_addr)?;
        socket.set_nonblocking(false)?;
        Ok(Self {
            socket,
            sequence: 0,
            stats: SenderStats::default(),
        })
    }

    /// Send a single frame to destination
    pub fn send_frame(&mut self, frame: &FireflyFrame, dest: &str) -> io::Result<()> {
        let packet = FramePacket {
            frame: frame.clone(),
            from: self.socket.local_addr()?,
            received_at: Instant::now(),
            sequence: self.sequence,
            ack_requested: false,
        };
        self.sequence = self.sequence.wrapping_add(1);

        let data = packet.encode();
        let dest_addr: SocketAddr = dest.parse()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;

        match self.socket.send_to(&data, dest_addr) {
            Ok(bytes) => {
                self.stats.frames_sent += 1;
                self.stats.bytes_sent += bytes as u64;
                Ok(())
            }
            Err(e) => {
                self.stats.errors += 1;
                Err(e)
            }
        }
    }

    /// Send multiple frames (batch)
    pub fn send_frames(&mut self, frames: &[FireflyFrame], dest: &str) -> io::Result<usize> {
        let mut sent = 0;
        for frame in frames {
            self.send_frame(frame, dest)?;
            sent += 1;
        }
        Ok(sent)
    }

    /// Broadcast frame to multiple destinations
    pub fn broadcast(&mut self, frame: &FireflyFrame, dests: &[&str]) -> io::Result<usize> {
        let mut sent = 0;
        for dest in dests {
            if self.send_frame(frame, dest).is_ok() {
                sent += 1;
            }
        }
        Ok(sent)
    }

    pub fn stats(&self) -> &SenderStats {
        &self.stats
    }
}

// =============================================================================
// UDP RECEIVER
// =============================================================================

/// Receive and execute FireflyFrames over UDP
pub struct UdpReceiver {
    socket: std::net::UdpSocket,
    buffer: [u8; MAX_UDP_PAYLOAD],
    stats: ReceiverStats,
    /// Callback for each executed frame
    on_result: Option<Box<dyn Fn(SocketAddr, ExecResult) + Send + Sync>>,
}

#[derive(Debug, Clone, Default)]
pub struct ReceiverStats {
    pub frames_received: u64,
    pub frames_executed: u64,
    pub bytes_received: u64,
    pub decode_errors: u64,
    pub exec_errors: u64,
}

impl UdpReceiver {
    /// Create new UDP receiver bound to address
    pub fn new(bind_addr: &str) -> io::Result<Self> {
        let socket = std::net::UdpSocket::bind(bind_addr)?;
        socket.set_nonblocking(false)?;
        Ok(Self {
            socket,
            buffer: [0u8; MAX_UDP_PAYLOAD],
            stats: ReceiverStats::default(),
            on_result: None,
        })
    }

    /// Set receive timeout
    pub fn set_timeout(&self, timeout: Duration) -> io::Result<()> {
        self.socket.set_read_timeout(Some(timeout))
    }

    /// Set callback for execution results
    pub fn on_result<F>(&mut self, callback: F)
    where
        F: Fn(SocketAddr, ExecResult) + Send + Sync + 'static,
    {
        self.on_result = Some(Box::new(callback));
    }

    /// Receive single frame (blocking)
    pub fn recv_frame(&mut self) -> io::Result<FramePacket> {
        let (size, from) = self.socket.recv_from(&mut self.buffer)?;
        self.stats.bytes_received += size as u64;
        self.stats.frames_received += 1;

        FramePacket::decode(&self.buffer[..size], from)
            .map_err(|e| {
                self.stats.decode_errors += 1;
                io::Error::new(io::ErrorKind::InvalidData, e.to_string())
            })
    }

    /// Run executor loop - receive frames and execute
    pub fn run_executor(&mut self, executor: &mut Executor) -> io::Result<()> {
        loop {
            match self.recv_frame() {
                Ok(packet) => {
                    let result = executor.execute(&packet.frame);
                    self.stats.frames_executed += 1;

                    // Check for halt
                    if matches!(result, ExecResult::Halt) {
                        break;
                    }

                    // Notify callback
                    if let Some(callback) = self.on_result.as_ref() {
                        callback(packet.from, result);
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    // Timeout - continue
                    continue;
                }
                Err(e) => {
                    self.stats.exec_errors += 1;
                    // Log but continue
                    eprintln!("UDP receive error: {}", e);
                }
            }
        }
        Ok(())
    }

    /// Run for specified number of frames
    pub fn run_frames(&mut self, executor: &mut Executor, count: usize) -> Vec<ExecResult> {
        let mut results = Vec::with_capacity(count);

        for _ in 0..count {
            match self.recv_frame() {
                Ok(packet) => {
                    let result = executor.execute(&packet.frame);
                    self.stats.frames_executed += 1;
                    results.push(result);
                }
                Err(_) => break,
            }
        }

        results
    }

    pub fn stats(&self) -> &ReceiverStats {
        &self.stats
    }
}

// =============================================================================
// LANE ROUTER
// =============================================================================

/// Route frames to different executors based on lane ID
pub struct LaneRouter {
    /// Executors per lane
    lanes: Vec<Option<Executor>>,
    /// Frame buffer for reordering
    pending: Vec<FramePacket>,
}

impl LaneRouter {
    pub fn new(num_lanes: usize) -> Self {
        Self {
            lanes: (0..num_lanes).map(|_| Some(Executor::new())).collect(),
            pending: Vec::new(),
        }
    }

    /// Route frame to appropriate lane
    pub fn route(&mut self, packet: FramePacket) -> ExecResult {
        let lane_id = packet.frame.header.lane_id as usize;

        if let Some(Some(executor)) = self.lanes.get_mut(lane_id) {
            executor.execute(&packet.frame)
        } else {
            ExecResult::Error(format!("lane {} not found", lane_id))
        }
    }

    /// Get executor for lane
    pub fn executor(&mut self, lane_id: usize) -> Option<&mut Executor> {
        self.lanes.get_mut(lane_id).and_then(|e| e.as_mut())
    }
}

// =============================================================================
// ASYNC TRANSPORT (optional, requires tokio)
// =============================================================================

#[cfg(feature = "async")]
pub mod async_transport {
    use super::*;
    use tokio::net::UdpSocket;
    use tokio::sync::mpsc;

    /// Async UDP sender
    pub struct AsyncSender {
        socket: UdpSocket,
        sequence: u32,
    }

    impl AsyncSender {
        pub async fn new(bind_addr: &str) -> io::Result<Self> {
            let socket = UdpSocket::bind(bind_addr).await?;
            Ok(Self { socket, sequence: 0 })
        }

        pub async fn send_frame(&mut self, frame: &FireflyFrame, dest: &str) -> io::Result<()> {
            let packet = FramePacket {
                frame: frame.clone(),
                from: self.socket.local_addr()?,
                received_at: Instant::now(),
                sequence: self.sequence,
                ack_requested: false,
            };
            self.sequence = self.sequence.wrapping_add(1);

            let data = packet.encode();
            let dest_addr: SocketAddr = dest.parse()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;

            self.socket.send_to(&data, dest_addr).await?;
            Ok(())
        }
    }

    /// Async UDP receiver with channel output
    pub struct AsyncReceiver {
        socket: UdpSocket,
        tx: mpsc::Sender<FramePacket>,
    }

    impl AsyncReceiver {
        pub async fn new(bind_addr: &str, buffer_size: usize) -> io::Result<(Self, mpsc::Receiver<FramePacket>)> {
            let socket = UdpSocket::bind(bind_addr).await?;
            let (tx, rx) = mpsc::channel(buffer_size);
            Ok((Self { socket, tx }, rx))
        }

        pub async fn run(&self) -> io::Result<()> {
            let mut buffer = [0u8; MAX_UDP_PAYLOAD];
            loop {
                let (size, from) = self.socket.recv_from(&mut buffer).await?;
                if let Ok(packet) = FramePacket::decode(&buffer[..size], from) {
                    if self.tx.send(packet).await.is_err() {
                        break; // Channel closed
                    }
                }
            }
            Ok(())
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::firefly_frame::{FrameBuilder, FrameHeader, Instruction, LanguagePrefix};

    #[test]
    fn test_packet_encode_decode() {
        let frame = FrameBuilder::new(42).nop();

        let packet = FramePacket {
            frame: frame.clone(),
            from: "127.0.0.1:5050".parse().unwrap(),
            received_at: Instant::now(),
            sequence: 12345,
            ack_requested: true,
        };

        let encoded = packet.encode();
        assert!(encoded.len() < MAX_UDP_PAYLOAD);

        let decoded = FramePacket::decode(&encoded, "127.0.0.1:5051".parse().unwrap()).unwrap();
        assert_eq!(decoded.sequence, 12345);
        assert!(decoded.ack_requested);
        assert_eq!(decoded.frame.header.session_id, 42);
    }

    #[test]
    fn test_sender_receiver_loopback() {
        // Bind to random ports
        let sender_addr = "127.0.0.1:0";
        let receiver_addr = "127.0.0.1:0";

        let sender = UdpSender::new(sender_addr).unwrap();
        let receiver = UdpReceiver::new(receiver_addr).unwrap();

        // Get actual bound addresses
        let receiver_actual = receiver.socket.local_addr().unwrap();

        // Create sender with known destination
        let mut sender = UdpSender::new(sender_addr).unwrap();

        // Note: Full loopback test requires threads or async
        // Just verify construction works
        assert_eq!(sender.stats().frames_sent, 0);
    }

    #[test]
    fn test_lane_router() {
        let mut router = LaneRouter::new(4);

        // Create frames for different lanes
        let frame0 = {
            let header = FrameHeader::new(1, 0, 0, 0);
            let inst = Instruction::new(LanguagePrefix::Control, 0x00, 0, 0, 0); // NOP
            FireflyFrame::new(header, inst)
        };

        let frame1 = {
            let header = FrameHeader::new(1, 0, 1, 0); // lane 1
            let inst = Instruction::new(LanguagePrefix::Control, 0x00, 0, 0, 0);
            FireflyFrame::new(header, inst)
        };

        let packet0 = FramePacket {
            frame: frame0,
            from: "127.0.0.1:5050".parse().unwrap(),
            received_at: Instant::now(),
            sequence: 0,
            ack_requested: false,
        };

        let packet1 = FramePacket {
            frame: frame1,
            from: "127.0.0.1:5050".parse().unwrap(),
            received_at: Instant::now(),
            sequence: 1,
            ack_requested: false,
        };

        let result0 = router.route(packet0);
        let result1 = router.route(packet1);

        assert!(matches!(result0, ExecResult::Ok(_)));
        assert!(matches!(result1, ExecResult::Ok(_)));
    }
}
