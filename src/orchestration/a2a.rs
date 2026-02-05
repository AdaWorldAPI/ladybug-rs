//! A2A — Agent-to-Agent Protocol via BindSpace 0x0F
//!
//! Non-blocking message routing between agents through dedicated BindSpace
//! addresses. Each A2A channel occupies a slot in prefix 0x0F.
//!
//! # Channel Addressing
//!
//! ```text
//! 0x0F:XX where XX = hash(sender_slot, receiver_slot) & 0xFF
//! ```
//!
//! Messages are serialized into the fingerprint field using XOR composition,
//! allowing multiple messages to be stacked and retrieved via unbinding.
//!
//! # Protocol Flow
//!
//! ```text
//! Agent A (0x0C:02) ──► encode message ──► XOR bind to channel ──► 0x0F:hash(02,05)
//!                                                                        │
//! Agent B (0x0C:05) ◄── decode message ◄── XOR unbind from channel ◄─────┘
//! ```

use serde::{Deserialize, Serialize};
use crate::storage::bind_space::{
    Addr, BindSpace, FINGERPRINT_WORDS, PREFIX_A2A,
};

/// Message kind for A2A communication
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum MessageKind {
    /// Task delegation request
    Delegate,
    /// Result from delegated task
    Result,
    /// Status update / heartbeat
    Status,
    /// Knowledge sharing (fingerprint payload)
    Knowledge,
    /// Coordination signal (sync point)
    Sync,
    /// Query to another agent
    Query,
    /// Response to a query
    Response,
}

/// Delivery status for A2A messages
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeliveryStatus {
    Pending,
    Delivered,
    Acknowledged,
    Failed,
}

/// A2A message between agents
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct A2AMessage {
    pub id: String,
    pub sender_slot: u8,
    pub receiver_slot: u8,
    pub kind: MessageKind,
    pub payload: String,
    /// Optional fingerprint payload for knowledge transfer
    #[serde(skip)]
    pub fingerprint: Option<[u64; FINGERPRINT_WORDS]>,
    pub timestamp: u64,
    pub status: DeliveryStatus,
    /// Thinking style hint for the receiver
    pub thinking_style_hint: Option<String>,
}

impl A2AMessage {
    /// Compute the channel address for this message
    pub fn channel_addr(&self) -> Addr {
        let channel = compute_channel(self.sender_slot, self.receiver_slot);
        Addr::new(PREFIX_A2A, channel)
    }

    /// Encode message metadata into a fingerprint for channel storage
    pub fn to_fingerprint(&self) -> [u64; FINGERPRINT_WORDS] {
        use sha2::{Sha256, Digest};

        let serialized = format!(
            "{}:{}:{}:{:?}:{}",
            self.id, self.sender_slot, self.receiver_slot,
            self.kind, self.payload
        );

        let mut hasher = Sha256::new();
        hasher.update(serialized.as_bytes());
        let hash = hasher.finalize();

        let mut fp = [0u64; FINGERPRINT_WORDS];
        for (i, word) in fp.iter_mut().enumerate() {
            let mut h = Sha256::new();
            h.update(&hash);
            h.update(&(i as u32).to_le_bytes());
            let block = h.finalize();
            *word = u64::from_le_bytes(block[..8].try_into().unwrap());
        }
        fp
    }
}

/// A2A channel — a named communication path between two agents
#[derive(Clone, Debug)]
pub struct A2AChannel {
    pub sender_slot: u8,
    pub receiver_slot: u8,
    pub channel_slot: u8,
    pub message_count: u32,
}

impl A2AChannel {
    pub fn new(sender: u8, receiver: u8) -> Self {
        Self {
            sender_slot: sender,
            receiver_slot: receiver,
            channel_slot: compute_channel(sender, receiver),
            message_count: 0,
        }
    }

    pub fn addr(&self) -> Addr {
        Addr::new(PREFIX_A2A, self.channel_slot)
    }
}

/// Compute channel slot from sender/receiver pair
/// Uses XOR + rotation to distribute channels across the 256-slot space
fn compute_channel(sender: u8, receiver: u8) -> u8 {
    // XOR sender and receiver, then mix with rotation to reduce collisions
    let mixed = sender ^ receiver;
    let rotated = sender.wrapping_mul(17).wrapping_add(receiver.wrapping_mul(31));
    mixed ^ rotated
}

/// A2A protocol manager
pub struct A2AProtocol {
    channels: Vec<A2AChannel>,
    pending_messages: Vec<A2AMessage>,
}

impl A2AProtocol {
    pub fn new() -> Self {
        Self {
            channels: Vec::new(),
            pending_messages: Vec::new(),
        }
    }

    /// Open a channel between two agents
    pub fn open_channel(&mut self, sender: u8, receiver: u8) -> A2AChannel {
        let channel = A2AChannel::new(sender, receiver);
        self.channels.push(channel.clone());
        channel
    }

    /// Send a message through the protocol
    pub fn send(&mut self, msg: A2AMessage, space: &mut BindSpace) -> DeliveryStatus {
        let addr = msg.channel_addr();
        let fp = msg.to_fingerprint();

        // XOR-compose into the channel's fingerprint slot
        if let Some(node) = space.read(addr) {
            let mut composed = node.fingerprint;
            for (i, word) in composed.iter_mut().enumerate() {
                *word ^= fp[i];
            }
            space.write_at(addr, composed);
        } else {
            space.write_at(addr, fp);
        }

        if let Some(node) = space.read_mut(addr) {
            node.label = Some(format!(
                "a2a:{}->{}",
                msg.sender_slot, msg.receiver_slot
            ));
        }

        self.pending_messages.push(msg);
        DeliveryStatus::Delivered
    }

    /// Receive pending messages for a given agent slot
    pub fn receive(&mut self, receiver_slot: u8) -> Vec<A2AMessage> {
        let mut received = Vec::new();
        self.pending_messages.retain(|msg| {
            if msg.receiver_slot == receiver_slot && msg.status == DeliveryStatus::Pending {
                let mut msg = msg.clone();
                msg.status = DeliveryStatus::Delivered;
                received.push(msg);
                false
            } else {
                true
            }
        });
        received
    }

    /// List active channels
    pub fn channels(&self) -> &[A2AChannel] {
        &self.channels
    }

    /// Get pending message count for a receiver
    pub fn pending_for(&self, receiver_slot: u8) -> usize {
        self.pending_messages
            .iter()
            .filter(|m| m.receiver_slot == receiver_slot && m.status == DeliveryStatus::Pending)
            .count()
    }
}

impl Default for A2AProtocol {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_addressing() {
        let ch = A2AChannel::new(0x02, 0x05);
        assert_eq!(ch.addr().prefix(), PREFIX_A2A);
        // Channel slot is deterministic
        let ch2 = A2AChannel::new(0x02, 0x05);
        assert_eq!(ch.channel_slot, ch2.channel_slot);
    }

    #[test]
    fn test_channel_asymmetry() {
        // sender→receiver and receiver→sender should use different channels
        let ch_ab = A2AChannel::new(0x02, 0x05);
        let ch_ba = A2AChannel::new(0x05, 0x02);
        assert_ne!(ch_ab.channel_slot, ch_ba.channel_slot);
    }

    #[test]
    fn test_message_fingerprint_deterministic() {
        let msg = A2AMessage {
            id: "msg-1".to_string(),
            sender_slot: 0,
            receiver_slot: 1,
            kind: MessageKind::Delegate,
            payload: "do the thing".to_string(),
            fingerprint: None,
            timestamp: 12345,
            status: DeliveryStatus::Pending,
            thinking_style_hint: Some("analytical".to_string()),
        };

        let fp1 = msg.to_fingerprint();
        let fp2 = msg.to_fingerprint();
        assert_eq!(fp1, fp2);
    }
}
