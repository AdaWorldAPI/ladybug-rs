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

use crate::storage::bind_space::{Addr, BindSpace, FINGERPRINT_WORDS, PREFIX_A2A};
use serde::{Deserialize, Serialize};

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
    /// Persona exchange (feature-aware A2A customization)
    PersonaExchange,
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
    /// Resonance weight — how strongly this message should resonate in the field.
    /// Higher weight messages dominate the XOR superposition.
    /// Default 1.0 (equal contribution).
    #[serde(default = "default_resonance_weight")]
    pub resonance_weight: f32,
}

fn default_resonance_weight() -> f32 {
    1.0
}

impl A2AMessage {
    /// Compute the channel address for this message
    pub fn channel_addr(&self) -> Addr {
        let channel = compute_channel(self.sender_slot, self.receiver_slot);
        Addr::new(PREFIX_A2A, channel)
    }

    /// Encode message metadata into a fingerprint for channel storage
    pub fn to_fingerprint(&self) -> [u64; FINGERPRINT_WORDS] {
        use sha2::{Digest, Sha256};

        let serialized = format!(
            "{}:{}:{}:{:?}:{}",
            self.id, self.sender_slot, self.receiver_slot, self.kind, self.payload
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

/// A2A channel — a named communication path between two agents.
///
/// # XOR Field Resonance Superposition
///
/// Concurrent writes to the same channel compose naturally through XOR.
/// This is a **feature**, not an error: the channel fingerprint encodes
/// the superposition of ALL messages sent through it. Each new message
/// XOR-composes into the field, and any known message can be extracted
/// via XOR-unbinding (A ⊗ B ⊗ B = A).
///
/// This means:
/// - Two agents writing simultaneously → both messages are encoded
/// - The field resonance (popcount of the superposition) measures
///   channel activity and awareness density
/// - High resonance = many distinct messages = high channel awareness
/// - Low resonance = few/repeated messages = quiet channel
#[derive(Clone, Debug)]
pub struct A2AChannel {
    pub sender_slot: u8,
    pub receiver_slot: u8,
    pub channel_slot: u8,
    pub message_count: u32,
    /// Channel resonance — popcount of the XOR superposition field
    /// normalized to [0.0, 1.0]. Measures awareness density.
    pub field_resonance: f32,
    /// Number of distinct XOR compositions in the field
    pub superposition_depth: u32,
}

impl A2AChannel {
    pub fn new(sender: u8, receiver: u8) -> Self {
        Self {
            sender_slot: sender,
            receiver_slot: receiver,
            channel_slot: compute_channel(sender, receiver),
            message_count: 0,
            field_resonance: 0.0,
            superposition_depth: 0,
        }
    }

    pub fn addr(&self) -> Addr {
        Addr::new(PREFIX_A2A, self.channel_slot)
    }

    /// Update field resonance from the current channel fingerprint
    pub fn update_resonance(&mut self, fingerprint: &[u64; FINGERPRINT_WORDS]) {
        let popcount: u32 = fingerprint.iter().map(|w| w.count_ones()).sum();
        let max_bits = (FINGERPRINT_WORDS * 64) as f32;
        self.field_resonance = popcount as f32 / max_bits;
    }
}

/// Compute channel slot from sender/receiver pair
/// Uses XOR + rotation to distribute channels across the 256-slot space
fn compute_channel(sender: u8, receiver: u8) -> u8 {
    // XOR sender and receiver, then mix with rotation to reduce collisions
    let mixed = sender ^ receiver;
    let rotated = sender
        .wrapping_mul(17)
        .wrapping_add(receiver.wrapping_mul(31));
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

    /// Send a message through the protocol.
    ///
    /// # XOR Superposition
    ///
    /// Messages compose into the channel via XOR. This means concurrent
    /// writes are naturally handled — the field accumulates all messages
    /// as a superposition. Any known message can be extracted later via
    /// XOR-unbinding (the XOR self-inverse property).
    ///
    /// The channel's `field_resonance` and `superposition_depth` are
    /// updated after each send, reflecting the growing awareness density.
    pub fn send(&mut self, msg: A2AMessage, space: &mut BindSpace) -> DeliveryStatus {
        let addr = msg.channel_addr();
        let fp = msg.to_fingerprint();
        let channel_slot = compute_channel(msg.sender_slot, msg.receiver_slot);

        // XOR-compose into the channel's fingerprint slot.
        // This IS the superposition: each message adds to the field.
        // Concurrent writes naturally compose — XOR is associative and commutative.
        if let Some(node) = space.read(addr) {
            let mut composed = node.fingerprint;
            for (i, word) in composed.iter_mut().enumerate() {
                *word ^= fp[i];
            }
            space.write_at(addr, composed);

            // Update channel resonance from the new superposition
            if let Some(channel) = self
                .channels
                .iter_mut()
                .find(|c| c.channel_slot == channel_slot)
            {
                channel.superposition_depth += 1;
                channel.message_count += 1;
                channel.update_resonance(&composed);
            }
        } else {
            space.write_at(addr, fp);

            // First message — resonance is the message's own density
            if let Some(channel) = self
                .channels
                .iter_mut()
                .find(|c| c.channel_slot == channel_slot)
            {
                channel.superposition_depth = 1;
                channel.message_count += 1;
                channel.update_resonance(&fp);
            }
        }

        if let Some(node) = space.read_mut(addr) {
            node.label = Some(format!("a2a:{}->{}", msg.sender_slot, msg.receiver_slot));
        }

        self.pending_messages.push(msg);
        DeliveryStatus::Delivered
    }

    /// Read the current XOR superposition field for a channel.
    /// This is the accumulated resonance of all messages in the channel.
    pub fn read_field(
        &self,
        space: &BindSpace,
        sender: u8,
        receiver: u8,
    ) -> Option<[u64; FINGERPRINT_WORDS]> {
        let channel_slot = compute_channel(sender, receiver);
        let addr = Addr::new(PREFIX_A2A, channel_slot);
        space.read(addr).map(|n| n.fingerprint)
    }

    /// Get field resonance for a channel (0.0-1.0).
    /// Measures awareness density — how much information the channel carries.
    pub fn field_resonance(&self, sender: u8, receiver: u8) -> f32 {
        let channel_slot = compute_channel(sender, receiver);
        self.channels
            .iter()
            .find(|c| c.channel_slot == channel_slot)
            .map(|c| c.field_resonance)
            .unwrap_or(0.0)
    }

    /// Get superposition depth for a channel — how many distinct messages
    /// have been XOR-composed into the field.
    pub fn superposition_depth(&self, sender: u8, receiver: u8) -> u32 {
        let channel_slot = compute_channel(sender, receiver);
        self.channels
            .iter()
            .find(|c| c.channel_slot == channel_slot)
            .map(|c| c.superposition_depth)
            .unwrap_or(0)
    }

    /// Compute total awareness across all channels.
    /// This is the sum of field resonances — a measure of the system's
    /// overall inter-agent awareness density.
    pub fn total_awareness(&self) -> f32 {
        self.channels.iter().map(|c| c.field_resonance).sum()
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
            resonance_weight: 1.0,
        };

        let fp1 = msg.to_fingerprint();
        let fp2 = msg.to_fingerprint();
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_xor_superposition_accumulates() {
        let mut protocol = A2AProtocol::new();
        let mut space = BindSpace::new();

        let _ch = protocol.open_channel(0, 1);

        let msg1 = A2AMessage {
            id: "msg-1".to_string(),
            sender_slot: 0,
            receiver_slot: 1,
            kind: MessageKind::Knowledge,
            payload: "first insight".to_string(),
            fingerprint: None,
            timestamp: 1,
            status: DeliveryStatus::Pending,
            thinking_style_hint: None,
            resonance_weight: 1.0,
        };

        let msg2 = A2AMessage {
            id: "msg-2".to_string(),
            sender_slot: 0,
            receiver_slot: 1,
            kind: MessageKind::Knowledge,
            payload: "second insight".to_string(),
            fingerprint: None,
            timestamp: 2,
            status: DeliveryStatus::Pending,
            thinking_style_hint: None,
            resonance_weight: 1.0,
        };

        protocol.send(msg1, &mut space);
        let depth1 = protocol.superposition_depth(0, 1);

        protocol.send(msg2, &mut space);
        let depth2 = protocol.superposition_depth(0, 1);

        // Superposition depth should increase with each message
        assert_eq!(depth1, 1);
        assert_eq!(depth2, 2);

        // Field resonance should be non-zero
        let resonance = protocol.field_resonance(0, 1);
        assert!(
            resonance > 0.0,
            "Field resonance should be positive: {}",
            resonance
        );
    }

    #[test]
    fn test_xor_unbind_recovers_from_superposition() {
        // When two messages are XOR-composed, knowing one lets you extract the other
        let msg_a = A2AMessage {
            id: "a".to_string(),
            sender_slot: 0,
            receiver_slot: 1,
            kind: MessageKind::Knowledge,
            payload: "alpha".to_string(),
            fingerprint: None,
            timestamp: 1,
            status: DeliveryStatus::Pending,
            thinking_style_hint: None,
            resonance_weight: 1.0,
        };
        let msg_b = A2AMessage {
            id: "b".to_string(),
            sender_slot: 0,
            receiver_slot: 1,
            kind: MessageKind::Knowledge,
            payload: "beta".to_string(),
            fingerprint: None,
            timestamp: 2,
            status: DeliveryStatus::Pending,
            thinking_style_hint: None,
            resonance_weight: 1.0,
        };

        let fp_a = msg_a.to_fingerprint();
        let fp_b = msg_b.to_fingerprint();

        // Superposition: A ⊗ B
        let mut superposition = [0u64; FINGERPRINT_WORDS];
        for i in 0..FINGERPRINT_WORDS {
            superposition[i] = fp_a[i] ^ fp_b[i];
        }

        // Unbind B to recover A: (A ⊗ B) ⊗ B = A
        let mut recovered = [0u64; FINGERPRINT_WORDS];
        for i in 0..FINGERPRINT_WORDS {
            recovered[i] = superposition[i] ^ fp_b[i];
        }

        assert_eq!(recovered, fp_a);
    }

    #[test]
    fn test_total_awareness() {
        let mut protocol = A2AProtocol::new();
        let mut space = BindSpace::new();

        let _ch1 = protocol.open_channel(0, 1);
        let _ch2 = protocol.open_channel(2, 3);

        let msg1 = A2AMessage {
            id: "m1".to_string(),
            sender_slot: 0,
            receiver_slot: 1,
            kind: MessageKind::Status,
            payload: "hi".to_string(),
            fingerprint: None,
            timestamp: 1,
            status: DeliveryStatus::Pending,
            thinking_style_hint: None,
            resonance_weight: 1.0,
        };
        let msg2 = A2AMessage {
            id: "m2".to_string(),
            sender_slot: 2,
            receiver_slot: 3,
            kind: MessageKind::Status,
            payload: "hello".to_string(),
            fingerprint: None,
            timestamp: 2,
            status: DeliveryStatus::Pending,
            thinking_style_hint: None,
            resonance_weight: 1.0,
        };

        protocol.send(msg1, &mut space);
        protocol.send(msg2, &mut space);

        let awareness = protocol.total_awareness();
        assert!(
            awareness > 0.0,
            "Total awareness should be positive: {}",
            awareness
        );
    }
}
