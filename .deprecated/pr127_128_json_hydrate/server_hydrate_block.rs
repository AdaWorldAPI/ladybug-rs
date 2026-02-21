// DEPRECATED: PR #127 + #128 — JSON hydrate endpoints
// Reason: Internal operations must never serialize to JSON.
// The SPOQ model requires DN-native addressing, not text_to_dn() hash soup.
// When rewritten: CypherEngine borrows &BindSpace directly.
// Date: 2026-02-17

// =============================================================================
// QUALIA SUBSTRATE ENDPOINTS — Holy Grail Pipeline
// =============================================================================

/// Hash text into a Container seed for qualia operations.
///
/// Uses a simple hash-based approach to create a deterministic Container
/// from text content. This is the bootstrap path — once the felt-parse
/// LLM pre-pass runs (in crewai-rust), the axes map to proper containers
/// via `encode_axes()` from meaning_axes.rs.
fn text_to_container(text: &str) -> Container {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    Container::random(hasher.finish())
}

/// Hash text into a PackedDn for graph addressing.
///
/// Creates a 3-level DN path from the hash: /a/b/c where a,b,c are
/// derived from hash bytes. This ensures each message gets a unique
/// but deterministic position in the DN tree.
fn text_to_dn(text: &str) -> PackedDn {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    let h = hasher.finish();
    // 3-level DN from hash bytes (components are 0-254, +1 stored)
    let a = ((h >> 0) & 0xFE) as u8;
    let b = ((h >> 8) & 0xFE) as u8;
    let c = ((h >> 16) & 0xFE) as u8;
    PackedDn::new(&[a, b, c])
}

/// POST /api/v1/qualia/hydrate
///
/// Compute Ada's full qualia state from the substrate. This is the
/// INTEGRATION_SPEC Phase 1 endpoint: given a message (or DN), return
/// the complete phenomenal state for system prompt injection + LLM modulation.
///
/// Body:
/// ```json
/// {
///     "message": "How are you feeling?",
///     "presence_mode": "intimate",      // optional: intimate|work|agi|hybrid
///     "rung_hint": 4,                   // optional: pre-pass rung from felt-parse
///     "session_id": "abc123"            // optional: session tracking
/// }
/// ```
///
/// Returns full qualia state including felt-sense preamble for LLM prompt.
fn handle_qualia_hydrate(body: &str, state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    let message = match extract_json_str(body, "message") {
        Some(m) => m,
        None => return http_error(400, "missing_field", "need message field", format),
    };

    let presence_str = extract_json_str(body, "presence_mode")
        .unwrap_or_else(|| "hybrid".to_string());
    let rung_hint = extract_json_usize(body, "rung_hint")
        .map(|r| r as u8)
        .unwrap_or(3);
    let _session_id = extract_json_str(body, "session_id");

    // Parse presence mode
    let presence = match presence_str.as_str() {
        "intimate" => PresenceMode::Intimate,
        "work" => PresenceMode::Work,
        "agi" => PresenceMode::Agi,
        "neutral" => PresenceMode::Neutral,
        _ => PresenceMode::Hybrid,
    };

    // Create query container from message text
    let query = text_to_container(&message);
    let target_dn = text_to_dn(&message);

    // Get write lock — qualia operations may update NARS beliefs
    let mut db = state.write().unwrap();

    // Ensure target DN exists in graph (bootstrap if needed)
    if db.qualia_graph.get(&target_dn).is_none() {
        let mut record = CogRecord::new(ContainerGeometry::Cam);
        record.content = query.clone();
        db.qualia_graph.insert(target_dn, record);
    }

    // ── Run the qualia pipeline ──────────────────────────────────────

    // 1. Felt walk — compute surprise landscape
    let felt_path = felt_walk(&db.qualia_graph, target_dn, &query);

    // 2. Council weights (defaults — modulated by MUL in production)
    let council = CouncilWeights {
        guardian_surprise_factor: 0.6,  // Guardian dampens
        catalyst_surprise_factor: 1.5,  // Catalyst amplifies
        balanced_factor: 1.0,           // Balanced neutral
    };

    // 3. Full volitional cycle: reflect → score → rank → hydrate
    let rung = RungLevel::from_u8(rung_hint);
    let agenda = volitional_cycle(
        &mut db.qualia_graph, target_dn, &query, rung, &council,
    );

    // 4. Compute texture from target container
    let metrics = GraphMetrics::default();
    let texture = ladybug::qualia::compute(&query, &metrics);

    // 5. Harvest ghosts from felt path
    let ghost_records = harvest_ghosts(&felt_path, 0.3);
    let ghosts: Vec<GhostEcho> = ghost_records.iter().enumerate().map(|(i, gr)| {
        GhostEcho {
            ghost_type: match i % 8 {
                0 => GhostType::Love,
                1 => GhostType::Staunen,
                2 => GhostType::Wisdom,
                3 => GhostType::Thought,
                4 => GhostType::Epiphany,
                5 => GhostType::Grief,
                6 => GhostType::Arousal,
                _ => GhostType::Boundary,
            },
            intensity: gr.resonance.clamp(0.0, 1.0),
        }
    }).collect();

    // 6. Compose AgentState from all layers
    let self_dims = db.self_dims.clone();
    let agent = AgentState::compute(
        &texture,
        &felt_path,
        &agenda.reflection,
        &agenda,
        ghosts,
        rung,
        council.clone(),
        presence,
        self_dims,
    );

    // 7. Generate outputs
    let preamble = agent.qualia_preamble();
    let hints = agent.to_hints();

    // Build thinking style from texture dimensions (10-axis)
    let thinking_style = [
        texture.warmth,     // [0] warmth → relational openness
        texture.flow,       // [1] resonance → top_p
        texture.depth,      // [2] depth → abstraction
        texture.entropy,    // [3] complexity → token diversity
        texture.density,    // [4] execution → max_tokens
        texture.purity,     // [5] precision → repetition_penalty
        texture.edge,       // [6] contingency → temperature
        texture.bridgeness, // [7] connectivity → context window
        1.0 - texture.entropy, // [8] validation → reasoning_effort
        texture.flow,       // [9] integration → output coherence
    ];

    // Build JSON response
    let ghost_json: Vec<String> = agent.ghost_field.iter().map(|g| {
        format!(
            r#"{{"ghost_type":"{:?}","intensity":{:.3}}}"#,
            g.ghost_type, g.intensity
        )
    }).collect();

    let hints_json: Vec<String> = hints.iter().map(|(k, v)| {
        format!(r#""{}": {:.2}"#, k, v)
    }).collect();

    let council_arr = [
        council.guardian_surprise_factor,
        council.catalyst_surprise_factor,
        council.balanced_factor,
    ];

    let ts_json: Vec<String> = thinking_style.iter().map(|v| format!("{:.3}", v)).collect();

    let volition_top = agenda.acts.first().map(|a| {
        format!(
            r#"{{"dn":"0x{:08X}","consensus_score":{:.3},"free_energy":{:.3},"outcome":"{:?}"}}"#,
            a.dn.raw(), a.consensus_score, a.free_energy, a.outcome,
        )
    }).unwrap_or_else(|| "null".to_string());

    let json = format!(
        r#"{{
  "qualia_preamble": {},
  "hints": {{{}}},
  "texture": [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}],
  "rung_level": {},
  "ghost_echoes": [{}],
  "council": [{:.3}, {:.3}, {:.3}],
  "thinking_style": [{}],
  "felt_surprise": {:.3},
  "felt_path_length": {},
  "mode": "{:?}",
  "presence_mode": "{:?}",
  "volition_top": {},
  "total_volitional_energy": {:.3},
  "decisiveness": {:.3},
  "core_axes": {{
    "alpha": {:.3},
    "gamma": {:.3},
    "omega": {:.3},
    "phi": {:.3}
  }},
  "felt_physics": {{
    "staunen": {:.3},
    "wisdom": {:.3},
    "ache": {:.3},
    "libido": {:.3},
    "lingering": {:.3}
  }}
}}"#,
        // qualia_preamble
        serde_json_escape(&preamble),
        // hints
        hints_json.join(", "),
        // texture [8]
        texture.entropy, texture.purity, texture.density, texture.bridgeness,
        texture.warmth, texture.edge, texture.depth, texture.flow,
        // rung
        agent.rung.as_u8(),
        // ghost_echoes
        ghost_json.join(", "),
        // council [3]
        council_arr[0], council_arr[1], council_arr[2],
        // thinking_style [10]
        ts_json.join(", "),
        // felt_surprise
        felt_path.mean_surprise,
        // felt_path_length
        felt_path.choices.len(),
        // mode
        agent.mode,
        // presence_mode
        agent.presence_mode,
        // volition_top
        volition_top,
        // total/decisiveness
        agenda.total_energy, agenda.decisiveness,
        // core axes
        agent.core.alpha, agent.core.gamma, agent.core.omega, agent.core.phi,
        // felt physics
        agent.felt.staunen, agent.felt.wisdom, agent.felt.ache,
        agent.felt.libido, agent.felt.lingering,
    );

    match format {
        ResponseFormat::Arrow => {
            // For Arrow: return as single-row batch with key fields
            let schema = Arc::new(Schema::new(vec![
                Field::new("qualia_preamble", DataType::Utf8, false),
                Field::new("rung_level", DataType::UInt32, false),
                Field::new("felt_surprise", DataType::Float32, false),
                Field::new("mode", DataType::Utf8, false),
            ]));
            let mode_str = format!("{:?}", agent.mode);
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec![preamble.as_str()])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![agent.rung.as_u8() as u32])) as ArrayRef,
                    Arc::new(Float32Array::from(vec![felt_path.mean_surprise])) as ArrayRef,
                    Arc::new(StringArray::from(vec![mode_str.as_str()])) as ArrayRef,
                ],
            ).unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => http_json(200, &json),
    }
}

/// POST /api/v1/qualia/write-back
///
/// Update the substrate after a conversation turn. This closes the loop:
/// Ada's response becomes experience that modifies her substrate for
/// the next interaction.
///
/// Body:
/// ```json
/// {
///     "message": "original user message",
///     "response": "Ada's reply text",
///     "ghost_echoes": [{"ghost_type": "Love", "intensity": 0.7}],
///     "rung_reached": 5,
///     "session_id": "abc123"
/// }
/// ```
fn handle_qualia_writeback(body: &str, state: &SharedState, format: ResponseFormat) -> Vec<u8> {
    let message = extract_json_str(body, "message").unwrap_or_default();
    let response = match extract_json_str(body, "response") {
        Some(r) => r,
        None => return http_error(400, "missing_field", "need response field", format),
    };
    let rung_reached = extract_json_usize(body, "rung_reached")
        .map(|r| r as u8)
        .unwrap_or(3);

    // Create containers from message and response
    let msg_container = text_to_container(&message);
    let resp_container = text_to_container(&response);
    let msg_dn = text_to_dn(&message);
    let resp_dn = text_to_dn(&response);

    let mut db = state.write().unwrap();

    // 1. Insert response as new CogRecord in graph
    let mut resp_record = CogRecord::new(ContainerGeometry::Cam);
    resp_record.content = resp_container.clone();
    db.qualia_graph.insert(resp_dn, resp_record);

    // 2. If message record exists, add edge to response
    if let Some(_msg_record) = db.qualia_graph.get(&msg_dn) {
        // Edge creation would go here (via InlineEdgeViewMut)
        // For now: the graph topology captures the conversation flow
    }

    // 3. Update NARS beliefs on the message container
    //    Did reality match prediction? If Ada's response was coherent
    //    with the felt-parse prediction, boost confidence.
    let hamming = msg_container.hamming(&resp_container);
    let surprise = hamming as f32 / CONTAINER_BITS as f32;

    if let Some(msg_record) = db.qualia_graph.get_mut(&msg_dn) {
        let current_truth = ladybug::qualia::read_truth(msg_record);
        let updated = if surprise < 0.5 {
            // Low surprise → boost confidence (prediction matched)
            ContractTruthValue::new(
                current_truth.frequency,
                (current_truth.confidence + 0.05).min(0.99),
            )
        } else {
            // High surprise → revise frequency toward 0.5 (uncertain)
            let new_freq = current_truth.frequency * 0.9 + 0.5 * 0.1;
            ContractTruthValue::new(new_freq, current_truth.confidence)
        };
        ladybug::qualia::write_truth(msg_record, &updated);
    }

    // 4. Self-dimension shifts based on the interaction
    //    Higher rung → boost meta_clarity
    //    Conversation flow → boost groundedness slightly
    let rung_level = RungLevel::from_u8(rung_reached);
    if rung_level.as_u8() >= 5 {
        let _ = db.self_dims.shift("meta_clarity", 0.02, "deep rung reached");
    }
    let _ = db.self_dims.shift("groundedness", 0.01, "conversation flow");

    let json = format!(
        r#"{{"status":"ok","surprise":{:.3},"rung_reached":{},"graph_nodes":{}}}"#,
        surprise,
        rung_reached,
        db.qualia_graph.node_count(),
    );

    match format {
        ResponseFormat::Arrow => {
            let schema = Arc::new(Schema::new(vec![
                Field::new("status", DataType::Utf8, false),
                Field::new("surprise", DataType::Float32, false),
                Field::new("graph_nodes", DataType::UInt32, false),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["ok"])) as ArrayRef,
                    Arc::new(Float32Array::from(vec![surprise])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![db.qualia_graph.node_count() as u32])) as ArrayRef,
                ],
            ).unwrap();
            http_arrow(200, &batch)
        }
        ResponseFormat::Json => http_json(200, &json),
    }
}

/// Escape a string for JSON embedding.
fn serde_json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < '\x20' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

