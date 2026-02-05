//! crewAI Flight Actions — DoAction handlers for orchestration
//!
//! Adds the following Flight DoAction endpoints:
//!
//! | Action | Description | Arrow IPC Params |
//! |--------|-------------|-----------------|
//! | `crew.register_agent` | Register agent from YAML | {yaml: binary} |
//! | `crew.register_style` | Register thinking template | {yaml: binary} |
//! | `crew.submit_task` | Submit task for dispatch | {json: binary} |
//! | `crew.dispatch` | Dispatch full crew | {json: binary} |
//! | `crew.complete_task` | Mark task completed | {task_id, outcome} |
//! | `crew.status` | Get bridge status | {} |
//! | `a2a.send` | Send A2A message | {json: binary} |
//! | `a2a.receive` | Receive pending messages | {agent_slot: u8} |
//! | `sci.query` | Route sci/v1 query | {endpoint, params: binary} |
//! | `style.resolve` | Resolve thinking template | {name: string} |
//! | `style.list` | List all thinking templates | {} |
//! | `agent.blackboard` | Get agent blackboard state | {agent_slot: u8} |

use std::sync::Arc;
use parking_lot::RwLock;

use arrow_array::{
    ArrayRef, RecordBatch, StringArray, UInt8Array, UInt32Array, BooleanArray, Float32Array,
};
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use crate::storage::BindSpace;
use crate::orchestration::crew_bridge::CrewBridge;

// =============================================================================
// SCHEMAS
// =============================================================================

fn status_result_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("agents_registered", DataType::UInt32, false),
        Field::new("templates_registered", DataType::UInt32, false),
        Field::new("personas_registered", DataType::UInt32, false),
        Field::new("tasks_queued", DataType::UInt32, false),
        Field::new("tasks_in_progress", DataType::UInt32, false),
        Field::new("tasks_completed", DataType::UInt32, false),
        Field::new("a2a_channels", DataType::UInt32, false),
        Field::new("filters_registered", DataType::UInt32, false),
        Field::new("guardrail_topics", DataType::UInt32, false),
        Field::new("memories_stored", DataType::UInt32, false),
        Field::new("verification_rules", DataType::UInt32, false),
    ]))
}

fn agent_list_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("slot", DataType::UInt8, false),
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("role", DataType::Utf8, false),
        Field::new("thinking_style", DataType::Utf8, false),
        Field::new("allow_delegation", DataType::Boolean, false),
    ]))
}

fn template_list_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("slot", DataType::UInt8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("base_style", DataType::Utf8, false),
        Field::new("resonance_threshold", DataType::Float32, false),
        Field::new("fan_out", DataType::UInt32, false),
        Field::new("depth_bias", DataType::Float32, false),
        Field::new("exploration", DataType::Float32, false),
    ]))
}

fn blackboard_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("agent_slot", DataType::UInt8, false),
        Field::new("agent_id", DataType::Utf8, false),
        Field::new("active_style", DataType::Utf8, false),
        Field::new("coherence", DataType::Float32, false),
        Field::new("progress", DataType::Float32, false),
        Field::new("cycle", DataType::UInt32, false),
        Field::new("knowledge_count", DataType::UInt32, false),
        Field::new("ice_caked_count", DataType::UInt32, false),
        Field::new("pending_messages", DataType::UInt32, false),
    ]))
}

fn dispatch_result_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("task_id", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, false),
        Field::new("agent_slot", DataType::UInt8, true),
        Field::new("message", DataType::Utf8, false),
    ]))
}

fn error_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("error", DataType::Boolean, false),
        Field::new("message", DataType::Utf8, false),
    ]))
}

// =============================================================================
// IPC HELPERS
// =============================================================================

fn encode_to_ipc(batch: &RecordBatch) -> Result<Vec<u8>, String> {
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, batch.schema().as_ref())
            .map_err(|e| e.to_string())?;
        writer.write(batch).map_err(|e| e.to_string())?;
        writer.finish().map_err(|e| e.to_string())?;
    }
    Ok(buffer)
}

fn error_result(msg: &str) -> Result<Vec<u8>, String> {
    let schema = error_schema();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(BooleanArray::from(vec![true])) as ArrayRef,
            Arc::new(StringArray::from(vec![msg])) as ArrayRef,
        ],
    ).map_err(|e| e.to_string())?;
    encode_to_ipc(&batch)
}

fn ok_message(msg: &str) -> Result<Vec<u8>, String> {
    let schema = error_schema();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(BooleanArray::from(vec![false])) as ArrayRef,
            Arc::new(StringArray::from(vec![msg])) as ArrayRef,
        ],
    ).map_err(|e| e.to_string())?;
    encode_to_ipc(&batch)
}

// =============================================================================
// ACTION EXECUTION
// =============================================================================

/// Execute a crew/a2a/sci/style/agent action
pub fn execute_crew_action(
    action_type: &str,
    body: &[u8],
    bridge: Arc<RwLock<CrewBridge>>,
    bind_space: Arc<RwLock<BindSpace>>,
) -> Result<Vec<u8>, String> {
    match action_type {
        // === Crew Actions ===
        "crew.register_agent" => {
            let yaml = std::str::from_utf8(body)
                .map_err(|e| format!("Invalid UTF-8: {}", e))?;

            let mut bridge = bridge.write();
            match bridge.register_agents_yaml(yaml) {
                Ok(addrs) => {
                    let msg = format!("Registered {} agents at {:?}",
                        addrs.len(),
                        addrs.iter().map(|a| format!("{:#06x}", a.0)).collect::<Vec<_>>()
                    );
                    ok_message(&msg)
                }
                Err(e) => error_result(&e),
            }
        }

        "crew.register_style" => {
            let yaml = std::str::from_utf8(body)
                .map_err(|e| format!("Invalid UTF-8: {}", e))?;

            let mut bridge = bridge.write();
            match bridge.register_templates_yaml(yaml) {
                Ok(addrs) => {
                    let msg = format!("Registered {} templates at {:?}",
                        addrs.len(),
                        addrs.iter().map(|a| format!("{:#06x}", a.0)).collect::<Vec<_>>()
                    );
                    ok_message(&msg)
                }
                Err(e) => error_result(&e),
            }
        }

        "crew.submit_task" => {
            let task: crate::orchestration::crew_bridge::CrewTask =
                serde_json::from_slice(body)
                    .map_err(|e| format!("JSON parse error: {}", e))?;

            let mut bridge = bridge.write();
            let result = bridge.submit_task(task);
            let json = serde_json::to_vec(&result).map_err(|e| e.to_string())?;
            Ok(json)
        }

        "crew.dispatch" => {
            let dispatch: crate::orchestration::crew_bridge::CrewDispatch =
                serde_json::from_slice(body)
                    .map_err(|e| format!("JSON parse error: {}", e))?;

            let mut bridge = bridge.write();
            let results = bridge.dispatch_crew(dispatch);

            let schema = dispatch_result_schema();
            let task_ids: Vec<&str> = results.iter().map(|r| r.task_id.as_str()).collect();
            let statuses: Vec<String> = results.iter().map(|r| format!("{:?}", r.status)).collect();
            let status_refs: Vec<&str> = statuses.iter().map(|s| s.as_str()).collect();
            let slots: Vec<Option<u8>> = results.iter().map(|r| r.agent_slot).collect();
            let messages: Vec<&str> = results.iter().map(|r| r.message.as_str()).collect();

            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(task_ids)) as ArrayRef,
                    Arc::new(StringArray::from(status_refs)) as ArrayRef,
                    Arc::new(UInt8Array::from(slots)) as ArrayRef,
                    Arc::new(StringArray::from(messages)) as ArrayRef,
                ],
            ).map_err(|e| e.to_string())?;

            encode_to_ipc(&batch)
        }

        "crew.complete_task" => {
            #[derive(serde::Deserialize)]
            struct CompleteReq { task_id: String, outcome: String }
            let req: CompleteReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            let mut bridge = bridge.write();
            match bridge.complete_task(&req.task_id, &req.outcome) {
                Some(result) => {
                    let json = serde_json::to_vec(&result).map_err(|e| e.to_string())?;
                    Ok(json)
                }
                None => error_result(&format!("Task {} not found", req.task_id)),
            }
        }

        "crew.status" => {
            let bridge = bridge.read();
            let status = bridge.status_summary();

            let schema = status_result_schema();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt32Array::from(vec![status.agents_registered as u32])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![status.templates_registered as u32])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![status.personas_registered as u32])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![status.tasks_queued as u32])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![status.tasks_in_progress as u32])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![status.tasks_completed as u32])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![status.a2a_channels as u32])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![status.filters_registered as u32])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![status.guardrail_topics as u32])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![status.memories_stored as u32])) as ArrayRef,
                    Arc::new(UInt32Array::from(vec![status.verification_rules as u32])) as ArrayRef,
                ],
            ).map_err(|e| e.to_string())?;

            encode_to_ipc(&batch)
        }

        "crew.bind" => {
            let bridge = bridge.read();
            let mut space = bind_space.write();
            bridge.bind_all(&mut space);
            ok_message("All orchestration state bound to BindSpace")
        }

        // === A2A Actions ===
        "a2a.send" => {
            let msg: crate::orchestration::a2a::A2AMessage =
                serde_json::from_slice(body)
                    .map_err(|e| format!("JSON parse error: {}", e))?;

            let mut bridge = bridge.write();
            let mut space = bind_space.write();
            let status = bridge.send_a2a(msg, &mut space);
            ok_message(&format!("Delivery: {:?}", status))
        }

        "a2a.receive" => {
            #[derive(serde::Deserialize)]
            struct RecvReq { agent_slot: u8 }
            let req: RecvReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            let mut bridge = bridge.write();
            let messages = bridge.receive_a2a(req.agent_slot);
            let json = serde_json::to_vec(&messages).map_err(|e| e.to_string())?;
            Ok(json)
        }

        // === Style Actions ===
        "style.resolve" => {
            let name = std::str::from_utf8(body)
                .map_err(|e| format!("Invalid UTF-8: {}", e))?
                .trim();

            let bridge = bridge.read();
            match bridge.templates.get(name) {
                Some(template) => {
                    let m = template.effective_modulation();
                    let schema = template_list_schema();
                    let batch = RecordBatch::try_new(
                        schema,
                        vec![
                            Arc::new(UInt8Array::from(vec![template.slot.unwrap_or(0)])) as ArrayRef,
                            Arc::new(StringArray::from(vec![template.name.as_str()])) as ArrayRef,
                            Arc::new(StringArray::from(vec![template.base_style.as_str()])) as ArrayRef,
                            Arc::new(Float32Array::from(vec![m.resonance_threshold])) as ArrayRef,
                            Arc::new(UInt32Array::from(vec![m.fan_out as u32])) as ArrayRef,
                            Arc::new(Float32Array::from(vec![m.depth_bias])) as ArrayRef,
                            Arc::new(Float32Array::from(vec![m.exploration])) as ArrayRef,
                        ],
                    ).map_err(|e| e.to_string())?;
                    encode_to_ipc(&batch)
                }
                None => error_result(&format!("Template '{}' not found", name)),
            }
        }

        "style.list" => {
            let bridge = bridge.read();
            let templates = bridge.templates.list();

            let slots: Vec<u8> = templates.iter().map(|t| t.slot.unwrap_or(0)).collect();
            let names: Vec<&str> = templates.iter().map(|t| t.name.as_str()).collect();
            let bases: Vec<&str> = templates.iter().map(|t| t.base_style.as_str()).collect();
            let thresholds: Vec<f32> = templates.iter().map(|t| t.effective_modulation().resonance_threshold).collect();
            let fan_outs: Vec<u32> = templates.iter().map(|t| t.effective_modulation().fan_out as u32).collect();
            let depths: Vec<f32> = templates.iter().map(|t| t.effective_modulation().depth_bias).collect();
            let explorations: Vec<f32> = templates.iter().map(|t| t.effective_modulation().exploration).collect();

            let schema = template_list_schema();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt8Array::from(slots)) as ArrayRef,
                    Arc::new(StringArray::from(names)) as ArrayRef,
                    Arc::new(StringArray::from(bases)) as ArrayRef,
                    Arc::new(Float32Array::from(thresholds)) as ArrayRef,
                    Arc::new(UInt32Array::from(fan_outs)) as ArrayRef,
                    Arc::new(Float32Array::from(depths)) as ArrayRef,
                    Arc::new(Float32Array::from(explorations)) as ArrayRef,
                ],
            ).map_err(|e| e.to_string())?;

            encode_to_ipc(&batch)
        }

        // === Agent Blackboard ===
        "agent.blackboard" => {
            #[derive(serde::Deserialize)]
            struct BbReq { agent_slot: u8 }
            let req: BbReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            let bridge = bridge.read();
            match bridge.blackboards.get(req.agent_slot) {
                Some(bb) => {
                    let schema = blackboard_schema();
                    let batch = RecordBatch::try_new(
                        schema,
                        vec![
                            Arc::new(UInt8Array::from(vec![bb.agent_slot])) as ArrayRef,
                            Arc::new(StringArray::from(vec![bb.agent_id.as_str()])) as ArrayRef,
                            Arc::new(StringArray::from(vec![bb.awareness.active_style.as_str()])) as ArrayRef,
                            Arc::new(Float32Array::from(vec![bb.awareness.coherence])) as ArrayRef,
                            Arc::new(Float32Array::from(vec![bb.awareness.progress])) as ArrayRef,
                            Arc::new(UInt32Array::from(vec![bb.cycle as u32])) as ArrayRef,
                            Arc::new(UInt32Array::from(vec![bb.knowledge_addrs.len() as u32])) as ArrayRef,
                            Arc::new(UInt32Array::from(vec![bb.awareness.ice_caked.len() as u32])) as ArrayRef,
                            Arc::new(UInt32Array::from(vec![bb.awareness.pending_messages])) as ArrayRef,
                        ],
                    ).map_err(|e| e.to_string())?;
                    encode_to_ipc(&batch)
                }
                None => error_result(&format!("No blackboard for agent slot {}", req.agent_slot)),
            }
        }

        "agent.blackboard.yaml" => {
            #[derive(serde::Deserialize)]
            struct BbReq { agent_slot: u8 }
            let req: BbReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            let bridge = bridge.read();
            match bridge.blackboards.get(req.agent_slot) {
                Some(bb) => Ok(bb.to_yaml().into_bytes()),
                None => error_result(&format!("No blackboard for agent slot {}", req.agent_slot)),
            }
        }

        "agent.list" => {
            let bridge = bridge.read();
            let agents = bridge.agents.list();

            let slots: Vec<u8> = agents.iter().map(|a| a.slot.unwrap_or(0)).collect();
            let ids: Vec<&str> = agents.iter().map(|a| a.id.as_str()).collect();
            let names: Vec<&str> = agents.iter().map(|a| a.name.as_str()).collect();
            let roles: Vec<&str> = agents.iter().map(|a| a.role.name.as_str()).collect();
            let styles: Vec<&str> = agents.iter().map(|a| a.thinking_style.as_str()).collect();
            let delegations: Vec<bool> = agents.iter().map(|a| a.allow_delegation).collect();

            let schema = agent_list_schema();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt8Array::from(slots)) as ArrayRef,
                    Arc::new(StringArray::from(ids)) as ArrayRef,
                    Arc::new(StringArray::from(names)) as ArrayRef,
                    Arc::new(StringArray::from(roles)) as ArrayRef,
                    Arc::new(StringArray::from(styles)) as ArrayRef,
                    Arc::new(BooleanArray::from(delegations)) as ArrayRef,
                ],
            ).map_err(|e| e.to_string())?;

            encode_to_ipc(&batch)
        }

        // === Persona Actions ===
        "persona.attach" => {
            #[derive(serde::Deserialize)]
            struct AttachReq {
                agent_slot: u8,
                persona: crate::orchestration::persona::Persona,
            }
            let req: AttachReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            let mut bridge = bridge.write();
            bridge.personas.attach(req.agent_slot, req.persona);
            ok_message(&format!("Persona attached to agent slot {}", req.agent_slot))
        }

        "persona.attach_yaml" => {
            let yaml = std::str::from_utf8(body)
                .map_err(|e| format!("Invalid UTF-8: {}", e))?;

            let persona = crate::orchestration::persona::Persona::from_yaml(yaml)?;
            // Extract agent_slot from first line comment or separate field
            // For now, require JSON wrapper with agent_slot
            #[derive(serde::Deserialize)]
            struct YamlAttach { agent_slot: u8 }

            // Try parsing as YAML with agent_slot field
            #[derive(serde::Deserialize)]
            struct YamlReq {
                agent_slot: u8,
                #[serde(flatten)]
                persona: crate::orchestration::persona::Persona,
            }

            let req: YamlReq = serde_yml::from_str(yaml)
                .map_err(|e| format!("YAML parse error: {}", e))?;

            let mut bridge = bridge.write();
            bridge.personas.attach(req.agent_slot, req.persona);
            ok_message(&format!("Persona attached to agent slot {}", req.agent_slot))
        }

        "persona.get" => {
            #[derive(serde::Deserialize)]
            struct GetReq { agent_slot: u8 }
            let req: GetReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            let bridge = bridge.read();
            match bridge.personas.get(req.agent_slot) {
                Some(p) => Ok(p.to_json().into_bytes()),
                None => error_result(&format!("No persona for agent slot {}", req.agent_slot)),
            }
        }

        "persona.get_yaml" => {
            #[derive(serde::Deserialize)]
            struct GetReq { agent_slot: u8 }
            let req: GetReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            let bridge = bridge.read();
            match bridge.personas.get(req.agent_slot) {
                Some(p) => Ok(p.to_yaml().into_bytes()),
                None => error_result(&format!("No persona for agent slot {}", req.agent_slot)),
            }
        }

        "persona.compatible" => {
            #[derive(serde::Deserialize)]
            struct CompatReq { agent_slot: u8, threshold: f32 }
            let req: CompatReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            let bridge = bridge.read();
            match bridge.personas.get(req.agent_slot) {
                Some(persona) => {
                    let results = bridge.personas.find_compatible(persona, req.threshold);
                    let json = serde_json::to_vec(&results).map_err(|e| e.to_string())?;
                    Ok(json)
                }
                None => error_result(&format!("No persona for agent slot {}", req.agent_slot)),
            }
        }

        "persona.best_for_task" => {
            let description = std::str::from_utf8(body)
                .map_err(|e| format!("Invalid UTF-8: {}", e))?;

            let bridge = bridge.read();
            match bridge.personas.best_for_task(description) {
                Some((slot, score)) => {
                    let json = serde_json::json!({
                        "agent_slot": slot,
                        "alignment_score": score
                    });
                    Ok(serde_json::to_vec(&json).map_err(|e| e.to_string())?)
                }
                None => ok_message("No matching agent found for task"),
            }
        }

        // === Handover Actions ===
        "handover.evaluate" => {
            #[derive(serde::Deserialize)]
            struct EvalReq { agent_slot: u8, task_description: Option<String> }
            let req: EvalReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            let mut bridge = bridge.write();
            let decision = bridge.evaluate_handover(
                req.agent_slot,
                req.task_description.as_deref(),
            );
            let json = serde_json::to_vec(&decision).map_err(|e| e.to_string())?;
            Ok(json)
        }

        "handover.execute" => {
            let decision: crate::orchestration::handover::HandoverDecision =
                serde_json::from_slice(body)
                    .map_err(|e| format!("JSON parse error: {}", e))?;

            let mut bridge = bridge.write();
            let mut space = bind_space.write();
            let status = bridge.execute_handover(&decision, &mut space);
            let json = serde_json::json!({
                "executed": status.is_some(),
                "delivery_status": status.map(|s| format!("{:?}", s)),
            });
            Ok(serde_json::to_vec(&json).map_err(|e| e.to_string())?)
        }

        "handover.update_flow" => {
            #[derive(serde::Deserialize)]
            struct FlowReq { agent_slot: u8, gate_state: String }
            let req: FlowReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            let gate = match req.gate_state.to_lowercase().as_str() {
                "flow" => crate::cognitive::GateState::Flow,
                "hold" => crate::cognitive::GateState::Hold,
                "block" => crate::cognitive::GateState::Block,
                _ => return error_result(&format!("Unknown gate state: {}", req.gate_state)),
            };

            let mut bridge = bridge.write();
            bridge.update_flow(req.agent_slot, gate);
            ok_message(&format!("Flow updated for agent slot {}", req.agent_slot))
        }

        // === Orchestrator Actions ===
        "orchestrator.status" => {
            let bridge = bridge.read();
            let status = bridge.orchestrator.status();
            let json = serde_json::to_vec(&status).map_err(|e| e.to_string())?;
            Ok(json)
        }

        "orchestrator.route_task" => {
            let description = std::str::from_utf8(body)
                .map_err(|e| format!("Invalid UTF-8: {}", e))?;

            let mut bridge = bridge.write();
            match bridge.route_task(description) {
                Some((slot, score)) => {
                    let json = serde_json::json!({
                        "agent_slot": slot,
                        "resonance_score": score,
                    });
                    Ok(serde_json::to_vec(&json).map_err(|e| e.to_string())?)
                }
                None => ok_message("No suitable agent found for task"),
            }
        }

        "orchestrator.tick" => {
            let mut bridge = bridge.write();
            let decisions = bridge.tick_orchestrator();
            let json = serde_json::to_vec(&decisions).map_err(|e| e.to_string())?;
            Ok(json)
        }

        "orchestrator.affinity" => {
            #[derive(serde::Deserialize)]
            struct AffReq { agent_a: u8, agent_b: u8 }
            let req: AffReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            let bridge = bridge.read();
            let affinity = bridge.orchestrator.affinity(req.agent_a, req.agent_b);
            let json = serde_json::json!({
                "agent_a": req.agent_a,
                "agent_b": req.agent_b,
                "affinity": affinity,
            });
            Ok(serde_json::to_vec(&json).map_err(|e| e.to_string())?)
        }

        "orchestrator.awareness" => {
            let bridge = bridge.read();
            let awareness = bridge.total_awareness();
            let json = serde_json::json!({
                "total_awareness": awareness,
                "a2a_channels": bridge.a2a.channels().len(),
            });
            Ok(serde_json::to_vec(&json).map_err(|e| e.to_string())?)
        }

        // === Kernel Actions ===
        "kernel.describe" => {
            let bridge = bridge.read();
            let desc = bridge.kernel.describe();
            let json = serde_json::to_vec(&desc).map_err(|e| e.to_string())?;
            Ok(json)
        }

        "kernel.introspect" => {
            let bridge = bridge.read();
            let space = bind_space.read();
            let intro = bridge.kernel.introspect(&space);
            let json = serde_json::to_vec(&intro).map_err(|e| e.to_string())?;
            Ok(json)
        }

        "kernel.zone_density" => {
            #[derive(serde::Deserialize)]
            struct ZoneReq { prefix: u8 }
            let req: ZoneReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            let bridge = bridge.read();
            let space = bind_space.read();
            let zone = crate::orchestration::semantic_kernel::KernelZone::from_prefix(req.prefix);
            let density = bridge.kernel.zone_density(&space, &zone);
            let json = serde_json::json!({
                "prefix": req.prefix,
                "zone": format!("{:?}", zone),
                "populated_slots": density,
            });
            Ok(serde_json::to_vec(&json).map_err(|e| e.to_string())?)
        }

        "kernel.expansion" => {
            let bridge = bridge.read();
            let summary = bridge.kernel.expansion.summary();
            let json = serde_json::to_vec(&summary).map_err(|e| e.to_string())?;
            Ok(json)
        }

        "kernel.prefix_map" => {
            let bridge = bridge.read();
            let prefixes = bridge.kernel.expansion.all_prefixes();
            let json = serde_json::to_vec(&prefixes).map_err(|e| e.to_string())?;
            Ok(json)
        }

        // === Filter Pipeline Actions ===
        "filter.add" => {
            let filter: crate::orchestration::kernel_extensions::KernelFilter =
                serde_json::from_slice(body)
                    .map_err(|e| format!("JSON parse error: {}", e))?;
            let name = filter.name.clone();
            let mut bridge = bridge.write();
            bridge.filters.add(filter);
            ok_message(&format!("Filter '{}' added to pipeline", name))
        }

        "filter.remove" => {
            let name = std::str::from_utf8(body)
                .map_err(|e| format!("Invalid UTF-8: {}", e))?
                .trim();
            let mut bridge = bridge.write();
            bridge.filters.remove(name);
            ok_message(&format!("Filter '{}' removed", name))
        }

        "filter.list" => {
            let bridge = bridge.read();
            let filters = bridge.filters.filters();
            let json = serde_json::to_vec(filters).map_err(|e| e.to_string())?;
            Ok(json)
        }

        "filter.apply" => {
            let ctx: crate::orchestration::kernel_extensions::FilterContext =
                serde_json::from_slice(body)
                    .map_err(|e| format!("JSON parse error: {}", e))?;
            let bridge = bridge.read();
            let result = bridge.filters.apply(ctx);
            let json = serde_json::json!({
                "short_circuit": result.context.short_circuit,
                "filters_applied": result.filters_applied,
                "short_circuited_by": result.short_circuited_by,
                "metadata": result.context.metadata,
                "label": result.context.label,
            });
            Ok(serde_json::to_vec(&json).map_err(|e| e.to_string())?)
        }

        // === Guardrail Actions ===
        "guardrail.apply" => {
            #[derive(serde::Deserialize)]
            struct GuardrailReq {
                fingerprint: Vec<u64>,
                source_addrs: Option<Vec<u16>>,
            }
            let req: GuardrailReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            if req.fingerprint.len() != crate::storage::bind_space::FINGERPRINT_WORDS {
                return error_result(&format!(
                    "Fingerprint must have {} words, got {}",
                    crate::storage::bind_space::FINGERPRINT_WORDS,
                    req.fingerprint.len()
                ));
            }

            let mut fp = [0u64; crate::storage::bind_space::FINGERPRINT_WORDS];
            fp.copy_from_slice(&req.fingerprint);

            let space = bind_space.read();
            let sources: Option<Vec<crate::storage::bind_space::Addr>> = req.source_addrs.map(|addrs|
                addrs.iter().map(|&a| crate::storage::bind_space::Addr(a)).collect()
            );

            let bridge = bridge.read();
            let result = bridge.guardrail.apply(
                &fp,
                &space,
                sources.as_deref(),
            );
            let json = serde_json::to_vec(&result).map_err(|e| e.to_string())?;
            Ok(json)
        }

        "guardrail.add_topic" => {
            let topic: crate::orchestration::kernel_extensions::DeniedTopic =
                serde_json::from_slice(body)
                    .map_err(|e| format!("JSON parse error: {}", e))?;
            let name = topic.name.clone();
            let mut bridge = bridge.write();
            bridge.guardrail.add_denied_topic(topic);
            ok_message(&format!("Denied topic '{}' added", name))
        }

        "guardrail.enable_grounding" => {
            #[derive(serde::Deserialize)]
            struct GroundReq { threshold: f32 }
            let req: GroundReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;
            let mut bridge = bridge.write();
            bridge.guardrail.enable_grounding(req.threshold);
            ok_message(&format!("Grounding enabled with threshold {:.2}", req.threshold))
        }

        "guardrail.add_content_filter" => {
            #[derive(serde::Deserialize)]
            struct ContentReq {
                category: crate::orchestration::kernel_extensions::ContentCategory,
                max_severity: crate::orchestration::kernel_extensions::GuardrailSeverity,
            }
            let req: ContentReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;
            let mut bridge = bridge.write();
            bridge.guardrail.add_content_filter(req.category, req.max_severity);
            ok_message("Content filter added")
        }

        // === Workflow Actions ===
        "workflow.execute" => {
            let node: crate::orchestration::kernel_extensions::WorkflowNode =
                serde_json::from_slice(body)
                    .map_err(|e| format!("JSON parse error: {}", e))?;

            let bridge = bridge.read();
            let mut space = bind_space.write();
            let result = crate::orchestration::kernel_extensions::execute_workflow(
                &node, &mut space, &bridge.kernel,
            );
            let json = serde_json::to_vec(&result).map_err(|e| e.to_string())?;
            Ok(json)
        }

        // === Memory Bank Actions ===
        "memory.store" => {
            #[derive(serde::Deserialize)]
            struct StoreReq {
                kind: crate::orchestration::kernel_extensions::MemoryKind,
                content: String,
                fingerprint: Vec<u64>,
                cycle: u64,
                source_agent: Option<u8>,
            }
            let req: StoreReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            if req.fingerprint.len() != crate::storage::bind_space::FINGERPRINT_WORDS {
                return error_result(&format!(
                    "Fingerprint must have {} words, got {}",
                    crate::storage::bind_space::FINGERPRINT_WORDS,
                    req.fingerprint.len()
                ));
            }

            let mut fp = [0u64; crate::storage::bind_space::FINGERPRINT_WORDS];
            fp.copy_from_slice(&req.fingerprint);

            let mut bridge = bridge.write();
            let mut space = bind_space.write();
            match bridge.memory.store(req.kind, &req.content, fp, &mut space, req.cycle, req.source_agent) {
                Some(addr) => ok_message(&format!("Memory stored at {:04X}", addr.0)),
                None => error_result("Memory prefix full (255 slots)"),
            }
        }

        "memory.retrieve" => {
            #[derive(serde::Deserialize)]
            struct RetrieveReq {
                query: Vec<u64>,
                kind: Option<crate::orchestration::kernel_extensions::MemoryKind>,
                threshold: f32,
                limit: usize,
                cycle: u64,
            }
            let req: RetrieveReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            if req.query.len() != crate::storage::bind_space::FINGERPRINT_WORDS {
                return error_result(&format!(
                    "Query fingerprint must have {} words, got {}",
                    crate::storage::bind_space::FINGERPRINT_WORDS,
                    req.query.len()
                ));
            }

            let mut query = [0u64; crate::storage::bind_space::FINGERPRINT_WORDS];
            query.copy_from_slice(&req.query);

            let mut bridge = bridge.write();
            let space = bind_space.read();
            let results = bridge.memory.retrieve(&query, req.kind, &space, req.threshold, req.limit, req.cycle);
            let json = serde_json::to_vec(&results).map_err(|e| e.to_string())?;
            Ok(json)
        }

        "memory.extract_semantic" => {
            #[derive(serde::Deserialize)]
            struct ExtractReq { cycle: u64 }
            let req: ExtractReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            let mut bridge = bridge.write();
            let mut space = bind_space.write();
            let addrs = bridge.memory.extract_semantic(&mut space, req.cycle);
            let json = serde_json::json!({
                "extracted": addrs.len(),
                "addrs": addrs.iter().map(|a| format!("{:04X}", a.0)).collect::<Vec<_>>(),
            });
            Ok(serde_json::to_vec(&json).map_err(|e| e.to_string())?)
        }

        "memory.list" => {
            #[derive(serde::Deserialize)]
            struct ListReq { kind: Option<crate::orchestration::kernel_extensions::MemoryKind> }
            let req: ListReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            let bridge = bridge.read();
            let memories = bridge.memory.list(req.kind);
            let json = serde_json::to_vec(&memories).map_err(|e| e.to_string())?;
            Ok(json)
        }

        // === Observability Actions ===
        "observability.start_session" => {
            #[derive(serde::Deserialize)]
            struct SessionReq { agent_slot: Option<u8>, cycle: u64 }
            let req: SessionReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            let mut bridge = bridge.write();
            let id = bridge.observability.start_session(req.agent_slot, req.cycle);
            ok_message(&format!("Session started: {}", id))
        }

        "observability.start_trace" => {
            #[derive(serde::Deserialize)]
            struct TraceReq { operation: String, cycle: u64 }
            let req: TraceReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            let mut bridge = bridge.write();
            let id = bridge.observability.start_trace(&req.operation, req.cycle);
            ok_message(&format!("Trace started: {}", id))
        }

        "observability.add_span" => {
            #[derive(serde::Deserialize)]
            struct SpanReq {
                trace_id: String,
                span: crate::orchestration::kernel_extensions::KernelSpan,
            }
            let req: SpanReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            let mut bridge = bridge.write();
            bridge.observability.add_span(&req.trace_id, req.span);
            ok_message("Span added to trace")
        }

        "observability.complete_trace" => {
            #[derive(serde::Deserialize)]
            struct CompleteReq {
                trace_id: String,
                cycle: u64,
                grounding: Option<crate::orchestration::kernel_extensions::GroundingMetadata>,
            }
            let req: CompleteReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            let mut bridge = bridge.write();
            bridge.observability.complete_trace(&req.trace_id, req.cycle, req.grounding);
            ok_message(&format!("Trace {} completed", req.trace_id))
        }

        "observability.summary" => {
            let bridge = bridge.read();
            let summary = bridge.observability.summary();
            let json = serde_json::to_vec(&summary).map_err(|e| e.to_string())?;
            Ok(json)
        }

        // === Verification Actions ===
        "verification.add_rule" => {
            let rule: crate::orchestration::kernel_extensions::VerificationRule =
                serde_json::from_slice(body)
                    .map_err(|e| format!("JSON parse error: {}", e))?;
            let name = rule.name.clone();
            let mut bridge = bridge.write();
            bridge.verification.add_rule(rule);
            ok_message(&format!("Verification rule '{}' added", name))
        }

        "verification.verify" => {
            #[derive(serde::Deserialize)]
            struct VerifyReq {
                fingerprint: Vec<u64>,
                addr: u16,
            }
            let req: VerifyReq = serde_json::from_slice(body)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            if req.fingerprint.len() != crate::storage::bind_space::FINGERPRINT_WORDS {
                return error_result(&format!(
                    "Fingerprint must have {} words, got {}",
                    crate::storage::bind_space::FINGERPRINT_WORDS,
                    req.fingerprint.len()
                ));
            }

            let mut fp = [0u64; crate::storage::bind_space::FINGERPRINT_WORDS];
            fp.copy_from_slice(&req.fingerprint);

            let bridge = bridge.read();
            let results = bridge.verification.verify_fingerprint(
                &fp,
                crate::storage::bind_space::Addr(req.addr),
            );
            let json = serde_json::to_vec(&results).map_err(|e| e.to_string())?;
            Ok(json)
        }

        "verification.list_rules" => {
            let bridge = bridge.read();
            let rules = bridge.verification.rules();
            let json = serde_json::to_vec(rules).map_err(|e| e.to_string())?;
            Ok(json)
        }

        _ => error_result(&format!("Unknown crew action: {}", action_type)),
    }
}
