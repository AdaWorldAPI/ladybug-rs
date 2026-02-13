//! CAM Query Example
//!
//! Demonstrates Content-Addressable Memory operations using 8+8 addressing.

use ladybug::storage::bind_space::{FINGERPRINT_WORDS, PREFIX_LANCE, PREFIX_SQL, PREFIX_VERBS};
use ladybug::storage::{Addr, BindSpace, CogRedis};

fn main() {
    println!("CAM Query: 8-bit prefix : 8-bit slot addressing\n");

    // Create bind space
    let bind_space = BindSpace::new();

    // Show address structure
    println!("Address space layout:");
    println!("  Surface: 0x00-0x0F:XX (4,096 CAM ops)");
    println!("  Fluid:   0x10-0x7F:XX (28,672 working memory)");
    println!("  Nodes:   0x80-0xFF:XX (32,768 persistent)\n");

    // Access surface operations
    let lance_vector_search = Addr::new(PREFIX_LANCE, 0x00);
    let sql_select = Addr::new(PREFIX_SQL, 0x00);
    let verb_causes = Addr::new(PREFIX_VERBS, 0x00);

    println!("Surface operation addresses:");
    println!("  LANCE:VECTOR_SEARCH = 0x{:04X}", lance_vector_search.0);
    println!("  SQL:SELECT          = 0x{:04X}", sql_select.0);
    println!("  VERBS:CAUSES        = 0x{:04X}", verb_causes.0);

    // Read from bind space
    if let Some(node) = bind_space.read(lance_vector_search)
        && let Some(label) = &node.label
    {
        println!("\n  LANCE:0x00 = '{}'", label);
    }

    if let Some(node) = bind_space.read(sql_select)
        && let Some(label) = &node.label
    {
        println!("  SQL:0x00   = '{}'", label);
    }

    if let Some(node) = bind_space.read(verb_causes)
        && let Some(label) = &node.label
    {
        println!("  VERBS:0x00 = '{}'", label);
    }

    // Stats
    let stats = bind_space.stats();
    println!("\nBind space stats:");
    println!("  Surface count: {}", stats.surface_count);
    println!("  Node count: {}", stats.node_count);
    println!("  Edge count: {}", stats.edge_count);

    // Direct array indexing: 3-5 cycles
    println!("\n✓ O(1) array indexing - no HashMap lookup!");
}
