//! World state and counterfactual reasoning

mod state;
mod counterfactual;

pub use state::World;
pub use counterfactual::{Counterfactual, Change};
