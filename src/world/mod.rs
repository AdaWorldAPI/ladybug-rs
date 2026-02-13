//! World state and counterfactual reasoning

pub mod counterfactual;
mod state;

pub use counterfactual::{
    Change, Counterfactual, CounterfactualWorld, Intervention, intervene, multi_intervene,
    worlds_differ,
};
pub use state::World;
