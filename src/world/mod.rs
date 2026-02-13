//! World state and counterfactual reasoning

mod state;
pub mod counterfactual;

pub use state::World;
pub use counterfactual::{
    Counterfactual, Change,
    CounterfactualWorld, Intervention,
    intervene, worlds_differ, multi_intervene,
};
