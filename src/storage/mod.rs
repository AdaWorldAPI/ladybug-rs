//! Storage layer - LanceDB integration

mod database;

pub use database::Database;

#[derive(thiserror::Error, Debug)]
pub enum StorageError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Not found: {0}")]
    NotFound(String),
}
