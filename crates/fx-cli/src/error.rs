//! Error type with the process exit codes: 2 usage, 3 effect or param, 4 io.

use std::fmt;

#[derive(Debug, PartialEq)]
pub enum CliError {
    Usage(String),
    Effect(String),
    Io(String),
}

impl CliError {
    pub fn code(&self) -> i32 {
        match self {
            CliError::Usage(_) => 2,
            CliError::Effect(_) => 3,
            CliError::Io(_) => 4,
        }
    }

    pub fn usage(msg: impl Into<String>) -> CliError {
        CliError::Usage(msg.into())
    }

    pub fn io(context: impl fmt::Display, err: impl fmt::Display) -> CliError {
        CliError::Io(format!("{context}: {err}"))
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            CliError::Usage(m) | CliError::Effect(m) | CliError::Io(m) => m,
        };
        // errors are always a single line on stderr
        write!(f, "{}", msg.split_whitespace().collect::<Vec<_>>().join(" "))
    }
}

impl From<fx_core::FxError> for CliError {
    fn from(e: fx_core::FxError) -> CliError {
        match e {
            fx_core::FxError::Io(_) => CliError::Io(e.to_string()),
            _ => CliError::Effect(e.to_string()),
        }
    }
}
