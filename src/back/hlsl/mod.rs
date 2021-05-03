mod keywords;
mod writer;

use std::fmt::Error as FmtError;
use thiserror::Error;

pub use writer::Writer;

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    IoError(#[from] FmtError),
}

#[derive(Debug, Clone)]
pub struct Options {
    /// The stage of the entry point
    pub shader_stage: crate::ShaderStage,
    /// The name of the entry point
    pub entry_point: String,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            shader_stage: crate::ShaderStage::Compute,
            entry_point: String::from("main"),
        }
    }
}

pub fn write_string(module: &crate::Module, options: &Options) -> Result<String, Error> {
    let mut w = Writer::new(String::new(), module, options);
    w.write()?;
    let output = w.finish();
    Ok(output)
}
