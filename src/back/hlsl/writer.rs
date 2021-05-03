use super::{Error, Options};
use crate::back::hlsl::keywords::RESERVED;
use crate::proc::{NameKey, Namer};
use crate::{FastHashMap, ShaderStage};
use std::fmt::Write;

const INDENT: &str = "    ";

pub struct Writer<'a, W> {
    out: W,
    module: &'a crate::Module,
    options: &'a Options,
    names: FastHashMap<NameKey, String>,
    namer: Namer,
}

impl<'a, W: Write> Writer<'a, W> {
    pub fn new(out: W, module: &'a crate::Module, options: &'a Options) -> Self {
        Writer {
            out,
            module,
            options,
            names: FastHashMap::default(),
            namer: Namer::default(),
        }
    }

    pub fn write(&mut self) -> Result<(), Error> {
        self.names.clear();
        self.namer.reset(self.module, RESERVED, &mut self.names);

        for (ep_index, ep) in self.module.entry_points.iter().enumerate() {
            let fun = &ep.function;
            let fun_name = &self.names[&NameKey::EntryPoint(ep_index as _)];
            writeln!(self.out)?;

            let return_type_name = match fun.result {
                None => "void",
                _ => "",
            };

            if ep.stage == ShaderStage::Compute {
                // HLSL is calling workgroup size, num threads
                let num_threads = ep.workgroup_size;
                writeln!(
                    self.out,
                    "[numthreads({}, {}, {})]",
                    num_threads[0], num_threads[1], num_threads[2]
                )?;
            }

            writeln!(
                self.out,
                "{} {}({}",
                return_type_name,
                fun_name,
                if fun.arguments.is_empty() { ")" } else { "" }
            )?;

            // TODO Support arguments
            self.write_block(&ep.function.body)?;
        }
        Ok(())
    }

    fn write_block(&mut self, statements: &[crate::Statement]) -> Result<(), Error> {
        writeln!(self.out, "{{")?;

        for statement in statements {
            if let crate::Statement::Return { value: None } = *statement {
                writeln!(self.out, "{}return;", INDENT)?;
            }
        }

        writeln!(self.out, "}}")?;

        Ok(())
    }

    pub fn finish(self) -> W {
        self.out
    }
}
