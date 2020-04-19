use spirv::*;

use crate::back::spv::{Instruction, LogicalLayout, PhysicalLayout};
use std::iter;

impl PhysicalLayout {
    pub(crate) fn new(header: &crate::Header) -> Self {
        let version: Word = (0x0u32 << 24)
            | ((header.version.0 as u32) << 16)
            | ((header.version.1 as u32) << 8)
            | header.version.2 as u32;

        PhysicalLayout {
            magic_number: MAGIC_NUMBER,
            version,
            generator: header.generator,
            bound: 0,
            instruction_schema: 0x0u32,
        }
    }

    pub(crate) fn in_words(&self, sink: &mut impl Extend<Word>) {
        sink.extend(iter::once(self.magic_number));
        sink.extend(iter::once(self.version));
        sink.extend(iter::once(self.generator));
        sink.extend(iter::once(self.bound));
        sink.extend(iter::once(self.instruction_schema));
    }
}

impl LogicalLayout {
    pub(crate) fn new() -> Self {
        LogicalLayout {
            capabilities: vec![],
            extensions: vec![],
            ext_inst_imports: vec![],
            memory_model: vec![],
            entry_points: vec![],
            execution_modes: vec![],
            debugs: vec![],
            annotations: vec![],
            type_declarations: vec![],
            constants: vec![],
            global_variables: vec![],
            function_declarations: vec![],
            function_definitions: vec![],
        }
    }

    pub(crate) fn in_words(&self, sink: &mut impl Extend<Word>) {
        sink.extend(self.capabilities.iter().cloned());
        sink.extend(self.extensions.iter().cloned());
        sink.extend(self.ext_inst_imports.iter().cloned());
        sink.extend(self.memory_model.iter().cloned());
        sink.extend(self.entry_points.iter().cloned());
        sink.extend(self.execution_modes.iter().cloned());
        sink.extend(self.debugs.iter().cloned());
        sink.extend(self.annotations.iter().cloned());
        sink.extend(self.type_declarations.iter().cloned());
        sink.extend(self.constants.iter().cloned());
        sink.extend(self.global_variables.iter().cloned());
        sink.extend(self.function_declarations.iter().cloned());
        sink.extend(self.function_definitions.iter().cloned());
    }
}

impl Instruction {
    pub(crate) fn new(op: Op) -> Self {
        Instruction {
            op,
            wc: 1, // Always start at 1 for the first word (OP + WC),
            type_id: None,
            result_id: None,
            operands: vec![],
        }
    }

    pub(crate) fn set_type(&mut self, id: Word) {
        assert!(self.type_id.is_none(), "Type can only be set once");
        self.type_id = Some(id);
        self.wc += 1;
    }

    pub(crate) fn set_result(&mut self, id: Word) {
        assert!(self.result_id.is_none(), "Result can only be set once");
        self.result_id = Some(id);
        self.wc += 1;
    }

    pub(crate) fn add_operand(&mut self, operand: Word) {
        self.operands.push(operand);
        self.wc += 1;
    }

    pub(crate) fn add_operands(&mut self, operands: Vec<Word>) {
        for operand in operands.into_iter() {
            self.add_operand(operand)
        }
    }

    pub(crate) fn to_words(&self, sink: &mut impl Extend<Word>) {
        sink.extend(Some((self.wc << 16 | self.op as u32) as u32));
        sink.extend(self.type_id);
        sink.extend(self.result_id);
        sink.extend(self.operands.iter().cloned());
    }
}
