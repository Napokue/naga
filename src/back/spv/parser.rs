/*! Standard Portable Intermediate Representation (SPIR-V) backend !*/
use crate::arena::Handle;
use crate::back::spv::{Instruction, LogicalLayout, ParserFlags, PhysicalLayout};
use crate::{FastHashMap, FastHashSet, ScalarKind};
use spirv::*;

#[derive(Debug, PartialEq)]
struct LookupFunctionType {
    parameter_type_ids: Vec<Word>,
    return_type_id: Word,
}

#[derive(PartialEq)]
enum LookupType<T> {
    Handle(Handle<T>),
    Standalone(T),
}

pub struct Parser {
    physical_layout: PhysicalLayout,
    logical_layout: LogicalLayout,
    id_count: u32,
    capabilities: FastHashSet<Capability>,
    debugs: Vec<Instruction>,
    annotations: Vec<Instruction>,
    parser_flags: ParserFlags,
    void_type: Option<Word>,
    bool_type: Option<Word>,
    lookup_type: FastHashMap<Word, LookupType<crate::Type>>,
    lookup_function: FastHashMap<Word, LookupType<crate::Function>>,
    lookup_local_variable: FastHashMap<String, Word>,
    lookup_function_type: FastHashMap<Word, LookupFunctionType>,
    lookup_constant: FastHashMap<Word, LookupType<crate::Constant>>,
    lookup_global_variable: FastHashMap<Word, LookupType<crate::GlobalVariable>>,
    lookup_import: FastHashMap<Word, String>,
    labels: Vec<Word>
}

impl Parser {
    pub fn new(header: &crate::Header, parser_flags: ParserFlags) -> Self {
        Parser {
            physical_layout: PhysicalLayout::new(header),
            logical_layout: LogicalLayout::new(),
            id_count: 0,
            capabilities: FastHashSet::default(),
            debugs: vec![],
            annotations: vec![],
            parser_flags,
            void_type: None,
            bool_type: None,
            lookup_type: FastHashMap::default(),
            lookup_function: FastHashMap::default(),
            lookup_local_variable: FastHashMap::default(),
            lookup_function_type: FastHashMap::default(),
            lookup_constant: FastHashMap::default(),
            lookup_global_variable: FastHashMap::default(),
            lookup_import: FastHashMap::default(),
            labels: vec![]
        }
    }

    fn generate_id(&mut self) -> Word {
        self.id_count += 1;
        self.id_count
    }

    fn bytes_to_words(&self, bytes: &[u8]) -> Vec<Word> {
        bytes
            .chunks(4)
            .map(|chars| chars.iter().rev().fold(0u32, |u, c| (u << 8) | *c as u32))
            .collect()
    }

    fn string_to_words(&self, input: &str) -> Vec<Word> {
        let bytes = input.as_bytes();
        let mut words = self.bytes_to_words(bytes);

        if bytes.len() % 4 == 0 {
            // nul-termination
            words.push(0x0u32);
        }

        words
    }

    fn get_id<T>(
        &self,
        lookup_table: &FastHashMap<Word, LookupType<T>>,
        lookup: LookupType<T>,
    ) -> Option<Word> {
        match lookup {
            LookupType::Handle(handle) => {
                let result = lookup_table
                    .iter()
                    .find(|(_, v)| match v {
                        LookupType::Handle(lookup_handle) => {
                            handle.index() == lookup_handle.index()
                        }
                        _ => false,
                    })
                    .map(|(k, _)| *k);
                result
            }
            LookupType::Standalone(_ty) => {
                // TODO how to check without implementing PartialEq everywhere
                unimplemented!()
            }
        }
    }

    fn instruction_capability(&self, capability: &Capability) -> Instruction {
        let mut instruction = Instruction::new(Op::Capability);
        instruction.add_operand(*capability as u32);
        instruction
    }

    fn try_add_capabilities(&mut self, capabilities: &[Capability]) {
        for capability in capabilities.iter() {
            self.capabilities.insert(*capability);
        }
    }

    fn instruction_ext_inst_import(&mut self) -> Instruction {
        let mut instruction = Instruction::new(Op::ExtInstImport);
        let id = self.generate_id();
        instruction.set_result(id);

        // TODO Support other imports
        instruction.add_operands(self.string_to_words("GLSL.std.450"));
        self.lookup_import.insert(id, String::from("GLSL.std.450"));

        instruction
    }

    fn instruction_memory_model(&mut self) -> Instruction {
        let mut instruction = Instruction::new(Op::MemoryModel);
        let addressing_model = AddressingModel::Logical;
        let memory_model = MemoryModel::GLSL450;
        self.try_add_capabilities(addressing_model.required_capabilities());
        self.try_add_capabilities(memory_model.required_capabilities());

        instruction.add_operand(addressing_model as u32);
        instruction.add_operand(memory_model as u32);
        instruction
    }

    fn instruction_entry_point(
        &mut self,
        entry_point: &crate::EntryPoint,
        ir_module: &crate::Module,
    ) -> Instruction {
        let mut instruction = Instruction::new(Op::EntryPoint);

        let function_id = self
            .get_id(
                &self.lookup_function,
                LookupType::Handle(*&entry_point.function),
            )
            .unwrap();

        instruction.add_operand(entry_point.exec_model as u32);
        instruction.add_operand(function_id);

        if self.parser_flags.contains(ParserFlags::DEBUG) {
            let mut debug_instruction = Instruction::new(Op::Name);
            debug_instruction.set_result(function_id);
            debug_instruction.add_operands(self.string_to_words((&ir_module.functions[*&entry_point.function]).name.as_ref().unwrap().as_str()));
            self.debugs.push(debug_instruction);
        }

        instruction.add_operands(self.string_to_words(entry_point.name.as_str()));

        let function = &ir_module.functions[entry_point.function];
        for ((handle, _), &usage) in ir_module
            .global_variables
            .iter()
            .zip(&function.global_usage)
        {
            if usage.contains(crate::GlobalUse::STORE) || usage.contains(crate::GlobalUse::LOAD) {
                let id = self.get_global_variable_id(
                    &ir_module.types,
                    &ir_module.global_variables,
                    handle,
                );
                instruction.add_operand(id);
            }
        }

        self.try_add_capabilities(entry_point.exec_model.required_capabilities());
        match entry_point.exec_model {
            ExecutionModel::Vertex | ExecutionModel::GLCompute => {}
            ExecutionModel::Fragment => {
                let execution_mode = ExecutionMode::OriginUpperLeft;
                self.try_add_capabilities(execution_mode.required_capabilities());
                let mut execution_mode_instruction = Instruction::new(Op::ExecutionMode);
                execution_mode_instruction.add_operand(function_id);
                execution_mode_instruction.add_operand(execution_mode as u32);
                execution_mode_instruction.to_words(&mut self.logical_layout.execution_modes);
            }
            _ => unimplemented!("{:?}", entry_point.exec_model),
        }

        instruction
    }

    fn get_type_id(
        &mut self,
        arena: &crate::Arena<crate::Type>,
        handle: crate::Handle<crate::Type>,
    ) -> Word {
        match self.get_id(&self.lookup_type, LookupType::Handle(handle)) {
            Some(word) => word,
            None => {
                let (instruction, id) = self.instruction_type_declaration(arena, handle);
                self.lookup_type.insert(id, LookupType::Handle(handle));
                instruction.to_words(&mut self.logical_layout.type_declarations);
                id
            }
        }
    }

    fn get_constant_id(
        &mut self,
        handle: crate::Handle<crate::Constant>,
        ir_module: &crate::Module,
    ) -> Word {
        match self.get_id(&self.lookup_constant, LookupType::Handle(handle)) {
            Some(word) => word,
            None => {
                let (instruction, id) = self.instruction_constant_type(&LookupType::Handle(handle), ir_module);
                instruction.to_words(&mut self.logical_layout.constants);
                self.lookup_constant.insert(id, LookupType::Handle(handle));

                id
            }
        }
    }

    fn get_global_variable_id(
        &mut self,
        arena: &crate::Arena<crate::Type>,
        global_arena: &crate::Arena<crate::GlobalVariable>,
        handle: crate::Handle<crate::GlobalVariable>,
    ) -> Word {
        match self.get_id(&self.lookup_global_variable, LookupType::Handle(handle)) {
            Some(word) => word,
            None => {
                let global_variable = &global_arena[handle];
                let (instruction, id) =
                    self.instruction_global_variable(arena, global_variable, handle);
                instruction.to_words(&mut self.logical_layout.global_variables);
                id
            }
        }
    }

    fn get_function_type(
        &mut self,
        ty: Option<crate::Handle<crate::Type>>,
        arena: &crate::Arena<crate::Type>,
    ) -> Word {
        match ty {
            Some(handle) => self.get_type_id(arena, handle),
            None => match self.void_type {
                Some(id) => id,
                None => {
                    let id = self.generate_id();

                    let mut instruction = Instruction::new(Op::TypeVoid);
                    instruction.set_result(id);

                    self.void_type = Some(id);
                    instruction.to_words(&mut self.logical_layout.type_declarations);
                    id
                }
            },
        }
    }

    fn find_scalar_handle(
        &self,
        arena: &crate::Arena<crate::Type>,
        kind: &crate::ScalarKind,
        width: &u8,
    ) -> crate::Handle<crate::Type> {
        let mut scalar_handle = None;
        for (handle, ty) in arena.iter() {
            match ty.inner {
                crate::TypeInner::Scalar {
                    kind: _kind,
                    width: _width,
                } => {
                    if kind == &_kind && width == &_width {
                        scalar_handle = Some(handle);
                        break;
                    }
                }
                _ => continue,
            }
        }
        scalar_handle.unwrap()
    }

    fn instruction_type_declaration(
        &mut self,
        arena: &crate::Arena<crate::Type>,
        handle: crate::Handle<crate::Type>,
    ) -> (Instruction, Word) {
        let ty = &arena[handle];
        let id = self.generate_id();
        let mut instruction;

        match ty.inner {
            crate::TypeInner::Scalar { kind, width } => {
                match kind {
                    crate::ScalarKind::Sint => {
                        instruction = Instruction::new(Op::TypeInt);
                        instruction.set_result(id);
                        instruction.add_operand(width as u32);
                        instruction.add_operand(0x1u32);
                    }
                    crate::ScalarKind::Uint => {
                        instruction = Instruction::new(Op::TypeInt);
                        instruction.set_result(id);
                        instruction.add_operand(width as u32);
                        instruction.add_operand(0x0u32);
                    }
                    crate::ScalarKind::Float => {
                        instruction = Instruction::new(Op::TypeFloat);
                        instruction.set_result(id);
                        instruction.add_operand(width as u32);
                    }
                    crate::ScalarKind::Bool => {
                        instruction = Instruction::new(Op::TypeBool);
                        instruction.set_result(id);
                    }
                }
            }
            crate::TypeInner::Vector { size, kind, width } => {
                let scalar_handle = self.find_scalar_handle(arena, &kind, &width);
                let scalar_id = self.get_type_id(arena, scalar_handle);

                instruction = Instruction::new(Op::TypeVector);
                instruction.set_result(id);
                instruction.add_operand(scalar_id);
                instruction.add_operand(size as u32);
            }
            crate::TypeInner::Matrix {
                columns,
                rows: _,
                kind,
                width,
            } => {
                let scalar_handle = self.find_scalar_handle(arena, &kind, &width);
                let scalar_id = self.get_type_id(arena, scalar_handle);

                instruction = Instruction::new(Op::TypeMatrix);
                instruction.set_result(id);
                instruction.add_operand(scalar_id);
                instruction.add_operand(columns as u32);
            }
            crate::TypeInner::Pointer { base, class } => {
                let type_id = self.get_type_id(arena, base);
                instruction = Instruction::new(Op::TypePointer);
                instruction.set_result(id);
                instruction.add_operand(class as u32);
                instruction.add_operand(type_id);
            }
            crate::TypeInner::Array { base, size } => {
                let type_id = self.get_type_id(arena, base);

                instruction = Instruction::new(Op::TypeArray);
                instruction.set_result(id);
                instruction.add_operand(type_id);

                match size {
                    crate::ArraySize::Static(word) => {
                        instruction.add_operand(word);
                    }
                    _ => panic!("Array size {:?} unsupported", size),
                }
            }
            crate::TypeInner::Struct { ref members } => {
                instruction = Instruction::new(Op::TypeStruct);
                instruction.set_result(id);

                for member in members {
                    let type_id = self.get_type_id(arena, member.ty);
                    instruction.add_operand(type_id);
                }
            }
            crate::TypeInner::Image { base, dim, flags } => {
                let type_id = self.get_type_id(arena, base);
                self.try_add_capabilities(dim.required_capabilities());

                instruction = Instruction::new(Op::TypeImage);
                instruction.set_result(id);
                instruction.add_operand(type_id);
                instruction.add_operand(dim as u32);

                // TODO Add Depth, but how to determine? Not yet in the WGSL spec
                instruction.add_operand(1);

                instruction.add_operand(if flags.contains(crate::ImageFlags::ARRAYED) {
                    1
                } else {
                    0
                });

                instruction.add_operand(if flags.contains(crate::ImageFlags::MULTISAMPLED) {
                    1
                } else {
                    0
                });

                if let Dim::DimSubpassData = dim {
                    instruction.add_operand(2);
                    instruction.add_operand(ImageFormat::Unknown as u32);
                } else {
                    instruction.add_operand(if flags.contains(crate::ImageFlags::SAMPLED) {
                        1
                    } else {
                        0
                    });

                    // TODO Add Image Format, but how to determine? Not yet in the WGSL spec
                    instruction.add_operand(ImageFormat::Unknown as u32);
                };

                instruction.add_operand(
                    if flags.contains(crate::ImageFlags::CAN_STORE)
                        && flags.contains(crate::ImageFlags::CAN_LOAD)
                    {
                        2
                    } else if flags.contains(crate::ImageFlags::CAN_STORE) {
                        1
                    } else {
                        0
                    },
                );
            }
            crate::TypeInner::Sampler => {
                instruction = Instruction::new(Op::TypeSampler);
                instruction.set_result(id);
            }
        }

        (instruction, id)
    }

    fn instruction_constant_type(
        &mut self,
        lookup_type: &LookupType<crate::Constant>,
        ir_module: &crate::Module,
    ) -> (Instruction, Word) {
        let id = self.generate_id();

        let constant = match lookup_type {
            LookupType::Handle(handle) => &ir_module.constants[*handle],
            LookupType::Standalone(constant) => constant,
        };

        let arena = &ir_module.types;

        match constant.inner {
            crate::ConstantInner::Sint(val) => {
                let type_id = self.get_type_id(arena, constant.ty);

                let mut instruction = Instruction::new(Op::Constant);
                instruction.set_type(type_id);
                instruction.set_result(id);

                let ty = &ir_module.types[constant.ty];
                match ty.inner {
                    crate::TypeInner::Scalar { kind: _, width } => match width {
                        32 => {
                            instruction.add_operand(val as u32);
                        }
                        64 => {
                            let (low, high) = ((val >> 32) as u32, val as u32);
                            instruction.add_operand(low);
                            instruction.add_operand(high);
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                }

                (instruction, id)
            }
            crate::ConstantInner::Uint(val) => {
                let type_id = self.get_type_id(arena, constant.ty);

                let mut instruction = Instruction::new(Op::Constant);
                instruction.set_type(type_id);
                instruction.set_result(id);

                let ty = &ir_module.types[constant.ty];
                match ty.inner {
                    crate::TypeInner::Scalar { kind: _, width } => match width {
                        32 => {
                            instruction.add_operand(val as u32);
                        }
                        64 => {
                            let (low, high) = ((val >> 32) as u32, val as u32);
                            instruction.add_operand(low);
                            instruction.add_operand(high);
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                }

                (instruction, id)
            }
            crate::ConstantInner::Float(val) => {
                let type_id = self.get_type_id(arena, constant.ty);

                let mut instruction = Instruction::new(Op::Constant);
                instruction.set_type(type_id);
                instruction.set_result(id);

                let ty = &ir_module.types[constant.ty];
                match ty.inner {
                    crate::TypeInner::Scalar { kind: _, width } => match width {
                        32 => {
                            instruction.add_operand((val as f32).to_bits());
                        }
                        64 => {
                            let bits = f64::to_bits(val);
                            let (low, high) = ((bits >> 32) as u32, bits as u32);
                            instruction.add_operand(low);
                            instruction.add_operand(high);
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                }

                (instruction, id)
            }
            crate::ConstantInner::Bool(val) => {
                let type_id = self.get_type_id(arena, constant.ty);

                let mut instruction = Instruction::new(if val {
                    Op::ConstantTrue
                } else {
                    Op::ConstantFalse
                });

                instruction.set_type(type_id);
                instruction.set_result(id);
                (instruction, id)
            }
            crate::ConstantInner::Composite(ref constituents) => {
                let type_id = self.get_type_id(arena, constant.ty);

                let mut instruction = Instruction::new(Op::ConstantComposite);
                instruction.set_type(type_id);
                instruction.set_result(id);

                for constituent in constituents.iter() {
                    let id = self.get_constant_id(*constituent, &ir_module);
                    instruction.add_operand(id);
                }

                (instruction, id)
            }
        }
    }

    fn create_pointer(
        &mut self,
        type_id: Word,
        handle: crate::Handle<crate::Type>,
        class: StorageClass,
    ) -> Word {
        let pointer_id = self.generate_id();
        let mut instruction = Instruction::new(Op::TypePointer);
        instruction.set_result(pointer_id);
        instruction.add_operand((class) as u32);
        instruction.add_operand(type_id);

        self.lookup_type.insert(
            pointer_id,
            LookupType::Standalone(crate::Type {
                name: None,
                inner: crate::TypeInner::Pointer {
                    base: handle,
                    class,
                },
            }),
        );

        instruction.to_words(&mut self.logical_layout.type_declarations);
        pointer_id
    }

    fn get_pointer_id(
        &mut self,
        arena: &crate::Arena<crate::Type>,
        handle: crate::Handle<crate::Type>,
        class: StorageClass,
    ) -> Word {
        let ty = &arena[handle];
        let type_id = self.get_type_id(arena, handle);
        match ty.inner {
            crate::TypeInner::Pointer { .. } => type_id,
            _ => {
                self.create_pointer(type_id, handle, class)
            }
        }
    }

    fn instruction_global_variable(
        &mut self,
        arena: &crate::Arena<crate::Type>,
        global_variable: &crate::GlobalVariable,
        handle: crate::Handle<crate::GlobalVariable>,
    ) -> (Instruction, Word) {
        let mut instruction = Instruction::new(Op::Variable);
        let id = self.generate_id();

        self.try_add_capabilities(global_variable.class.required_capabilities());

        let pointer_id = self.get_pointer_id(arena, global_variable.ty, global_variable.class);

        instruction.set_type(pointer_id);
        instruction.set_result(id);
        instruction.add_operand(global_variable.class as u32);

        if self.parser_flags.contains(ParserFlags::DEBUG) {
            let mut debug_instruction = Instruction::new(Op::Name);
            debug_instruction.set_result(id);
            debug_instruction.add_operands(
                self.string_to_words(global_variable.name.as_ref().unwrap().as_str()),
            );
            self.debugs.push(debug_instruction);
        }

        match global_variable.binding.as_ref().unwrap() {
            crate::Binding::Location(location) => {
                let mut instruction = Instruction::new(Op::Decorate);
                instruction.add_operand(id);
                instruction.add_operand(Decoration::Location as u32);
                instruction.add_operand(*location);
                self.annotations.push(instruction);
            }
            crate::Binding::Descriptor { set, binding } => {
                let mut set_instruction = Instruction::new(Op::Decorate);
                set_instruction.add_operand(id);
                set_instruction.add_operand(Decoration::DescriptorSet as u32);
                set_instruction.add_operand(*set);
                self.annotations.push(set_instruction);

                let mut binding_instruction = Instruction::new(Op::Decorate);
                binding_instruction.add_operand(id);
                binding_instruction.add_operand(Decoration::Binding as u32);
                binding_instruction.add_operand(*binding);
                self.annotations.push(binding_instruction);
            }
            crate::Binding::BuiltIn(built_in) => {
                let built_in_u32: u32 = unsafe { std::mem::transmute(*built_in) };

                let mut instruction = Instruction::new(Op::Decorate);
                instruction.add_operand(id);
                instruction.add_operand(Decoration::BuiltIn as u32);
                instruction.add_operand(built_in_u32);
                self.annotations.push(instruction);
            }
        }

        // TODO Initializer is optional and not (yet) included in the IR

        self.lookup_global_variable
            .insert(id, LookupType::Handle(handle));
        (instruction, id)
    }

    fn write_physical_layout(&mut self) {
        self.physical_layout.bound = self.id_count + 1;
    }

    fn instruction_source(&self) -> Instruction {
        let version = 450u32;

        let mut instruction = Instruction::new(Op::Source);
        instruction.add_operand(SourceLanguage::GLSL as u32);
        instruction.add_operands(self.bytes_to_words(&version.to_le_bytes()));
        instruction
    }

    fn instruction_function_type(&mut self, lookup_function_type: LookupFunctionType) -> Word {
        let mut id = None;

        for (k, v) in self.lookup_function_type.iter() {
            if v.eq(&lookup_function_type) {
                id = Some(*k);
                break;
            }
        }

        if id.is_none() {
            let _id = self.generate_id();
            id = Some(_id);

            let mut instruction = Instruction::new(Op::TypeFunction);
            instruction.set_result(_id);
            instruction.add_operand(lookup_function_type.return_type_id);

            for parameter_type_id in lookup_function_type.parameter_type_ids.iter() {
                instruction.add_operand(*parameter_type_id);
            }

            self.lookup_function_type.insert(_id, lookup_function_type);
            instruction.to_words(&mut self.logical_layout.type_declarations);
        }

        id.unwrap()
    }

    fn instruction_function(
        &mut self,
        handle: crate::Handle<crate::Function>,
        function: &crate::Function,
        arena: &crate::Arena<crate::Type>,
    ) -> Instruction {
        let id = self.generate_id();

        let return_type_id = self.get_function_type(function.return_type, arena);

        let mut instruction = Instruction::new(Op::Function);
        instruction.set_type(return_type_id);
        instruction.set_result(id);

        let control_u32: Word = unsafe { std::mem::transmute(function.control) };

        instruction.add_operand(control_u32);

        let mut parameter_type_ids = Vec::with_capacity(function.parameter_types.len());
        for parameter_type in function.parameter_types.iter() {
            parameter_type_ids.push(self.get_type_id(arena, *parameter_type))
        }

        let lookup_function_type = LookupFunctionType {
            return_type_id,
            parameter_type_ids,
        };

        let type_function_id = self.instruction_function_type(lookup_function_type);

        instruction.add_operand(type_function_id);

        self.lookup_function.insert(id, LookupType::Handle(handle));

        instruction
    }

    fn get_type_by_inner(
        &self,
        arena: &crate::Arena<crate::Type>,
        inner: &crate::TypeInner,
    ) -> Word {
        self.lookup_type
            .iter()
            .find(|(_, v)| match v {
                LookupType::Handle(handle) => {
                    if *(&arena[*handle].inner.eq(inner)) {
                        true
                    } else {
                        false
                    }
                }
                LookupType::Standalone(ty) => {
                    if ty.inner.eq(inner) {
                        true
                    } else {
                        false
                    }
                }
            })
            .map(|(k, _)| *k)
            .unwrap()
    }

    fn parse_expression<'a>(
        &mut self,
        ir_module: &'a crate::Module,
        function: &crate::Function,
        expression: &crate::Expression,
        output: &mut Vec<Instruction>,
    ) -> (Word, &'a crate::TypeInner) {
        match expression {
            crate::Expression::GlobalVariable(handle) => {
                let var = &ir_module.global_variables[*handle];
                let inner = &ir_module.types[var.ty].inner;
                let id = self.get_global_variable_id(
                    &ir_module.types,
                    &ir_module.global_variables,
                    *handle,
                );
                (id, inner)
            }
            crate::Expression::Constant(handle) => {
                let var = &ir_module.constants[*handle];
                let inner = &ir_module.types[var.ty].inner;
                let id = self.get_constant_id(*handle, ir_module);
                (id, inner)
            }
            crate::Expression::Compose { ty, components } => {
                let var = &ir_module.types[*ty];
                let inner = &var.inner;
                let id = self.generate_id();
                let type_id = self.get_type_id(&ir_module.types, *ty);

                let mut instruction = Instruction::new(Op::CompositeConstruct);
                instruction.set_type(type_id);
                instruction.set_result(id);

                for component in components {
                    let expression = &function.expressions[*component];
                    let (component_id, _) =
                        self.parse_expression(ir_module, &function, expression, output);
                    instruction.add_operand(component_id);
                }

                output.push(instruction);

                (id, inner)
            }
            crate::Expression::Binary { op, left, right } => {
                let mut instruction;

                let left_expression = &function.expressions[*left];
                let right_expression = &function.expressions[*right];
                let (left_id, left_inner) =
                    self.parse_expression(ir_module, function, left_expression, output);
                let (right_id, _) =
                    self.parse_expression(ir_module, function, right_expression, output);
                match op {
                    crate::BinaryOperator::Add => {
                        // TODO Always assuming now that left and right are the same type
                        match left_inner {
                            crate::TypeInner::Scalar { kind, .. } => match kind {
                                crate::ScalarKind::Sint | crate::ScalarKind::Uint => {
                                    instruction = Instruction::new(Op::IAdd);
                                }
                                crate::ScalarKind::Float => {
                                    instruction = Instruction::new(Op::FAdd);
                                }
                                _ => unimplemented!("{:?}", kind),
                            },
                            crate::TypeInner::Vector {
                                size: _,
                                kind,
                                width: _,
                            } => match kind {
                                crate::ScalarKind::Sint => {
                                    instruction = Instruction::new(Op::IAdd);
                                }
                                crate::ScalarKind::Float => {
                                    instruction = Instruction::new(Op::FAdd);
                                }
                                _ => unimplemented!("{:?}", kind),
                            },
                            _ => unimplemented!("{:?}", left_inner),
                        }
                    }
                    crate::BinaryOperator::Subtract => {
                        // TODO Always assuming now that left and right are the same type
                        match left_inner {
                            crate::TypeInner::Scalar { kind, .. } => match kind {
                                crate::ScalarKind::Sint => {
                                    instruction = Instruction::new(Op::SNegate);
                                }
                                crate::ScalarKind::Float => {
                                    instruction = Instruction::new(Op::FNegate);
                                }
                                _ => unimplemented!("{:?}", kind),
                            },
                            crate::TypeInner::Vector {
                                size: _,
                                kind,
                                width: _,
                            } => match kind {
                                crate::ScalarKind::Sint => {
                                    instruction = Instruction::new(Op::SNegate);
                                }
                                crate::ScalarKind::Float => {
                                    instruction = Instruction::new(Op::FNegate);
                                }
                                _ => unimplemented!("{:?}", kind),
                            },
                            _ => unimplemented!("{:?}", left_inner),
                        }
                    }
                    crate::BinaryOperator::Multiply => {
                        // TODO OpVectorTimesScalar is only supported
                        instruction = Instruction::new(Op::VectorTimesScalar);
                    }
                    crate::BinaryOperator::Divide => {
                        // TODO Always assuming now that left and right are the same type
                        match left_inner {
                            crate::TypeInner::Scalar { kind, .. } => match kind {
                                crate::ScalarKind::Uint => {
                                    instruction = Instruction::new(Op::UDiv);
                                }
                                crate::ScalarKind::Sint => {
                                    instruction = Instruction::new(Op::SDiv);
                                }
                                crate::ScalarKind::Float => {
                                    instruction = Instruction::new(Op::FDiv);
                                }
                                _ => unimplemented!("{:?}", kind),
                            },
                            crate::TypeInner::Vector {
                                size: _,
                                kind,
                                width: _,
                            } => match kind {
                                crate::ScalarKind::Uint => {
                                    instruction = Instruction::new(Op::UDiv);
                                }
                                crate::ScalarKind::Sint => {
                                    instruction = Instruction::new(Op::SDiv);
                                }
                                crate::ScalarKind::Float => {
                                    instruction = Instruction::new(Op::FDiv);
                                }
                                _ => unimplemented!("{:?}", kind),
                            },
                            _ => unimplemented!("{:?}", left_inner),
                        }
                    }
                    crate::BinaryOperator::Equal => {
                        // TODO Always assuming now that left and right are the same type
                        match left_inner {
                            crate::TypeInner::Scalar { kind, .. } => match kind {
                                crate::ScalarKind::Uint => {
                                    instruction = Instruction::new(Op::IEqual);
                                }
                                crate::ScalarKind::Sint => {
                                    instruction = Instruction::new(Op::IEqual);
                                }
                                _ => unimplemented!("{:?}", kind),
                            },
                            crate::TypeInner::Vector {
                                size: _,
                                kind,
                                width: _,
                            } => match kind {
                                crate::ScalarKind::Uint | crate::ScalarKind::Sint => {
                                    instruction = Instruction::new(Op::IEqual);
                                }
                                _ => unimplemented!("{:?}", kind),
                            },
                            _ => unimplemented!("{:?}", left_inner),
                        }
                    }
                    crate::BinaryOperator::Less => {
                        // TODO Always assuming now that left and right are the same type
                        match left_inner {
                            crate::TypeInner::Scalar { kind, .. } => match kind {
                                crate::ScalarKind::Uint => {
                                    instruction = Instruction::new(Op::ULessThan);
                                }
                                crate::ScalarKind::Sint => {
                                    instruction = Instruction::new(Op::SLessThan);
                                }
                                crate::ScalarKind::Float => {
                                    instruction = Instruction::new(Op::FOrdLessThan);
                                }
                                _ => unimplemented!("{:?}", kind),
                            },
                            crate::TypeInner::Vector {
                                size: _,
                                kind,
                                width: _,
                            } => match kind {
                                crate::ScalarKind::Uint => {
                                    instruction = Instruction::new(Op::ULessThan);
                                }
                                crate::ScalarKind::Sint => {
                                    instruction = Instruction::new(Op::SLessThan);
                                }
                                crate::ScalarKind::Float => {
                                    instruction = Instruction::new(Op::FOrdLessThan);
                                }
                                _ => unimplemented!("{:?}", kind),
                            },
                            _ => unimplemented!("{:?}", left_inner),
                        }
                    }
                    crate::BinaryOperator::Greater => {
                        // TODO Always assuming now that left and right are the same type
                        match left_inner {
                            crate::TypeInner::Scalar { kind, .. } => match kind {
                                crate::ScalarKind::Uint => {
                                    instruction = Instruction::new(Op::UGreaterThan);
                                }
                                crate::ScalarKind::Sint => {
                                    instruction = Instruction::new(Op::SGreaterThan);
                                }
                                crate::ScalarKind::Float => {
                                    instruction = Instruction::new(Op::FOrdGreaterThan);
                                }
                                _ => unimplemented!("{:?}", kind),
                            },
                            crate::TypeInner::Vector {
                                size: _,
                                kind,
                                width: _,
                            } => match kind {
                                crate::ScalarKind::Uint => {
                                    instruction = Instruction::new(Op::UGreaterThan);
                                }
                                crate::ScalarKind::Sint => {
                                    instruction = Instruction::new(Op::SGreaterThan);
                                }
                                crate::ScalarKind::Float => {
                                    instruction = Instruction::new(Op::FOrdGreaterThan);
                                }
                                _ => unimplemented!("{:?}", kind),
                            },
                            _ => unimplemented!("{:?}", left_inner),
                        }
                    }
                    crate::BinaryOperator::GreaterEqual => {
                        // TODO Always assuming now that left and right are the same type
                        match left_inner {
                            crate::TypeInner::Scalar { kind, .. } => match kind {
                                crate::ScalarKind::Uint => {
                                    instruction = Instruction::new(Op::UGreaterThanEqual);
                                }
                                crate::ScalarKind::Sint => {
                                    instruction = Instruction::new(Op::SGreaterThanEqual);
                                }
                                _ => unimplemented!("{:?}", kind),
                            },
                            crate::TypeInner::Vector {
                                size: _,
                                kind,
                                width: _,
                            } => match kind {
                                crate::ScalarKind::Uint => {
                                    instruction = Instruction::new(Op::UGreaterThanEqual);
                                }
                                crate::ScalarKind::Sint => {
                                    instruction = Instruction::new(Op::SGreaterThanEqual);
                                }
                                _ => unimplemented!("{:?}", kind),
                            },
                            _ => unimplemented!("{:?}", left_inner),
                        }
                    }
                    _ => unimplemented!("{:?}", op),
                }

                let id = self.generate_id();

                // TODO TypeBool is always created, while only one is needed
                //  Can't fix this right now, because lookup has handles
                if self.bool_type.is_none() {
                    let mut bool_instruction = Instruction::new(Op::TypeBool);
                    let bool_id = self.generate_id();
                    bool_instruction.set_result(bool_id);
                    bool_instruction.to_words(&mut self.logical_layout.type_declarations);

                    self.bool_type = Some(bool_id);
                }

                // TODO Check how to do this properly,
                //  without having to manually create a Load instruction here
                let (kind, width) = match left_inner {
                    crate::TypeInner::Scalar {
                        kind, width
                    } => (kind, width),
                    _ => unimplemented!("{:?}", left_inner)
                };

                let scalar_handle = self.find_scalar_handle(&ir_module.types, kind, width);

                let mut load = Instruction::new(Op::Load);
                let load_id = self.generate_id();
                load.set_type(self.get_type_id(&ir_module.types, scalar_handle));
                load.set_result(load_id);
                load.add_operand(left_id);
                output.push(load);

                instruction.set_type(self.bool_type.unwrap());
                instruction.set_result(id);
                instruction.add_operand(load_id);
                instruction.add_operand(right_id);
                output.push(instruction);
                (id, left_inner)
            }
            crate::Expression::LocalVariable(variable) => {

                let var = &function.local_variables[*variable];
                let ty = &ir_module.types[var.ty];

                if self.lookup_local_variable.contains_key(var.name.as_ref().unwrap().as_str()) {
                    (*self.lookup_local_variable.get(var.name.as_ref().unwrap().as_str()).unwrap(), &ty.inner)
                } else {
                    let id = self.generate_id();


                    let pointer_id =
                        self.get_pointer_id(&ir_module.types, var.ty, StorageClass::Function);

                    let mut instruction = Instruction::new(Op::Variable);
                    instruction.set_type(pointer_id);
                    instruction.set_result(id);
                    instruction.add_operand(StorageClass::Function as u32);
                    output.push(instruction);

                    if self.parser_flags.contains(ParserFlags::DEBUG) {
                        let mut debug_instruction = Instruction::new(Op::Name);
                        debug_instruction.set_result(id);
                        debug_instruction.add_operands(self.string_to_words(var.name.as_ref().unwrap().as_str()));
                        self.debugs.push(debug_instruction);
                    }

                    self.lookup_local_variable.insert(var.name.as_ref().unwrap().parse().unwrap(), id);
                    (id, &ty.inner)
                }
            }
            crate::Expression::AccessIndex { base, index } => {
                let (base_id, base_inner) = self.parse_expression(ir_module, function, &function.expressions[*base], output);

                let mut scalar_id;
                let base_pointer = match base_inner {
                    crate::TypeInner::Vector { kind, width, .. } => {
                        let scalar_handle = self.find_scalar_handle(&ir_module.types, &kind, &width);
                        scalar_id = self.get_type_id(&ir_module.types, scalar_handle);
                        // TODO Always passing Input Storage class.
                        //  Storage Class needs to be equal to the base pointer Storage Class.
                        self.create_pointer(scalar_id, scalar_handle, StorageClass::Input)
                    }
                    _ => unimplemented!("{:?}", base_inner)
                };

                let mut instruction = Instruction::new(Op::AccessChain);
                let id = self.generate_id();
                instruction.set_type(base_pointer);
                instruction.set_result(id);
                instruction.add_operand(base_id);

                // TODO Find a cleaner way to do this.
                match base_inner {
                    crate::TypeInner::Vector { kind, width, .. } => {
                        let inner = match kind {
                            crate::ScalarKind::Uint => crate::ConstantInner::Uint((*index) as u64),
                            _ => unimplemented!("{:?}", kind)
                        };

                        let constant = crate::Constant {
                          name: None,
                            specialization: None,
                            ty: self.find_scalar_handle(&ir_module.types, &kind, &width),
                            inner
                        };

                        let (const_instruction, id) = self.instruction_constant_type(&LookupType::Standalone(constant), ir_module);

                        // TODO Duplicate couple to by pass moved value
                        self.lookup_constant.insert(id, LookupType::Standalone(crate::Constant {
                            name: None,
                            specialization: None,
                            ty: self.find_scalar_handle(&ir_module.types, &kind, &width),
                            inner: match kind {
                                crate::ScalarKind::Uint => crate::ConstantInner::Uint((*index) as u64),
                                _ => unimplemented!("{:?}", kind)
                            }
                        }));

                        const_instruction.to_words(&mut self.logical_layout.constants);
                        instruction.add_operand(id);
                    }
                    _ => unimplemented!("{:?}", base_inner)
                }

                output.push(instruction);

                let mut load = Instruction::new(Op::Load);
                let load_id = self.generate_id();
                load.set_type(scalar_id);
                load.set_result(load_id);
                load.add_operand(id);
                output.push(load);

                (load_id, base_inner)
            }
            crate::Expression::Access { base, index: _ } => {
                self.parse_expression(ir_module, function, &function.expressions[*base], output)
            }
            crate::Expression::Call { name, arguments } => {
                let id = self.generate_id();
                let name_u32 = match name.as_str() {
                    "sin" => 13u32,
                    "cos" => 14u32,
                    "atan2" => 25u32,
                    "fclamp" => 43u32,
                    "length" => 66u32,
                    "normalize" => 69u32,
                    _ => unimplemented!("{:?}", name),
                };

                let mut instruction = Instruction::new(Op::ExtInst);
                let inner = &crate::TypeInner::Scalar {
                    kind: crate::ScalarKind::Float,
                    width: 32,
                };
                let type_id = self.get_type_by_inner(&ir_module.types, inner);
                instruction.set_type(type_id);
                instruction.set_result(id);

                // TODO Support other imports
                //  There is always one key for now
                for (k, _) in self.lookup_import.iter() {
                    instruction.add_operand(*k);
                }

                instruction.add_operand(name_u32);

                for arg in arguments {
                    let (id, _) = self.parse_expression(
                        ir_module,
                        function,
                        &function.expressions[*arg],
                        output,
                    );
                    instruction.add_operand(id);
                }
                output.push(instruction);
                (id, inner)
            }
            crate::Expression::Unary { op, expr } => {
                let expression = &function.expressions[*expr];
                let (operand_id, operand_inner) =
                    self.parse_expression(ir_module, function, expression, output);

                match op {
                    crate::UnaryOperator::Negate => {
                        let mut instruction;
                        let id = self.generate_id();
                        match operand_inner {
                            crate::TypeInner::Scalar { kind, .. } => match kind {
                                crate::ScalarKind::Sint => {
                                    instruction = Instruction::new(Op::SNegate);
                                    let inner = &crate::TypeInner::Scalar {
                                        kind: crate::ScalarKind::Sint,
                                        width: 32,
                                    };
                                    let type_id = self.get_type_by_inner(&ir_module.types, inner);
                                    instruction.set_type(type_id);
                                }
                                crate::ScalarKind::Float => {
                                    instruction = Instruction::new(Op::FNegate);
                                    let inner = &crate::TypeInner::Scalar {
                                        kind: crate::ScalarKind::Float,
                                        width: 32,
                                    };
                                    let type_id = self.get_type_by_inner(&ir_module.types, inner);
                                    instruction.set_type(type_id);
                                }
                                _ => unimplemented!("{:?}", kind),
                            },
                            _ => unimplemented!("{:?}", operand_inner),
                        }
                        instruction.set_result(id);
                        instruction.add_operand(operand_id);
                        output.push(instruction);
                        (id, operand_inner)
                    }
                    _ => unimplemented!("{:?}", op),
                }
            }
            _ => unimplemented!("{:?}", expression),
        }
    }

    fn instruction_function_block(
        &mut self,
        ir_module: &crate::Module,
        function: &crate::Function,
        statement: &crate::Statement,
        output: &mut Vec<Instruction>,
        label: Word,
    ) {
        match statement {
            crate::Statement::Return { value: _ } => match function.return_type {
                Some(_) => {}
                None => output.push(Instruction::new(Op::Return)),
            },
            crate::Statement::Store { pointer, value } => {
                let mut instruction = Instruction::new(Op::Store);

                let pointer_expression = &function.expressions[*pointer];
                let value_expression = &function.expressions[*value];
                let (pointer_id, _) =
                    self.parse_expression(ir_module, function, pointer_expression, output);
                let (value_id, _) =
                    self.parse_expression(ir_module, function, value_expression, output);

                instruction.add_operand(pointer_id);
                instruction.add_operand(value_id);

                output.push(instruction)
            }
            crate::Statement::If {
                condition,
                accept,
                reject,
            } => {
                let (condition_id, _) = self.parse_expression(
                    ir_module,
                    function,
                    &function.expressions[*condition],
                    output,
                );

                let (merge_label, merge_label_instruction) = self.instruction_label();
                let mut selection_merge = Instruction::new(Op::SelectionMerge);
                selection_merge.add_operand(merge_label);
                selection_merge.add_operand(SelectionControl::NONE.bits());

                let (accept_label, accept_label_instruction) = self.instruction_label();
                let (reject_label, reject_instruction) = if reject.is_empty() {
                    (merge_label, merge_label_instruction)
                } else {
                    self.instruction_label()
                };

                let mut branch_conditional = Instruction::new(Op::BranchConditional);
                branch_conditional.add_operand(condition_id);
                branch_conditional.add_operand(accept_label);
                branch_conditional.add_operand(reject_label);

                output.push(selection_merge);
                output.push(branch_conditional);
                output.push(accept_label_instruction);

                for statement in accept {
                    self.instruction_function_block(
                        ir_module,
                        function,
                        statement,
                        output,
                        accept_label);
                }

                output.push(reject_instruction);

                for statement in reject {
                    self.instruction_function_block(
                        ir_module,
                        function,
                        statement,
                        output,
                        reject_label);
                }
            }
            crate::Statement::Loop { body, continuing } => {

                let mut body_output = vec![];
                let (begin_body_label_id, begin_body_label_instruction) = self.instruction_label();
                for statement in body.iter() {
                    self.instruction_function_block(
                        ir_module,
                        function,
                        statement,
                        &mut body_output,
                        begin_body_label_id);
                }

                let (end_body_label_id, end_body_label_instruction) = self.instruction_label();

                let (continue_label_id, continue_label_instruction) = self.instruction_label();
                let mut continuing_output = vec![];
                for statement in continuing.iter() {
                    self.instruction_function_block(
                        ir_module,
                        function,
                        statement,
                        &mut continuing_output,
                        continue_label_id);
                }
                let mut branch = Instruction::new(Op::Branch);
                branch.add_operand(label);

                let mut loop_merge = Instruction::new(Op::LoopMerge);
                loop_merge.add_operand(end_body_label_id);
                loop_merge.add_operand(continue_label_id);
                loop_merge.add_operand(LoopControl::NONE.bits());

                output.push(loop_merge);
                output.push(begin_body_label_instruction);
                output.push(end_body_label_instruction);
                output.append(&mut body_output);
                output.append(&mut continuing_output);
            }
            crate::Statement::Continue => {}
            crate::Statement::Break => {
                output.push(Instruction::new(Op::Return));
            }
            crate::Statement::Empty => {}
            _ => unimplemented!("{:?}", statement),
        }
    }

    fn instruction_label(&mut self) -> (Word, Instruction) {
        let id = self.generate_id();
        let mut instruction = Instruction::new(Op::Label);
        instruction.set_result(id);
        (id, instruction)
    }

    fn instruction_function_end(&self) -> Instruction {
        Instruction::new(Op::FunctionEnd)
    }

    fn write_logical_layout(&mut self, ir_module: &crate::Module) {
        self.instruction_ext_inst_import()
            .to_words(&mut self.logical_layout.ext_inst_imports);

        if self.parser_flags.contains(ParserFlags::DEBUG) {
            self.debugs.push(self.instruction_source());
        }

        for (handle, function) in ir_module.functions.iter() {
            self.lookup_local_variable = FastHashMap::default();

            let mut function_instructions: Vec<Instruction> = vec![];
            function_instructions.push(self.instruction_function(
                handle,
                function,
                &ir_module.types,
            ));

            let (label_id, label_instruction) = self.instruction_label();
            function_instructions.push(label_instruction);

            for block in function.body.iter() {
                let mut output: Vec<Instruction> = vec![];
                self.instruction_function_block(ir_module, function, &block, &mut output, label_id);
                function_instructions.append(&mut output);
            }

            function_instructions.push(self.instruction_function_end());
            for instruction in function_instructions.iter() {
                instruction.to_words(&mut self.logical_layout.function_definitions);
            }
        }

        for entry_point in ir_module.entry_points.iter() {
            let entry_point_instruction = self.instruction_entry_point(entry_point, ir_module);
            entry_point_instruction.to_words(&mut self.logical_layout.entry_points);
        }

        // Looking through all global variable, types, constants.
        // Doing this because we also want to include not used parts of the module
        // to be included in the output
        for (handle, _) in ir_module.global_variables.iter() {
            let _ =
                self.get_global_variable_id(&ir_module.types, &ir_module.global_variables, handle);
        }

        for (handle, _) in ir_module.types.iter() {
            let _ = self.get_type_id(&ir_module.types, handle);
        }

        for (handle, _) in ir_module.constants.iter() {
            let _ = self.get_constant_id(handle, &ir_module);
        }

        for annotation in self.annotations.iter() {
            annotation.to_words(&mut self.logical_layout.annotations);
        }

        for capability in self.capabilities.iter() {
            let instruction = self.instruction_capability(capability);
            instruction.to_words(&mut self.logical_layout.capabilities);
        }

        self.instruction_memory_model()
            .to_words(&mut self.logical_layout.memory_model);

        if self.parser_flags.contains(ParserFlags::DEBUG) {
            for debug in self.debugs.iter() {
                debug.to_words(&mut self.logical_layout.debugs);
            }
        }
    }

    pub fn parse(&mut self, ir_module: &crate::Module) -> Vec<Word> {
        let mut words: Vec<Word> = vec![];

        self.write_logical_layout(ir_module);
        self.write_physical_layout();

        self.physical_layout.in_words(&mut words);
        self.logical_layout.in_words(&mut words);
        words
    }
}
