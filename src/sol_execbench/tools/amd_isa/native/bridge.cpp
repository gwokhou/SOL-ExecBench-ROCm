// SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
// SPDX-License-Identifier: Apache-2.0

// A deliberately small JSON-lines adapter around AMD's C++ API.  Keeping the
// ABI boundary here lets Python callers use stable project-owned models while
// isa_spec_manager remains an implementation detail.

#include <amdisa/isa_decoder.h>
#include <amdisa/isa_explorer.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <json.hpp>

using json = nlohmann::json;

namespace {

constexpr int kProtocolVersion = 1;
constexpr size_t kMaxMessageBytes = 64 * 1024 * 1024;

json encode(const amdisa::EncodingField& value) {
    return {{"name", value.field_name}, {"value", value.field_value},
            {"bit_count", value.bit_count}, {"bit_offset", value.bit_offset}};
}

json encode(const amdisa::InstructionOperand& value) {
    return {{"name", value.operand_name}, {"size", value.operand_size},
            {"data_format", value.data_format},
            {"encoding_field", value.encoding_field_name},
            {"is_input", value.is_input}, {"is_output", value.is_output}};
}

json encode(const amdisa::OperandModifer& value) {
    return {{"name", value.modifier_name}, {"value", value.value}};
}

json encode(const amdisa::InstructionInfo& value) {
    json fields = json::array();
    for (const auto& item : value.encoding_fields) fields.push_back(encode(item));
    json operands = json::array();
    for (const auto& item : value.instruction_operands) operands.push_back(encode(item));
    json modifiers = json::array();
    for (const auto& item : value.operand_modifiers) modifiers.push_back(encode(item));
    const auto& branch = value.instruction_semantic_info.branch_info;
    json subgroups = json::array();
    for (const auto item : value.functional_group_subgroup_info.isa_functional_subgroups) {
        subgroups.push_back(amdisa::FunctionalSubgroupNames[static_cast<int>(item)]);
    }
    return {{"name", value.instruction_name}, {"aliases", value.aliased_names},
            {"description", value.instruction_description},
            {"encoding_name", value.encoding_name},
            {"encoding_description", value.encoding_description},
            {"encoding_layout", value.encoding_layout}, {"encoding_fields", fields},
            {"operands", operands}, {"modifiers", modifiers},
            {"semantic", {{"is_branch", branch.is_branch},
                          {"is_conditional_branch", branch.is_conditional},
                          {"is_indirect_branch", branch.is_indirect},
                          {"branch_offset", branch.branch_offset},
                          {"branch_target_pc", branch.branch_target_pc},
                          {"branch_target_label", branch.branch_target_label},
                          {"branch_target_index", branch.branch_target_index},
                          {"is_program_terminator", value.instruction_semantic_info.is_program_terminator},
                          {"is_immediately_executed", value.instruction_semantic_info.is_immediately_executed}}},
            {"functional", {{"group", amdisa::FunctionalGroupNames[static_cast<int>(value.functional_group_subgroup_info.isa_functional_group)]},
                              {"subgroups", subgroups},
                              {"description", value.functional_group_subgroup_info.description}}}};
}

json encode(const amdisa::InstructionInfoBundle& value) {
    json result = json::array();
    for (const auto& item : value.bundle) result.push_back(encode(item));
    return result;
}

json encode(const amdisa::explorer::PredefinedValue& value) {
    return {{"name", value.Name()}, {"description", value.Description()}, {"value", value.Value()}};
}

json encode(const amdisa::explorer::Range& value) {
    const auto& padding = value.GetPadding();
    return {{"order", value.Order()}, {"bit_count", value.BitCount()},
            {"bit_offset", value.BitOffset()},
            {"padding", {{"bit_count", padding.BitCount()}, {"value", padding.Value()}}}};
}

json encode(const amdisa::explorer::Field& value) {
    json ranges = json::array();
    for (const auto& item : value.Ranges()) ranges.push_back(encode(item));
    json predefined = json::array();
    for (const auto& item : value.PredefinedValues()) predefined.push_back(encode(item));
    return {{"name", value.Name()}, {"description", value.Description()}, {"type", value.Type()},
            {"is_conditional", value.IsConditional()}, {"is_constant", value.IsConstant()},
            {"range_count", value.RangeCount()}, {"signedness", static_cast<int>(value.GetSignedness())},
            {"ranges", ranges}, {"predefined_values", predefined}};
}

json encode(const amdisa::explorer::DataAttributes& value) {
    json fields = json::array();
    for (const auto& item : value.BitMap()) fields.push_back(encode(item));
    return {{"order", value.Order()}, {"bit_map", fields}};
}

json encode(const amdisa::explorer::DataFormat& value) {
    json attributes = json::array();
    for (const auto& item : value.DataAttr()) attributes.push_back(encode(item));
    return {{"name", value.Name()}, {"description", value.Description()},
            {"data_type", static_cast<int>(value.DType())}, {"bit_count", value.BitCount()},
            {"component_count", value.ComponentCount()}, {"attributes", attributes}};
}

json encode(const amdisa::explorer::OperandType& value) {
    json predefined = json::array();
    for (const auto& item : value.PredefinedValues()) predefined.push_back(encode(item));
    return {{"name", value.Name()}, {"description", value.Description()},
            {"is_partitioned", value.IsPartitioned()}, {"predefined_values", predefined}};
}

json encode(const amdisa::explorer::Operand& value) {
    return {{"field_name", value.FieldName()}, {"encoding_field_name", value.EncodingFieldName()},
            {"data_format", value.DataFmt().Name()}, {"operand_type", value.Type().Name()},
            {"order", value.Order()}, {"size", value.Size()}, {"is_input", value.IsInput()},
            {"is_output", value.IsOutput()}, {"is_implicit", value.IsImplicit()},
            {"is_in_microcode", value.IsInMicrocode()}};
}

json encode(const amdisa::explorer::InstructionEncoding& value) {
    json operands = json::array();
    for (const auto& item : value.Operands()) operands.push_back(encode(item));
    return {{"name", value.Name()}, {"opcode", value.Opcode()}, {"operands", operands}};
}

json encode(const amdisa::explorer::Instruction& value) {
    json encodings = json::array();
    for (const auto& item : value.Encodings()) encodings.push_back(encode(item));
    json subgroups = json::array();
    for (const auto& item : value.FuncSubgroups()) subgroups.push_back(item.Name());
    const auto* group = value.FuncGroup();
    return {{"name", value.Name()}, {"description", value.Description()},
            {"is_branch", value.IsBranch()}, {"is_conditional_branch", value.IsConditionalBranch()},
            {"is_indirect_branch", value.IsIndirectBranch()},
            {"is_immediately_executed", value.IsImmediatelyExecuted()},
            {"is_program_terminator", value.IsProgramTerminator()},
            {"functional_group", group == nullptr ? "" : group->Name()},
            {"functional_subgroups", subgroups}, {"encodings", encodings}};
}

json encode(const amdisa::explorer::FunctionalGroup& value) {
    json instructions = json::array();
    for (const auto* item : value.Instructions()) instructions.push_back(item->Name());
    json subgroups = json::array();
    for (const auto& item : value.FuncSubgroups()) subgroups.push_back(item.Name());
    return {{"name", value.Name()}, {"description", value.Description()},
            {"instructions", instructions}, {"subgroups", subgroups}};
}

template <typename Value>
json encode_map(const std::map<std::string, Value>& values) {
    json result = json::array();
    for (const auto& item : values) result.push_back(encode(item.second));
    return result;
}

class Bridge {
public:
    json Dispatch(const json& request) {
        if (!request.is_object() || request.value("protocol_version", 0) != kProtocolVersion ||
            !request.contains("method") || !request.at("method").is_string()) {
            throw std::invalid_argument("invalid protocol request");
        }
        const auto method = request.at("method").get<std::string>();
        const auto params = request.value("params", json::object());
        if (method == "hello") return {{"protocol_version", kProtocolVersion}, {"decoder_version", decoder_.GetVersion()}};
        if (method == "shutdown") return {{"status", "stopping"}};
        if (method == "load") return Load(params.at("spec_path").get<std::string>());
        if (!loaded_) throw std::runtime_error("load must be called before ISA operations");
        if (method == "decoder.get_instruction") return GetInstruction(params);
        if (method == "decoder.decode_stream") return DecodeStream(params);
        if (method == "decoder.decode_disassembly") return DecodeDisassembly(params);
        if (method == "explorer.architecture") return {{"name", explorer_.GetArchitecture().Name()}};
        if (method == "explorer.list_instructions") return encode_map(explorer_.GetInstructions());
        if (method == "explorer.get_instruction") return GetExplorerInstruction(params);
        if (method == "explorer.list_data_formats") return encode_map(explorer_.GetDataFormats());
        if (method == "explorer.get_data_format") return GetDataFormat(params);
        if (method == "explorer.list_operand_types") return encode_map(explorer_.GetOperandTypes());
        if (method == "explorer.get_operand_type") return GetOperandType(params);
        if (method == "explorer.list_functional_groups") return encode_map(explorer_.GetFunctionalGroups());
        if (method == "explorer.get_functional_group") return GetFunctionalGroup(params);
        throw std::invalid_argument("unknown method: " + method);
    }

private:
    json Load(const std::string& path) {
        std::string error;
        if (!decoder_.Initialize(path, error)) throw std::runtime_error(error);
        if (!explorer_.Init(path, error)) throw std::runtime_error(error);
        loaded_ = true;
        return {{"architecture", explorer_.GetArchitecture().Name()}, {"decoder_version", decoder_.GetVersion()}};
    }

    json GetInstruction(const json& params) const {
        std::string error;
        if (params.contains("name")) {
            amdisa::InstructionInfo info;
            if (!decoder_.DecodeInstruction(params.at("name").get<std::string>(), info, error)) throw std::runtime_error(error);
            return encode(info);
        }
        amdisa::InstructionInfoBundle bundle;
        if (!decoder_.DecodeInstruction(params.at("machine_code").get<uint64_t>(), bundle, error)) throw std::runtime_error(error);
        return encode(bundle);
    }

    json DecodeStream(const json& params) const {
        std::vector<uint32_t> words = params.at("words").get<std::vector<uint32_t>>();
        std::vector<amdisa::InstructionInfoBundle> decoded;
        std::string error;
        if (!decoder_.DecodeInstructionStream(words, decoded, error)) throw std::runtime_error(error);
        json result = json::array();
        for (const auto& item : decoded) result.push_back(encode(item));
        return result;
    }

    json DecodeDisassembly(const json& params) const {
        std::vector<amdisa::InstructionInfoBundle> decoded;
        std::string error;
        if (!decoder_.DecodeShaderDisassemblyText(params.at("text").get<std::string>(), decoded, error,
                                                  params.value("resolve_direct_branch_targets", false))) {
            throw std::runtime_error(error);
        }
        json result = json::array();
        for (const auto& item : decoded) result.push_back(encode(item));
        return result;
    }

    template <typename Value>
    const Value& Find(const std::map<std::string, Value>& values, const json& params) const {
        const auto name = params.at("name").get<std::string>();
        const auto found = values.find(name);
        if (found == values.end()) throw std::runtime_error("item not found: " + name);
        return found->second;
    }

    json GetExplorerInstruction(const json& params) const { return encode(Find(explorer_.GetInstructions(), params)); }
    json GetDataFormat(const json& params) const { return encode(Find(explorer_.GetDataFormats(), params)); }
    json GetOperandType(const json& params) const { return encode(Find(explorer_.GetOperandTypes(), params)); }
    json GetFunctionalGroup(const json& params) const { return encode(Find(explorer_.GetFunctionalGroups(), params)); }

    amdisa::IsaDecoder decoder_;
    amdisa::explorer::Spec explorer_;
    bool loaded_ = false;
};

json ErrorResponse(const json& id, const std::string& code, const std::string& message) {
    return {{"id", id}, {"ok", false}, {"error", {{"code", code}, {"message", message}}}};
}

}  // namespace

int main() {
    Bridge bridge;
    std::string line;
    while (std::getline(std::cin, line)) {
        json id = nullptr;
        try {
            if (line.size() > kMaxMessageBytes) {
                throw std::invalid_argument("request exceeds protocol size limit");
            }
            const auto request = json::parse(line);
            id = request.value("id", json(nullptr));
            std::cout << json{{"id", id}, {"ok", true}, {"result", bridge.Dispatch(request)}}.dump() << std::endl;
            if (request.value("method", "") == "shutdown") break;
        } catch (const nlohmann::json::exception& error) {
            std::cout << ErrorResponse(id, "invalid_request", error.what()).dump() << std::endl;
        } catch (const std::exception& error) {
            std::cout << ErrorResponse(id, "operation_failed", error.what()).dump() << std::endl;
        }
    }
}
