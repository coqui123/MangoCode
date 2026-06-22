// AgentTool is defined in the query crate to avoid a circular dependency:
// tools → query → tools would be circular.
//
// The AgentTool implementation lives in crates/query/src/agent_tool.rs and is
// re-exported as `mangocode_query::AgentTool`.
//
// This file exists only as a placeholder to keep the directory tidy.
