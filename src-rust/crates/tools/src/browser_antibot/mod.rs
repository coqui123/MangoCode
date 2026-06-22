//! Browser hardening primitives; local-first, no remote solvers.

pub mod chrome_args;
pub mod expert_script;
pub mod localhost_devtools;
pub mod match_template;
pub mod png_template;
pub mod socks_stub;

#[inline]
pub fn browser_expert_env_enabled() -> bool {
    std::env::var("MANGOCODE_BROWSER_EXPERT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}
