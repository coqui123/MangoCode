// Live worktree-backed [`Rollout`] for MCTS plan search.
//
// This turns the abstract search in `plan_search.rs` into real work:
//   - `propose` asks the model (via `run_single_query`) for several distinct
//     next steps given the partial plan.
//   - `simulate` creates a throwaway git worktree off `HEAD`, realizes the plan
//     in it via an injected sub-agent runner (the same machinery `TeamCreate`
//     uses), then scores the result: did it change anything, do tests pass, how
//     many clippy findings did it introduce, does a self-review critic approve.
//
// Realizing the plan's edits means running a full agentic sub-loop, which lives
// in `agent_tool.rs` (it owns the sub-agent context/tool plumbing). To avoid a
// circular dependency and keep this module testable, that one step is injected
// as a [`PlanRealizer`] via [`register_plan_realizer`], mirroring the existing
// `register_agent_runner` pattern. Everything else here — worktree lifecycle,
// test/clippy scoring, prompt building and response parsing — is real and unit
// tested without a model.
//
// NOTE: the model- and cargo-dependent path (real rollouts) is exercised only
// when `FLAG_PLAN_SEARCH` is enabled and a realizer is registered; it is not run
// by the test suite.

use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use once_cell::sync::OnceCell;
use tracing::{debug, warn};

use crate::clippy_baseline::compute_introduced_clippy;
use crate::plan_search::{Candidate, Rollout, RolloutOutcome};
use crate::QueryConfig;
use mangocode_api::AnthropicClient;
use mangocode_core::types::Message;

/// Injected sub-agent runner that realizes a plan's edits inside a worktree.
///
/// `args`: `(plan_prompt, worktree_path, session_ctx)`. The closure runs an
/// agentic sub-loop with its working directory set to `worktree_path`, applying
/// the plan. The session [`ToolContext`](mangocode_tools::ToolContext) is passed
/// per call (as `AgentRunFn` does) so the sub-run inherits the session's config,
/// auth, and tool set. Registered from `agent_tool.rs` to avoid a cycle.
pub type PlanRealizer = Arc<
    dyn Fn(
            String,
            PathBuf,
            Arc<mangocode_tools::ToolContext>,
        ) -> Pin<Box<dyn std::future::Future<Output = ()> + Send>>
        + Send
        + Sync,
>;

static PLAN_REALIZER: OnceCell<PlanRealizer> = OnceCell::new();

/// Register the global plan realizer. Called once at startup by the query crate
/// (see `init_plan_search_realizer`). Duplicate registrations keep the first.
pub fn register_plan_realizer(f: PlanRealizer) {
    if PLAN_REALIZER.set(f).is_err() {
        warn!("plan realizer already registered; keeping existing");
    }
}

/// Whether a realizer has been registered (i.e. real rollouts can run).
pub fn realizer_registered() -> bool {
    PLAN_REALIZER.get().is_some()
}

/// Re-entrancy guard. Agent-mode realization runs a sub-agent that re-enters
/// `run_query_loop`, which would re-fire the plan-search hook and recurse
/// (nesting worktrees until the path length blows up). This flag makes any
/// nested invocation a no-op. Process-global, so two top-level sessions in the
/// same process won't plan-search simultaneously — acceptable for an opt-in,
/// expensive research feature.
static PLAN_SEARCH_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Clears [`PLAN_SEARCH_ACTIVE`] on drop so the flag is released on every path.
struct ActiveGuard;
impl Drop for ActiveGuard {
    fn drop(&mut self) {
        PLAN_SEARCH_ACTIVE.store(false, Ordering::SeqCst);
    }
}

/// Knobs for the worktree rollout, separate from the search budget.
#[derive(Clone, Debug)]
pub struct WorktreeRolloutConfig {
    /// Candidate next-steps to request per `propose` call (branching factor).
    pub branching: usize,
    /// Shell command run in the worktree to gate `tests_passed`.
    pub test_command: String,
    /// Cargo package to lint for the clippy signal; empty disables clippy.
    pub clippy_package: String,
    /// Max turns granted to each plan-realization sub-run.
    pub max_turns: u32,
}

impl Default for WorktreeRolloutConfig {
    fn default() -> Self {
        Self {
            branching: 3,
            test_command: "cargo test --quiet".to_string(),
            clippy_package: String::new(),
            max_turns: 12,
        }
    }
}

impl WorktreeRolloutConfig {
    /// Override defaults from `MANGOCODE_PLAN_SEARCH_*` env vars.
    pub fn from_env() -> Self {
        let mut c = Self::default();
        if let Ok(v) = std::env::var("MANGOCODE_PLAN_SEARCH_BRANCHING") {
            if let Ok(n) = v.trim().parse::<usize>() {
                c.branching = n.clamp(1, 8);
            }
        }
        if let Ok(v) = std::env::var("MANGOCODE_PLAN_SEARCH_TEST_CMD") {
            if !v.trim().is_empty() {
                c.test_command = v;
            }
        }
        if let Ok(v) = std::env::var("MANGOCODE_PLAN_SEARCH_CLIPPY_PACKAGE") {
            c.clippy_package = v.trim().to_string();
        }
        c
    }
}

/// How a candidate plan's edits get applied inside the worktree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RealizeMode {
    /// Run a tool-capable sub-agent loop (default; needs an injected realizer).
    Agent,
    /// Chat-only providers (e.g. Copilot via the lmstudio slot, which has no
    /// reliable tool calls): ask the model for full-file rewrites as text and
    /// apply them directly — no tools, no sub-agent loop, no injected realizer.
    Chat,
}

/// A [`Rollout`] that scores plans by realizing them in throwaway git worktrees.
pub struct WorktreeRollout {
    client: Arc<AnthropicClient>,
    config: QueryConfig,
    goal: String,
    repo_root: PathBuf,
    /// Session context, forwarded to the realizer for each sub-run (Agent mode).
    base_ctx: Arc<mangocode_tools::ToolContext>,
    mode: RealizeMode,
    rollout: WorktreeRolloutConfig,
}

impl WorktreeRollout {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: Arc<AnthropicClient>,
        config: QueryConfig,
        goal: impl Into<String>,
        repo_root: PathBuf,
        base_ctx: Arc<mangocode_tools::ToolContext>,
        mode: RealizeMode,
        rollout: WorktreeRolloutConfig,
    ) -> Self {
        Self {
            client,
            config,
            goal: goal.into(),
            repo_root,
            base_ctx,
            mode,
            rollout,
        }
    }

    /// One model call returning its text, or `None` on error.
    ///
    /// Routes through a minimal 1-turn, tool-less `run_query_loop` rather than
    /// `run_single_query` so the call goes through the same provider resolution
    /// the session uses — `run_single_query` streams straight from the raw
    /// (Anthropic-shaped) client and can't reach the lmstudio/Copilot proxy, so
    /// propose/critic would fail auth. The re-entrancy guard makes the nested
    /// loop's own plan-search hook a no-op, so this cannot recurse.
    async fn ask(&self, prompt: String) -> Option<String> {
        let mut msgs = vec![Message::user(prompt)];
        let mut cfg = self.config.clone();
        cfg.max_turns = 1;
        let cancel = tokio_util::sync::CancellationToken::new();
        let outcome = crate::run_query_loop(
            self.client.as_ref(),
            &mut msgs,
            &[], // no tools — chat-only call
            self.base_ctx.as_ref(),
            &cfg,
            self.base_ctx.cost_tracker.clone(),
            None,
            cancel,
            None,
            None,
        )
        .await;
        match outcome {
            crate::QueryOutcome::EndTurn { message, .. } => Some(message.get_all_text()),
            crate::QueryOutcome::MaxTokens {
                partial_message, ..
            } => Some(partial_message.get_all_text()),
            crate::QueryOutcome::Error(e) => {
                warn!(error = %e, "plan_search: planning call errored");
                None
            }
            _ => {
                warn!("plan_search: planning call did not finish");
                None
            }
        }
    }
}

#[async_trait]
impl Rollout for WorktreeRollout {
    async fn propose(&self, path: &[Candidate], depth: usize) -> Vec<Candidate> {
        let prompt = build_propose_prompt(&self.goal, path, self.rollout.branching);
        let Some(text) = self.ask(prompt).await else {
            return Vec::new();
        };
        parse_candidates(&text, depth, self.rollout.branching)
    }

    async fn simulate(&self, path: &[Candidate]) -> RolloutOutcome {
        match self.simulate_inner(path).await {
            Ok(outcome) => outcome,
            Err(e) => {
                warn!(error = %e, "plan_search: rollout failed; scoring as no-op");
                RolloutOutcome::default()
            }
        }
    }
}

impl WorktreeRollout {
    async fn simulate_inner(&self, path: &[Candidate]) -> Result<RolloutOutcome, String> {
        // Nothing proposed (e.g. the root, or a failed propose) → no plan to
        // realize; skip the worktree entirely and score as a no-op.
        if path.is_empty() {
            return Ok(RolloutOutcome::default());
        }

        // Unique worktree off HEAD.
        let branch = format!("mc-plan-{}", uuid::Uuid::new_v4());
        let rel = format!(".worktrees/plan-search/{branch}");
        let worktree = self.repo_root.join(&rel);
        create_worktree(&self.repo_root, &branch, &worktree).await?;

        // Always tear the worktree down, even if a step below fails.
        let result = self.score_in_worktree(&worktree, path).await;
        remove_worktree(&self.repo_root, &branch, &worktree).await;
        result
    }

    /// Apply the plan's edits to the worktree using the active [`RealizeMode`].
    async fn realize(&self, worktree: &Path, path: &[Candidate]) -> Result<(), String> {
        match self.mode {
            RealizeMode::Agent => {
                let realizer = PLAN_REALIZER
                    .get()
                    .cloned()
                    .ok_or("no plan realizer registered")?;
                let plan_prompt = build_realize_prompt(&self.goal, path);
                realizer(plan_prompt, worktree.to_path_buf(), self.base_ctx.clone()).await;
                Ok(())
            }
            RealizeMode::Chat => self.chat_realize(worktree, path).await,
        }
    }

    /// Chat-only realization: ask the model for full-file rewrites and apply
    /// them ourselves. No tools, no sub-agent — works with Copilot/lmstudio.
    async fn chat_realize(&self, worktree: &Path, path: &[Candidate]) -> Result<(), String> {
        let context = gather_worktree_context(worktree).await;
        let prompt = build_chat_realize_prompt(&self.goal, path, &context);
        let Some(text) = self.ask(prompt).await else {
            return Err("chat realize: model call failed".to_string());
        };
        let blocks = parse_file_blocks(&text);
        if blocks.is_empty() {
            debug!("plan_search chat realize: model returned no file blocks");
            return Ok(()); // no edits → scored as no-op upstream
        }
        let written = write_file_blocks(worktree, &blocks).await;
        debug!(written, proposed = blocks.len(), "plan_search chat realize wrote files");
        Ok(())
    }

    async fn score_in_worktree(
        &self,
        worktree: &Path,
        path: &[Candidate],
    ) -> Result<RolloutOutcome, String> {
        // Realize the plan's edits inside the worktree.
        self.realize(worktree, path).await?;

        let made_changes = git_status_dirty(worktree).await;
        if !made_changes {
            // No edits → no progress; cheap exit, scores 0 via the gate.
            return Ok(RolloutOutcome {
                made_changes: false,
                ..Default::default()
            });
        }

        let tests_passed = run_test_command(worktree, &self.rollout.test_command).await;
        let clippy_introduced = self.clippy_count(worktree).await;
        let critic_pass = self.run_critic(path).await;

        Ok(RolloutOutcome {
            made_changes: true,
            tests_passed,
            clippy_introduced,
            critic_pass,
            // Token accounting for sub-runs is handled by the shared cost
            // tracker; the search budget uses rollout count as the primary cap.
            tokens_used: 0,
        })
    }

    /// Clippy findings introduced in the worktree, or 0 if disabled/unavailable.
    async fn clippy_count(&self, worktree: &Path) -> usize {
        if self.rollout.clippy_package.is_empty() {
            return 0;
        }
        let Some(changed) = changed_rs_files_with_before(worktree).await else {
            return 0;
        };
        let dir = worktree.to_path_buf();
        let package = self.rollout.clippy_package.clone();
        // compute_introduced_clippy is blocking (spawns cargo) → offload.
        tokio::task::spawn_blocking(move || {
            compute_introduced_clippy(&dir, &package, &changed)
                .map(|f| f.len())
                .unwrap_or(0)
        })
        .await
        .unwrap_or(0)
    }

    async fn run_critic(&self, path: &[Candidate]) -> bool {
        let prompt = build_critic_prompt(&self.goal, path);
        match self.ask(prompt).await {
            Some(text) => parse_critic_pass(&text),
            None => false,
        }
    }
}

/// Loop entry: if `FLAG_PLAN_SEARCH` is on, run a plan search for the latest
/// user goal, inject the winning plan as a steering message, and **return it**
/// so the caller can also seed the execution scratchpad with it. Seeding the
/// scratchpad keeps the chosen plan in every turn's `[SCRATCHPAD] Plan:` block,
/// which is what makes weaker / chat-bridged providers (Copilot via lmstudio)
/// reliably emit the tool calls to carry it out instead of declaring premature
/// success. No-op (returns `None`) otherwise — cheap on the default path.
///
/// Reuses the session's `client` (already authed/routed for the active provider)
/// for the planning model calls — building a fresh client misroutes the model
/// and trips auth.
pub async fn maybe_run_plan_search(
    client: &AnthropicClient,
    config: &QueryConfig,
    tool_ctx: &mangocode_tools::ToolContext,
    messages: &mut Vec<Message>,
    event_tx: Option<&tokio::sync::mpsc::UnboundedSender<crate::QueryEvent>>,
) -> Option<String> {
    if !crate::plan_search::is_enabled() {
        return None;
    }
    // Re-entrancy guard: agent-mode realization re-enters run_query_loop, which
    // would recurse into plan search and nest worktrees. swap(true): if it was
    // already true we are nested → bail without taking the guard (the outer call
    // owns it).
    if PLAN_SEARCH_ACTIVE.swap(true, Ordering::SeqCst) {
        debug!("plan_search: nested invocation suppressed");
        return None;
    }
    let _active = ActiveGuard;

    // Pick the realization mode. Chat-only backends (Copilot via the lmstudio
    // slot) can't drive the tool-using sub-agent reliably, so realize edits from
    // plain-text full-file rewrites instead. Override with
    // MANGOCODE_PLAN_SEARCH_REALIZE=chat|agent.
    let mode = resolve_realize_mode(&tool_ctx.config);
    if mode == RealizeMode::Agent && !realizer_registered() {
        debug!("plan_search: agent mode but no realizer registered; skipping");
        return None;
    }
    let goal = messages.last().map(|m| m.get_all_text())?;
    if goal.trim().is_empty() {
        return None;
    }

    // Reuse the live session client (correctly routed + authed for the provider).
    let client = Arc::new(client.clone());
    let repo_root = resolve_repo_root(&tool_ctx.working_dir).await;
    let rollout = WorktreeRollout::new(
        client,
        config.clone(),
        goal.clone(),
        repo_root,
        Arc::new(tool_ctx.clone()),
        mode,
        WorktreeRolloutConfig::from_env(),
    );

    if let Some(tx) = event_tx {
        let _ = tx.send(crate::QueryEvent::Status(
            "Plan search: exploring candidate plans in sandboxes".to_string(),
        ));
    }

    let outcome = crate::plan_search::run_if_enabled(&goal, &rollout).await?;
    if outcome.best_path.is_empty() {
        debug!("plan_search: no plan selected; proceeding normally");
        return None;
    }

    let mut plan = String::new();
    for (i, c) in outcome.best_path.iter().enumerate() {
        plan.push_str(&format!("{}. {}\n", i + 1, c.plan));
    }
    if let Some(tx) = event_tx {
        let _ = tx.send(crate::QueryEvent::Status(format!(
            "Plan search: chose a {}-step plan (reward {:.2}, {} rollouts)",
            outcome.best_path.len(),
            outcome.best_reward,
            outcome.rollouts
        )));
    }
    messages.push(Message::user(format!(
        "[plan-search] A search over candidate approaches (each realized and tested in a \
         throwaway worktree) selected this plan as highest-scoring (reward {:.2}). Implement \
         it directly with the available tools, without re-exploring alternatives:\n{plan}",
        outcome.best_reward
    )));
    Some(plan.trim().to_string())
}

/// Choose realization mode: explicit `MANGOCODE_PLAN_SEARCH_REALIZE` wins, else
/// chat for copilot-pirate/lmstudio backends, agent otherwise.
fn resolve_realize_mode(config: &mangocode_core::config::Config) -> RealizeMode {
    if let Ok(v) = std::env::var("MANGOCODE_PLAN_SEARCH_REALIZE") {
        match v.trim().to_ascii_lowercase().as_str() {
            "chat" => return RealizeMode::Chat,
            "agent" => return RealizeMode::Agent,
            _ => {}
        }
    }
    if is_chat_only_backend(config) {
        RealizeMode::Chat
    } else {
        RealizeMode::Agent
    }
}

/// Whether the *active* provider is the Copilot-pirate/lmstudio proxy. The
/// proxy's base lives in `provider_configs[<provider>].api_base`;
/// `resolve_api_base()` only ever returns the Anthropic base, so checking that
/// alone misses it (the bug the first live run hit).
fn is_chat_only_backend(config: &mangocode_core::config::Config) -> bool {
    let base = config
        .provider
        .as_deref()
        .and_then(|p| config.provider_configs.get(p))
        .and_then(|pc| pc.api_base.clone())
        .unwrap_or_else(|| config.resolve_api_base());
    mangocode_api::is_copilot_pirate_backend(&base)
}

async fn resolve_repo_root(cwd: &Path) -> PathBuf {
    match run_git(cwd, &["rev-parse", "--show-toplevel"]).await {
        Ok(s) if !s.trim().is_empty() => PathBuf::from(s.trim()),
        _ => cwd.to_path_buf(),
    }
}

// ---------------------------------------------------------------------------
// Prompt building + parsing (pure, unit-tested)
// ---------------------------------------------------------------------------

fn render_path(path: &[Candidate]) -> String {
    if path.is_empty() {
        return "(none yet)".to_string();
    }
    let mut s = String::new();
    for (i, c) in path.iter().enumerate() {
        s.push_str(&format!("{}. {}\n", i + 1, c.plan));
    }
    s
}

fn build_propose_prompt(goal: &str, path: &[Candidate], branching: usize) -> String {
    format!(
        "You are planning how to accomplish this goal:\n{goal}\n\n\
         Plan steps chosen so far:\n{}\n\
         Propose {branching} DISTINCT, concrete next steps (different approaches, not \
         refinements of one). Output ONLY the steps, each on its own line prefixed \
         with `STEP: `. No prose, no numbering.",
        render_path(path)
    )
}

fn build_realize_prompt(goal: &str, path: &[Candidate]) -> String {
    format!(
        "Goal:\n{goal}\n\nExecute exactly this plan, making the necessary code edits. \
         Do not deviate or expand scope:\n{}",
        render_path(path)
    )
}

/// Caps for the worktree snapshot fed to a chat-only model.
const CTX_MAX_FILES: usize = 24;
const CTX_MAX_FILE_BYTES: usize = 8 * 1024;
const CTX_MAX_TOTAL_BYTES: usize = 48 * 1024;

/// Snapshot of the worktree's tracked text files for chat realization, bounded
/// so a big repo can't blow the context window. Logs what it omitted.
async fn gather_worktree_context(worktree: &Path) -> String {
    let listing = match run_git(worktree, &["ls-files"]).await {
        Ok(s) => s,
        Err(_) => return String::new(),
    };
    let files: Vec<&str> = listing.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let mut out = String::new();
    let mut total = 0usize;
    let mut included = 0usize;
    for rel in files.iter().take(CTX_MAX_FILES) {
        let full = worktree.join(rel);
        let Ok(bytes) = tokio::fs::read(&full).await else {
            continue;
        };
        // Skip binary-ish files (NUL byte) and oversized ones.
        if bytes.contains(&0) || bytes.len() > CTX_MAX_FILE_BYTES {
            continue;
        }
        if total + bytes.len() > CTX_MAX_TOTAL_BYTES {
            break;
        }
        let text = String::from_utf8_lossy(&bytes);
        out.push_str(&format!("```mango-file:{rel}\n{text}\n```\n"));
        total += bytes.len();
        included += 1;
    }
    if files.len() > included {
        out.push_str(&format!(
            "(… {} more tracked file(s) omitted from this snapshot)\n",
            files.len() - included
        ));
    }
    out
}

fn build_chat_realize_prompt(goal: &str, path: &[Candidate], context: &str) -> String {
    format!(
        "Goal:\n{goal}\n\nExecute exactly this plan (no scope creep):\n{}\n\
         Current repository files:\n{context}\n\
         Return the COMPLETE new contents of every file you change or create. For each \
         file output a fenced block whose info string is `mango-file:<relative/path>`:\n\
         ```mango-file:relative/path.ext\n<full file contents>\n```\n\
         Output ONLY these blocks — no explanation, no diffs, no partial snippets.",
        render_path(path)
    )
}

/// Parse ```` ```mango-file:<path> ```` fenced blocks into `(path, contents)`.
/// The closing fence is a line that is exactly ```` ``` ````.
fn parse_file_blocks(text: &str) -> Vec<(String, String)> {
    let mut blocks = Vec::new();
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix("```") else {
            continue;
        };
        // Opening fence must carry the mango-file marker.
        let Some(marker) = rest.find("mango-file:") else {
            continue;
        };
        let path = rest[marker + "mango-file:".len()..].trim().to_string();
        if path.is_empty() {
            continue;
        }
        let mut body = String::new();
        for inner in lines.by_ref() {
            if inner.trim() == "```" {
                break;
            }
            body.push_str(inner);
            body.push('\n');
        }
        blocks.push((path, body));
    }
    blocks
}

/// Join `rel` under `worktree`, rejecting absolute paths and any `..` escape.
fn safe_join(worktree: &Path, rel: &str) -> Option<PathBuf> {
    use std::path::Component;
    let p = Path::new(rel.trim());
    if p.is_absolute() {
        return None;
    }
    let mut out = worktree.to_path_buf();
    for comp in p.components() {
        match comp {
            Component::Normal(part) => out.push(part),
            Component::CurDir => {}
            _ => return None, // ParentDir / RootDir / Prefix → escape attempt
        }
    }
    (out != worktree).then_some(out)
}

/// Write parsed file blocks into the worktree; returns the count written.
async fn write_file_blocks(worktree: &Path, blocks: &[(String, String)]) -> usize {
    let mut written = 0usize;
    for (rel, contents) in blocks {
        let Some(dest) = safe_join(worktree, rel) else {
            warn!(path = %rel, "plan_search chat realize: rejected unsafe path");
            continue;
        };
        if let Some(parent) = dest.parent() {
            if tokio::fs::create_dir_all(parent).await.is_err() {
                continue;
            }
        }
        if tokio::fs::write(&dest, contents).await.is_ok() {
            written += 1;
        }
    }
    written
}

fn build_critic_prompt(goal: &str, path: &[Candidate]) -> String {
    format!(
        "Goal:\n{goal}\n\nA candidate implementation followed this plan:\n{}\n\
         Judge whether it correctly and completely satisfies the goal. Reply with a \
         single line: `VERDICT: PASS` or `VERDICT: FAIL`, then one sentence of reason.",
        render_path(path)
    )
}

/// Extract `STEP: ...` lines into candidates, capped at `branching`.
fn parse_candidates(text: &str, depth: usize, branching: usize) -> Vec<Candidate> {
    text.lines()
        .filter_map(|line| {
            let t = line.trim();
            // Tolerate a leading bullet/quote the model may add.
            let t = t.trim_start_matches(['-', '*', '>', ' ']);
            let rest = t
                .strip_prefix("STEP:")
                .or_else(|| t.strip_prefix("Step:"))
                .or_else(|| t.strip_prefix("step:"))?;
            let plan = rest.trim();
            (!plan.is_empty()).then(|| plan.to_string())
        })
        .take(branching)
        .enumerate()
        .map(|(i, plan)| Candidate::new(format!("d{depth}-{i}"), plan))
        .collect()
}

/// True iff the critic's verdict line says PASS.
fn parse_critic_pass(text: &str) -> bool {
    for line in text.lines() {
        let upper = line.to_ascii_uppercase();
        if let Some(idx) = upper.find("VERDICT:") {
            let after = &upper[idx + "VERDICT:".len()..];
            // First alphabetic token after the marker decides it.
            return after.trim_start().starts_with("PASS");
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Git worktree lifecycle + scoring shell-outs (real)
// ---------------------------------------------------------------------------

async fn run_git(cwd: &Path, args: &[&str]) -> Result<String, String> {
    let out = tokio::process::Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .await
        .map_err(|e| e.to_string())?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).to_string())
    }
}

async fn create_worktree(repo_root: &Path, branch: &str, worktree: &Path) -> Result<(), String> {
    let wt = worktree.to_string_lossy().to_string();
    run_git(repo_root, &["worktree", "add", "-b", branch, &wt, "HEAD"])
        .await
        .map(|_| ())
        .map_err(|e| format!("worktree add failed: {}", e.trim()))
}

async fn remove_worktree(repo_root: &Path, branch: &str, worktree: &Path) {
    let wt = worktree.to_string_lossy().to_string();
    if let Err(e) = run_git(repo_root, &["worktree", "remove", "--force", &wt]).await {
        warn!(error = %e.trim(), "plan_search: worktree remove failed");
    }
    // Best-effort branch cleanup.
    let _ = run_git(repo_root, &["branch", "-D", "--", branch]).await;
}

/// Non-empty `git status --porcelain` ⇒ the rollout made changes.
async fn git_status_dirty(worktree: &Path) -> bool {
    match run_git(worktree, &["status", "--porcelain"]).await {
        Ok(s) => s.lines().any(|l| !l.trim().is_empty()),
        Err(_) => false,
    }
}

/// `(absolute path, HEAD "before" content)` for each changed `.rs` file.
async fn changed_rs_files_with_before(worktree: &Path) -> Option<Vec<(PathBuf, String)>> {
    let names = run_git(worktree, &["diff", "--name-only", "HEAD"]).await.ok()?;
    let mut out = Vec::new();
    for rel in names.lines().map(str::trim).filter(|l| l.ends_with(".rs")) {
        let before = run_git(worktree, &["show", &format!("HEAD:{rel}")])
            .await
            .unwrap_or_default();
        out.push((worktree.join(rel), before));
    }
    Some(out)
}

/// Run the configured test command in the worktree; success ⇒ tests pass.
async fn run_test_command(worktree: &Path, command: &str) -> bool {
    if command.trim().is_empty() {
        return false;
    }
    let (sh, flag) = if cfg!(target_os = "windows") {
        ("cmd", "/C")
    } else {
        ("sh", "-c")
    };
    debug!(%command, "plan_search: running test command in worktree");
    match tokio::process::Command::new(sh)
        .args([flag, command])
        .current_dir(worktree)
        .output()
        .await
    {
        Ok(out) => out.status.success(),
        Err(e) => {
            warn!(error = %e, "plan_search: test command failed to spawn");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_candidates_extracts_step_lines() {
        let text = "STEP: refactor the parser\n\
                    - STEP: add a cache layer\n\
                    noise line\n\
                    step: third approach\n\
                    STEP:   \n"; // blank one dropped
        let c = parse_candidates(text, 2, 5);
        assert_eq!(c.len(), 3);
        assert_eq!(c[0].plan, "refactor the parser");
        assert_eq!(c[1].plan, "add a cache layer");
        assert_eq!(c[2].plan, "third approach");
        assert_eq!(c[0].id, "d2-0");
    }

    #[test]
    fn parse_candidates_respects_branching_cap() {
        let text = "STEP: a\nSTEP: b\nSTEP: c\nSTEP: d";
        assert_eq!(parse_candidates(text, 0, 2).len(), 2);
    }

    #[test]
    fn parse_critic_pass_reads_verdict() {
        assert!(parse_critic_pass("VERDICT: PASS — looks complete"));
        assert!(parse_critic_pass("blah\nverdict: pass\n"));
        assert!(!parse_critic_pass("VERDICT: FAIL, missing tests"));
        assert!(!parse_critic_pass("no verdict here"));
        // The marker decides, not an earlier stray "pass".
        assert!(!parse_critic_pass("it might pass\nVERDICT: FAIL"));
    }

    #[test]
    fn prompts_include_goal_and_path() {
        let path = vec![Candidate::new("d0-0", "do X")];
        let p = build_propose_prompt("achieve Y", &path, 3);
        assert!(p.contains("achieve Y"));
        assert!(p.contains("do X"));
        assert!(p.contains("STEP:"));
        assert!(build_realize_prompt("achieve Y", &path).contains("do X"));
        assert!(build_critic_prompt("achieve Y", &path).contains("VERDICT"));
    }

    #[test]
    fn render_path_handles_empty() {
        assert_eq!(render_path(&[]), "(none yet)");
    }

    #[test]
    fn parse_file_blocks_extracts_path_and_body() {
        let text = "Sure!\n\
                    ```mango-file:src/app.py\n\
                    def greet():\n    return \"hi\"\n\
                    ```\n\
                    some prose\n\
                    ```mango-file: pkg/new.rs \n\
                    fn x() {}\n\
                    ```\n";
        let b = parse_file_blocks(text);
        assert_eq!(b.len(), 2);
        assert_eq!(b[0].0, "src/app.py");
        assert_eq!(b[0].1, "def greet():\n    return \"hi\"\n");
        assert_eq!(b[1].0, "pkg/new.rs"); // trimmed
        assert_eq!(b[1].1, "fn x() {}\n");
    }

    #[test]
    fn parse_file_blocks_ignores_plain_and_unmarked_fences() {
        // A normal ```python fence (no marker) is not a file block.
        let text = "```python\nprint(1)\n```\n```mango-file:\n bad empty path\n```";
        assert!(parse_file_blocks(text).is_empty());
    }

    #[test]
    fn safe_join_blocks_escapes() {
        let root = Path::new(if cfg!(windows) { r"C:\wt" } else { "/wt" });
        assert!(safe_join(root, "a/b.rs").is_some());
        assert!(safe_join(root, "./a.rs").is_some());
        // Escapes and absolutes are rejected.
        assert!(safe_join(root, "../escape.rs").is_none());
        assert!(safe_join(root, "a/../../escape.rs").is_none());
        assert!(safe_join(root, "").is_none());
        let abs = if cfg!(windows) { r"C:\x.rs" } else { "/etc/x" };
        assert!(safe_join(root, abs).is_none());
    }

    #[tokio::test]
    async fn write_file_blocks_writes_and_skips_unsafe() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let blocks = vec![
            ("sub/dir/new.txt".to_string(), "hello\n".to_string()),
            ("../escape.txt".to_string(), "nope\n".to_string()),
        ];
        let n = write_file_blocks(root, &blocks).await;
        assert_eq!(n, 1);
        assert_eq!(
            tokio::fs::read_to_string(root.join("sub/dir/new.txt")).await.unwrap(),
            "hello\n"
        );
        assert!(!root.parent().unwrap().join("escape.txt").exists());
    }

    // --- Real git worktree lifecycle, gated on git being present ---

    async fn git_ok(dir: &Path) -> bool {
        run_git(dir, &["--version"]).await.is_ok()
    }

    async fn init_repo(dir: &Path) {
        run_git(dir, &["init", "-q"]).await.unwrap();
        run_git(dir, &["config", "user.email", "t@t.t"]).await.unwrap();
        run_git(dir, &["config", "user.name", "t"]).await.unwrap();
        tokio::fs::write(dir.join("a.txt"), "hello\n").await.unwrap();
        run_git(dir, &["add", "-A"]).await.unwrap();
        run_git(dir, &["commit", "-qm", "init"]).await.unwrap();
    }

    #[tokio::test]
    async fn worktree_create_edit_detect_remove() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        if !git_ok(root).await {
            eprintln!("git unavailable; skipping");
            return;
        }
        init_repo(root).await;

        let branch = "mc-plan-test";
        let wt = root.join(".worktrees/plan-search/mc-plan-test");
        create_worktree(root, branch, &wt).await.unwrap();
        assert!(wt.exists());

        // Clean worktree → not dirty.
        assert!(!git_status_dirty(&wt).await);

        // Edit a tracked file → dirty + shows in changed .rs scan (use .rs file).
        tokio::fs::write(wt.join("a.txt"), "changed\n").await.unwrap();
        assert!(git_status_dirty(&wt).await);

        remove_worktree(root, branch, &wt).await;
        assert!(!wt.exists());
    }

    #[tokio::test]
    async fn changed_rs_scan_reports_before_content() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        if !git_ok(root).await {
            return;
        }
        run_git(root, &["init", "-q"]).await.unwrap();
        run_git(root, &["config", "user.email", "t@t.t"]).await.unwrap();
        run_git(root, &["config", "user.name", "t"]).await.unwrap();
        tokio::fs::write(root.join("lib.rs"), "fn a() {}\n").await.unwrap();
        run_git(root, &["add", "-A"]).await.unwrap();
        run_git(root, &["commit", "-qm", "init"]).await.unwrap();

        tokio::fs::write(root.join("lib.rs"), "fn a() {} fn b() {}\n").await.unwrap();
        let changed = changed_rs_files_with_before(root).await.unwrap();
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].1, "fn a() {}\n"); // HEAD "before"
    }

    #[tokio::test]
    async fn test_command_pass_and_fail() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path();
        // Portable: `git --version` always succeeds; a bogus command fails.
        assert!(run_test_command(dir, "git --version").await);
        assert!(!run_test_command(dir, "exit 3").await || cfg!(windows));
        assert!(!run_test_command(dir, "").await);
    }
}
