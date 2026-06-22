// MCTS plan search — research-grade, frontier-tilted.
//
// Instead of committing to the first plan it thinks of, the agent grows a tree
// of candidate plans, scores each by running it out in a sandbox (tests +
// clippy + a self-review critic), and applies the highest-value branch. This is
// Monte Carlo Tree Search (the AlphaGo family) adapted to agent planning, in the
// spirit of Tree-of-Thoughts / LATS (Language Agent Tree Search).
//
// The four-step loop, run under a budget:
//   1. Select     — descend from the root by UCB1 (exploit good-so-far vs
//                   explore the under-visited).
//   2. Expand     — add one untried candidate action as a new child.
//   3. Simulate   — roll the partial plan out in a sandbox and score it.
//   4. Backprop   — push the reward up the visited path.
//
// This module owns the *algorithm* and the *scoring*. Generating candidate
// plans and realizing their edits is delegated to a [`Rollout`] implementor so
// the search is unit-testable without an LLM or a worktree. The production
// implementor (worktree sandbox + model-driven edits + `cargo test`/clippy +
// critic) plugs into the same trait; see the integration notes on [`Rollout`].
//
// Off by default and expensive (5–50x tokens). Gated behind
// `FLAG_PLAN_SEARCH`; see [`is_enabled`].

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::clippy_diff::ClippyFinding;

/// Default exploration constant for UCB1 (`sqrt(2)`), the textbook value.
const DEFAULT_EXPLORATION: f64 = std::f64::consts::SQRT_2;

/// Returns whether MCTS plan search is enabled for this session.
///
/// Off by default; opt in with `MANGOCODE_FLAG_PLAN_SEARCH=1` or via flags.json.
pub fn is_enabled() -> bool {
    mangocode_core::FeatureFlags::is_enabled(mangocode_core::FLAG_PLAN_SEARCH)
}

/// A single candidate action in the plan tree: one proposed next step, plus a
/// stable id used for logging and de-duplication.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Candidate {
    /// Stable identifier (e.g. a short slug or hash of `plan`).
    pub id: String,
    /// The proposed plan / next action, in natural language.
    pub plan: String,
}

impl Candidate {
    pub fn new(id: impl Into<String>, plan: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            plan: plan.into(),
        }
    }
}

/// Relative weights for the three rollout signals. Need not sum to 1 — the
/// reward is normalized by their total.
#[derive(Clone, Copy, Debug)]
pub struct RewardWeights {
    pub tests: f64,
    pub clippy: f64,
    pub critic: f64,
}

impl Default for RewardWeights {
    fn default() -> Self {
        // Tests are the strongest signal; the critic catches what tests miss;
        // clippy is a tie-breaker on code quality.
        Self {
            tests: 0.5,
            critic: 0.3,
            clippy: 0.2,
        }
    }
}

/// The measured outcome of rolling one partial plan out in a sandbox.
#[derive(Clone, Debug, Default)]
pub struct RolloutOutcome {
    /// Did the rollout actually change anything (non-empty worktree diff)?
    /// A rollout that made no edits did not address the goal and scores `0`,
    /// regardless of the other signals — otherwise a do-nothing branch coasts
    /// on the pre-existing green test suite and an empty lint delta.
    pub made_changes: bool,
    /// Did the configured test command pass in the sandbox?
    pub tests_passed: bool,
    /// Count of *newly introduced* clippy findings vs the baseline
    /// (see [`crate::clippy_diff::introduced_findings`]). Zero is best.
    pub clippy_introduced: usize,
    /// Did the self-review critic approve the change?
    pub critic_pass: bool,
    /// Output tokens this rollout consumed, for the budget ceiling.
    pub tokens_used: u64,
}

impl RolloutOutcome {
    /// Scalar reward in `[0, 1]`, higher is better.
    pub fn reward(&self, w: &RewardWeights) -> f64 {
        combine_signals(
            self.made_changes,
            self.tests_passed,
            self.clippy_introduced,
            self.critic_pass,
            w,
        )
    }

    /// Convenience constructor from a slice of introduced clippy findings.
    pub fn from_signals(
        made_changes: bool,
        tests_passed: bool,
        introduced: &[ClippyFinding],
        critic_pass: bool,
        tokens_used: u64,
    ) -> Self {
        Self {
            made_changes,
            tests_passed,
            clippy_introduced: introduced.len(),
            critic_pass,
            tokens_used,
        }
    }
}

/// Pure scoring core: fold the rollout signals into a normalized `[0, 1]` reward.
///
/// Two gates keep the score honest:
/// - A rollout that made **no changes** scores `0` — it did not address the
///   goal, so it must not coast on the pre-existing passing tests.
/// - The clippy quality term only earns credit when **tests pass**; a rollout
///   that breaks the build cannot buy reward back by being lint-clean.
///
/// Among test-passing rollouts the clippy term decays smoothly with the number
/// of introduced findings (`1 / (1 + n)`), so a single regression does not
/// dominate a passing, critic-approved change.
pub fn combine_signals(
    made_changes: bool,
    tests_passed: bool,
    clippy_introduced: usize,
    critic_pass: bool,
    w: &RewardWeights,
) -> f64 {
    let total = w.tests + w.clippy + w.critic;
    if total <= 0.0 || !made_changes {
        return 0.0;
    }
    let tests = if tests_passed { 1.0 } else { 0.0 };
    // No quality credit for a broken build.
    let clippy = if tests_passed {
        1.0 / (1.0 + clippy_introduced as f64)
    } else {
        0.0
    };
    let critic = if critic_pass { 1.0 } else { 0.0 };
    (w.tests * tests + w.clippy * clippy + w.critic * critic) / total
}

/// Budget and exploration knobs for a search. Build with [`PlanSearchConfig::from_env`].
#[derive(Clone, Debug)]
pub struct PlanSearchConfig {
    /// Hard cap on rollouts (simulations). The dominant cost control.
    pub max_rollouts: u32,
    /// Maximum plan-tree depth (length of a plan chain).
    pub max_depth: usize,
    /// UCB1 exploration constant `c`. Higher explores more.
    pub exploration: f64,
    /// Stop early once cumulative rollout tokens reach this. `0` = unlimited.
    pub token_ceiling: u64,
    /// Reward signal weights.
    pub weights: RewardWeights,
}

impl Default for PlanSearchConfig {
    fn default() -> Self {
        Self {
            max_rollouts: 8,
            max_depth: 3,
            exploration: DEFAULT_EXPLORATION,
            token_ceiling: 0,
            weights: RewardWeights::default(),
        }
    }
}

impl PlanSearchConfig {
    /// Load from `MANGOCODE_PLAN_SEARCH_*` env vars, falling back to defaults.
    /// Values are clamped to sane ranges via [`PlanSearchConfig::clamped`].
    pub fn from_env() -> Self {
        let mut c = Self::default();
        if let Some(v) = env_u32("MANGOCODE_PLAN_SEARCH_MAX_ROLLOUTS") {
            c.max_rollouts = v;
        }
        if let Some(v) = env_u32("MANGOCODE_PLAN_SEARCH_MAX_DEPTH") {
            c.max_depth = v as usize;
        }
        if let Some(v) = env_f64("MANGOCODE_PLAN_SEARCH_EXPLORATION") {
            c.exploration = v;
        }
        if let Some(v) = env_u64("MANGOCODE_PLAN_SEARCH_TOKEN_CEILING") {
            c.token_ceiling = v;
        }
        c.clamped()
    }

    /// Clamp to ranges that keep the search bounded and well-defined.
    ///
    /// Also sanitizes the reward weights: any negative or non-finite weight is
    /// forced to `0.0`, so the `[0, 1]` reward invariant that UCB1 relies on
    /// cannot be broken by a caller-supplied [`RewardWeights`] (its fields are
    /// public). If that zeroes every weight, [`combine_signals`] still returns a
    /// safe `0.0` via its `total <= 0.0` guard.
    pub fn clamped(mut self) -> Self {
        self.max_rollouts = self.max_rollouts.clamp(1, 1000);
        self.max_depth = self.max_depth.clamp(1, 32);
        // Bounded, finite, non-negative exploration constant.
        if !self.exploration.is_finite() || self.exploration < 0.0 {
            self.exploration = DEFAULT_EXPLORATION;
        }
        self.exploration = self.exploration.min(1000.0);
        for w in [
            &mut self.weights.tests,
            &mut self.weights.clippy,
            &mut self.weights.critic,
        ] {
            if !w.is_finite() || *w < 0.0 {
                *w = 0.0;
            }
        }
        self
    }
}

/// Pluggable plan generator + evaluator. The MCTS driver only ever calls these
/// two methods, which keeps the algorithm testable with a deterministic mock.
///
/// # Production integration
/// The real implementor backs onto MangoCode's existing machinery:
/// - [`propose`](Rollout::propose) asks the model for several distinct next
///   steps given the partial plan path (one model call).
/// - [`simulate`](Rollout::simulate) realizes the plan in an isolated
///   `crates/tools/src/worktree.rs` sandbox, runs the configured test command
///   and clippy ([`crate::clippy_baseline`]), runs the self-review critic, and
///   returns the measured [`RolloutOutcome`].
///
/// `Send + Sync` (matching the crate's `Tool` trait) so `run_plan_search` stays
/// spawnable on the multithreaded runtime the agent loop uses.
#[async_trait]
pub trait Rollout: Send + Sync {
    /// Propose candidate next actions from the current partial plan `path`
    /// (root-first; empty at the root). Returning an empty vec marks the node
    /// terminal. Implementors should cap the branching factor themselves.
    async fn propose(&self, path: &[Candidate], depth: usize) -> Vec<Candidate>;

    /// Roll `path` out in a sandbox and score the result.
    async fn simulate(&self, path: &[Candidate]) -> RolloutOutcome;
}

/// A [`Rollout`] built from two async closures, so the agent loop can supply the
/// real `propose`/`simulate` logic (model call + worktree sandbox) at the call
/// site without this crate hard-coding that plumbing.
///
/// The closures receive an **owned** `Vec<Candidate>` (a clone of the path) so
/// the returned futures borrow nothing from the tree.
///
/// ```ignore
/// // At the call site, `client`/`config` come from the agent loop:
/// let rollout = CallbackRollout::new(
///     |path, depth| async move { propose_plans(&client, &config, &path, depth).await },
///     |path|        async move { sandbox_eval(&client, &config, &path).await },
/// );
/// if let Some(outcome) = run_if_enabled("implement X", &rollout).await {
///     // feed outcome.best_path into the turn as the chosen plan
/// }
/// ```
pub struct CallbackRollout<P, S> {
    propose: P,
    simulate: S,
}

impl<P, S> CallbackRollout<P, S> {
    pub fn new(propose: P, simulate: S) -> Self {
        Self { propose, simulate }
    }
}

#[async_trait]
impl<P, Pf, S, Sf> Rollout for CallbackRollout<P, S>
where
    P: Fn(Vec<Candidate>, usize) -> Pf + Send + Sync,
    Pf: std::future::Future<Output = Vec<Candidate>> + Send,
    S: Fn(Vec<Candidate>) -> Sf + Send + Sync,
    Sf: std::future::Future<Output = RolloutOutcome> + Send,
{
    async fn propose(&self, path: &[Candidate], depth: usize) -> Vec<Candidate> {
        (self.propose)(path.to_vec(), depth).await
    }

    async fn simulate(&self, path: &[Candidate]) -> RolloutOutcome {
        (self.simulate)(path.to_vec()).await
    }
}

/// One node in the arena-allocated plan tree.
#[derive(Debug)]
struct Node {
    /// Action taken from the parent to reach this node. The root carries a
    /// sentinel and is excluded from the returned plan path.
    candidate: Candidate,
    parent: Option<usize>,
    children: Vec<usize>,
    /// Candidates not yet expanded into children. `None` until `propose` runs.
    untried: Option<Vec<Candidate>>,
    visits: u32,
    total_reward: f64,
    depth: usize,
}

impl Node {
    fn mean_reward(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_reward / self.visits as f64
        }
    }
}

/// Result of a completed search.
#[derive(Clone, Debug)]
pub struct PlanSearchOutcome {
    /// The chosen plan chain, root → leaf (root sentinel excluded). Empty if the
    /// search could not expand a single candidate.
    pub best_path: Vec<Candidate>,
    /// Mean reward of the chosen first move (the root's selected child).
    pub best_reward: f64,
    /// Rollouts actually performed.
    pub rollouts: u32,
    /// Cumulative rollout tokens spent.
    pub tokens_used: u64,
    /// Total nodes in the tree (including the root sentinel).
    pub tree_size: usize,
}

/// Flag-gated entry point for the agent loop.
///
/// Returns `None` immediately when `FLAG_PLAN_SEARCH` is off (the default), so
/// the call site stays a cheap one-liner on the hot path. When enabled, it loads
/// the budget from `MANGOCODE_PLAN_SEARCH_*` env vars ([`PlanSearchConfig::from_env`])
/// and runs the search. The caller is responsible for feeding
/// [`PlanSearchOutcome::best_path`] into the turn as the chosen plan.
pub async fn run_if_enabled<R>(goal: &str, rollout: &R) -> Option<PlanSearchOutcome>
where
    R: Rollout + ?Sized,
{
    if !is_enabled() {
        return None;
    }
    let config = PlanSearchConfig::from_env();
    debug!(
        max_rollouts = config.max_rollouts,
        max_depth = config.max_depth,
        "plan_search: enabled, starting search"
    );
    Some(run_plan_search(goal, rollout, &config).await)
}

/// Run MCTS plan search for `goal` using `rollout`, returning the best plan
/// chain found within the configured budget.
pub async fn run_plan_search<R>(
    goal: &str,
    rollout: &R,
    config: &PlanSearchConfig,
) -> PlanSearchOutcome
where
    R: Rollout + ?Sized,
{
    let config = config.clone().clamped();
    let mut tree: Vec<Node> = Vec::with_capacity(config.max_rollouts as usize + 1);
    tree.push(Node {
        candidate: Candidate::new("root", goal),
        parent: None,
        children: Vec::new(),
        untried: None,
        visits: 0,
        total_reward: 0.0,
        depth: 0,
    });

    let mut rollouts = 0u32;
    let mut tokens_used = 0u64;

    while rollouts < config.max_rollouts {
        // 1. SELECT: descend by UCB1 until we reach a node we can expand.
        let mut node = 0usize;
        loop {
            // Ensure children have been proposed before we treat this node as
            // an interior node to descend through.
            if tree[node].untried.is_none() && tree[node].depth < config.max_depth {
                let path = path_candidates(&tree, node);
                let proposed = rollout.propose(&path, tree[node].depth).await;
                tree[node].untried = Some(proposed);
            }
            let expandable = tree[node]
                .untried
                .as_ref()
                .is_some_and(|u| !u.is_empty())
                && tree[node].depth < config.max_depth;
            if expandable || tree[node].children.is_empty() {
                break;
            }
            node = best_uct_child(&tree, node, config.exploration);
        }

        // 2. EXPAND: take one untried candidate, if any room remains.
        let leaf = if tree[node].depth < config.max_depth {
            match tree[node].untried.as_mut().and_then(Vec::pop) {
                Some(candidate) => {
                    let depth = tree[node].depth + 1;
                    let child = tree.len();
                    tree.push(Node {
                        candidate,
                        parent: Some(node),
                        children: Vec::new(),
                        untried: None,
                        visits: 0,
                        total_reward: 0.0,
                        depth,
                    });
                    tree[node].children.push(child);
                    child
                }
                None => node,
            }
        } else {
            node
        };

        // 3. SIMULATE: roll the partial plan out and score it.
        let path = path_candidates(&tree, leaf);
        let outcome = rollout.simulate(&path).await;
        tokens_used = tokens_used.saturating_add(outcome.tokens_used);
        let reward = outcome.reward(&config.weights);

        // 4. BACKPROPAGATE up the visited chain.
        let mut cur = Some(leaf);
        while let Some(idx) = cur {
            tree[idx].visits += 1;
            tree[idx].total_reward += reward;
            cur = tree[idx].parent;
        }

        rollouts += 1;
        debug!(
            rollouts,
            tokens_used, reward, "plan_search: rollout complete"
        );

        if config.token_ceiling > 0 && tokens_used >= config.token_ceiling {
            debug!(tokens_used, "plan_search: token ceiling reached");
            break;
        }
    }

    // Extract the most-visited chain from the root (robust child selection).
    let best_reward = best_child(&tree, 0)
        .map(|c| tree[c].mean_reward())
        .unwrap_or(0.0);
    let best_path = best_path(&tree);

    PlanSearchOutcome {
        best_path,
        best_reward,
        rollouts,
        tokens_used,
        tree_size: tree.len(),
    }
}

/// Candidates from the root (exclusive) down to `node` (inclusive).
fn path_candidates(tree: &[Node], node: usize) -> Vec<Candidate> {
    let mut chain = Vec::new();
    let mut cur = Some(node);
    while let Some(idx) = cur {
        if tree[idx].parent.is_some() {
            chain.push(tree[idx].candidate.clone());
        }
        cur = tree[idx].parent;
    }
    chain.reverse();
    chain
}

/// UCB1 child selection. Assumes `parent` is fully expanded (every child has at
/// least one visit), so the exploration term never divides by zero.
fn best_uct_child(tree: &[Node], parent: usize, c: f64) -> usize {
    let parent_visits = (tree[parent].visits.max(1)) as f64;
    let ln_parent = parent_visits.ln();
    let mut best = tree[parent].children[0];
    let mut best_score = f64::NEG_INFINITY;
    for &child in &tree[parent].children {
        let n = tree[child].visits.max(1) as f64;
        let exploit = tree[child].mean_reward();
        let explore = c * (ln_parent / n).sqrt();
        let score = exploit + explore;
        if score > best_score {
            best_score = score;
            best = child;
        }
    }
    best
}

/// Most-visited child of `node`, ties broken by higher mean reward.
fn best_child(tree: &[Node], node: usize) -> Option<usize> {
    tree[node]
        .children
        .iter()
        .copied()
        .max_by(|&a, &b| {
            tree[a].visits.cmp(&tree[b].visits).then_with(|| {
                tree[a]
                    .mean_reward()
                    .partial_cmp(&tree[b].mean_reward())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        })
}

/// Greedy most-visited chain from root to a leaf.
fn best_path(tree: &[Node]) -> Vec<Candidate> {
    let mut path = Vec::new();
    let mut node = 0usize;
    while let Some(child) = best_child(tree, node) {
        path.push(tree[child].candidate.clone());
        node = child;
    }
    path
}

/// Parse an env var of type `T`, returning `None` if it is unset. If the var is
/// **set but unparseable**, warn (so a typo like `MAX_ROLLOUTS=5O` is not
/// silently swallowed into the default) and return `None`.
fn env_parse<T: std::str::FromStr>(key: &str) -> Option<T> {
    let raw = std::env::var(key).ok()?;
    match raw.trim().parse() {
        Ok(v) => Some(v),
        Err(_) => {
            warn!(
                "plan_search: ignoring unparseable {key}={raw:?}; using default"
            );
            None
        }
    }
}

fn env_u32(key: &str) -> Option<u32> {
    env_parse(key)
}

fn env_u64(key: &str) -> Option<u64> {
    env_parse(key)
}

fn env_f64(key: &str) -> Option<f64> {
    env_parse(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A deterministic rollout: at every node it proposes two children, "good"
    // and "bad". A path is scored by how many "good" moves it contains, so the
    // search should converge on the all-good branch. No RNG, no I/O.
    struct MockRollout {
        branching: usize,
    }

    #[async_trait]
    impl Rollout for MockRollout {
        async fn propose(&self, _path: &[Candidate], depth: usize) -> Vec<Candidate> {
            (0..self.branching)
                .map(|i| {
                    let label = if i == 0 { "good" } else { "bad" };
                    Candidate::new(format!("{label}-{depth}-{i}"), format!("{label} step"))
                })
                .collect()
        }

        async fn simulate(&self, path: &[Candidate]) -> RolloutOutcome {
            let good = path.iter().filter(|c| c.id.starts_with("good")).count();
            let all_good = !path.is_empty() && good == path.len();
            RolloutOutcome {
                made_changes: !path.is_empty(),
                tests_passed: all_good,
                clippy_introduced: path.len() - good,
                critic_pass: all_good,
                tokens_used: 10,
            }
        }
    }

    #[test]
    fn combine_signals_bounds_and_decay() {
        let w = RewardWeights::default();
        // All good → 1.0. A failing-but-changed rollout floors near 0: the
        // clippy term earns nothing once tests fail, leaving only the (absent)
        // critic credit.
        assert!((combine_signals(true, true, 0, true, &w) - 1.0).abs() < 1e-9);
        let all_bad = combine_signals(true, false, 5, false, &w);
        assert!((0.0..0.05).contains(&all_bad), "all_bad = {all_bad}");
        assert!(combine_signals(true, true, 0, true, &w) > all_bad);
        // Clippy term decays: more findings → lower reward.
        let few = combine_signals(true, true, 1, true, &w);
        let many = combine_signals(true, true, 9, true, &w);
        assert!(few > many);
        assert!((0.0..=1.0).contains(&few));
    }

    #[test]
    fn combine_signals_zero_weights_safe() {
        let w = RewardWeights {
            tests: 0.0,
            clippy: 0.0,
            critic: 0.0,
        };
        assert_eq!(combine_signals(true, true, 0, true, &w), 0.0);
    }

    #[test]
    fn no_op_rollout_scores_zero() {
        let w = RewardWeights::default();
        // made_changes=false → 0, even though tests pass and lint is clean.
        assert_eq!(combine_signals(false, true, 0, true, &w), 0.0);
        // The struct path agrees.
        let noop = RolloutOutcome {
            made_changes: false,
            tests_passed: true,
            clippy_introduced: 0,
            critic_pass: true,
            tokens_used: 0,
        };
        assert_eq!(noop.reward(&w), 0.0);
    }

    #[test]
    fn failing_tests_earn_no_clippy_credit() {
        let w = RewardWeights::default();
        // Broken build, lint-clean: clippy term is gated off, so reward comes
        // only from the (absent) critic — strictly below a passing rollout.
        let broken_clean = combine_signals(true, false, 0, false, &w);
        let broken_clean_vs_dirty = combine_signals(true, false, 8, false, &w);
        assert_eq!(broken_clean, broken_clean_vs_dirty); // clippy count irrelevant when tests fail
        assert!(broken_clean < combine_signals(true, true, 8, false, &w));
    }

    #[test]
    fn config_clamps_out_of_range() {
        let c = PlanSearchConfig {
            max_rollouts: 0,
            max_depth: 0,
            exploration: f64::NAN,
            token_ceiling: 0,
            weights: RewardWeights::default(),
        }
        .clamped();
        assert_eq!(c.max_rollouts, 1);
        assert_eq!(c.max_depth, 1);
        assert_eq!(c.exploration, DEFAULT_EXPLORATION);
    }

    #[test]
    fn config_sanitizes_bad_weights() {
        let c = PlanSearchConfig {
            weights: RewardWeights {
                tests: -1.0,
                clippy: f64::NAN,
                critic: f64::INFINITY,
            },
            ..Default::default()
        }
        .clamped();
        assert_eq!(c.weights.tests, 0.0);
        assert_eq!(c.weights.clippy, 0.0);
        assert_eq!(c.weights.critic, 0.0);
        // Reward stays in [0, 1] even though every weight was hostile.
        let r = combine_signals(true, true, 0, true, &c.weights);
        assert!((0.0..=1.0).contains(&r), "r = {r}");
    }

    #[tokio::test]
    async fn callback_rollout_adapts_closures() {
        // The loop supplies propose/simulate as closures; verify the adapter
        // drives a full search and converges, same as a hand-written impl.
        let rollout = CallbackRollout::new(
            |_path: Vec<Candidate>, depth: usize| async move {
                vec![
                    Candidate::new(format!("good-{depth}"), "good"),
                    Candidate::new(format!("bad-{depth}"), "bad"),
                ]
            },
            |path: Vec<Candidate>| async move {
                let good = path.iter().filter(|c| c.id.starts_with("good")).count();
                let all_good = !path.is_empty() && good == path.len();
                RolloutOutcome {
                    made_changes: !path.is_empty(),
                    tests_passed: all_good,
                    clippy_introduced: path.len() - good,
                    critic_pass: all_good,
                    tokens_used: 5,
                }
            },
        );
        let config = PlanSearchConfig {
            max_rollouts: 30,
            max_depth: 3,
            ..Default::default()
        };
        let outcome = run_plan_search("goal", &rollout, &config).await;
        assert_eq!(outcome.rollouts, 30);
        assert!(outcome.best_path[0].id.starts_with("good"));
    }

    #[tokio::test]
    async fn search_converges_on_best_branch() {
        let rollout = MockRollout { branching: 2 };
        let config = PlanSearchConfig {
            max_rollouts: 40,
            max_depth: 3,
            ..Default::default()
        };
        let outcome = run_plan_search("solve it", &rollout, &config).await;

        assert_eq!(outcome.rollouts, 40);
        assert!(!outcome.best_path.is_empty());
        // The first chosen move should be a "good" one.
        assert!(
            outcome.best_path[0].id.starts_with("good"),
            "expected good first move, got {:?}",
            outcome.best_path
        );
        assert!(outcome.best_reward > 0.5);
        assert!(outcome.tree_size > 1);
    }

    #[tokio::test]
    async fn respects_rollout_budget() {
        let rollout = MockRollout { branching: 3 };
        let config = PlanSearchConfig {
            max_rollouts: 5,
            max_depth: 4,
            ..Default::default()
        };
        let outcome = run_plan_search("goal", &rollout, &config).await;
        assert_eq!(outcome.rollouts, 5);
        assert_eq!(outcome.tokens_used, 50); // 5 rollouts * 10 tokens
    }

    #[tokio::test]
    async fn token_ceiling_stops_early() {
        let rollout = MockRollout { branching: 2 };
        let config = PlanSearchConfig {
            max_rollouts: 100,
            max_depth: 3,
            token_ceiling: 25, // 3 rollouts * 10 tokens crosses 25
            ..Default::default()
        };
        let outcome = run_plan_search("goal", &rollout, &config).await;
        assert!(outcome.rollouts < 100);
        assert!(outcome.tokens_used >= 25);
    }

    #[tokio::test]
    async fn terminal_root_yields_empty_path() {
        // Propose nothing → root is terminal, no plan can be built.
        struct Empty;
        #[async_trait]
        impl Rollout for Empty {
            async fn propose(&self, _p: &[Candidate], _d: usize) -> Vec<Candidate> {
                Vec::new()
            }
            async fn simulate(&self, _p: &[Candidate]) -> RolloutOutcome {
                RolloutOutcome {
                    tokens_used: 1,
                    ..Default::default()
                }
            }
        }
        let outcome = run_plan_search("goal", &Empty, &PlanSearchConfig::default()).await;
        assert!(outcome.best_path.is_empty());
        assert_eq!(outcome.tree_size, 1);
    }
}
