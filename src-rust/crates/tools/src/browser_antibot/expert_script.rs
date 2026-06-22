//! Shadow-root patching for expert mode (`_prepare_expert`-style hook).

pub const FORCE_OPEN_SHADOW_ON_NEW_DOCUMENT: &str = r#"
console.log("mangocode: attachShadow hook");
Element.prototype._mango_attachShadow = Element.prototype.attachShadow;
Element.prototype.attachShadow = function(init) {
  init = Object.assign({}, init || {}, { mode: "open" });
  return this._mango_attachShadow.call(this, init);
};
"#;
