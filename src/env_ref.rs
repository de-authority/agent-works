//! `${ENV:VAR}` — product-layer resolution interface (P1).
//!
//! Contract (design "`${ENV:VAR}` resolution spec"):
//! - **Resolution happens at the product layer**: the framework takes
//!   literals. The product calls this module to pre-resolve once at connection
//!   setup / tool registration, and hands the literal to the framework;
//! - timing = once at registration; **unset → report which variable is missing**
//!   (connection failure) — it does not wait until call time to blow up;
//! - **No recursive expansion**: resolved values are inserted as-is; `${ENV:}`
//!   inside a value is left untouched;
//! - Only the `${ENV:NAME}` form is recognized; `${param}` (tool-parameter
//!   substitution), `$FOO`, `${FOO}` are all kept as-is — two substitution
//!   syntaxes, two timings, neither swallows the other.

use std::collections::BTreeMap;

/// Reference errors. fail-closed: a malformed reference is not "treat it as a
/// literal" — it is an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvRefError {
    /// The environment variable in `${ENV:NAME}` is not set. Carries the
    /// variable name, so the error tells the user exactly which one to provide.
    MissingVar { name: String },
    /// `${ENV:}` (empty name).
    EmptyName,
    /// `${ENV:` without a closing `}`.
    Unclosed,
}

impl std::fmt::Display for EnvRefError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingVar { name } => {
                write!(
                    f,
                    "environment variable `{name}` is not set (required by ${{ENV:{name}}})"
                )
            }
            Self::EmptyName => write!(f, "empty environment variable name in `${{ENV:}}`"),
            Self::Unclosed => write!(
                f,
                "unclosed environment reference `${{ENV:…` (missing `}}`)"
            ),
        }
    }
}

impl std::error::Error for EnvRefError {}

const OPEN: &str = "${ENV:";

/// Resolve the `${ENV:NAME}` references in a string.
///
/// `lookup` is the value accessor (tests inject a dictionary, the product uses
/// [`system_env`]). One string may hold several references; inserted values are
/// **not re-scanned** (no recursive expansion). Any missing variable errors out
/// immediately, and the error carries the variable name.
pub fn resolve_env_refs<F>(input: &str, lookup: F) -> Result<String, EnvRefError>
where
    F: Fn(&str) -> Option<String>,
{
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(start) = rest.find(OPEN) {
        out.push_str(&rest[..start]);
        let after = &rest[start + OPEN.len()..];
        let Some(end) = after.find('}') else {
            return Err(EnvRefError::Unclosed);
        };
        let name = &after[..end];
        if name.is_empty() {
            return Err(EnvRefError::EmptyName);
        }
        match lookup(name) {
            Some(value) => out.push_str(&value), // inserted as-is, no second expansion
            None => {
                return Err(EnvRefError::MissingVar {
                    name: name.to_string(),
                });
            }
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Resolve one env/headers map value by value. Keys are untouched; only values
/// are resolved.
pub fn resolve_env_map<F>(
    map: &BTreeMap<String, String>,
    lookup: F,
) -> Result<BTreeMap<String, String>, EnvRefError>
where
    F: Fn(&str) -> Option<String>,
{
    map.iter()
        .map(|(k, v)| Ok((k.clone(), resolve_env_refs(v, &lookup)?)))
        .collect()
}

/// Process-environment accessor (the product layer's default).
pub fn system_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup<'a>(dict: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            dict.iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| (*v).to_string())
        }
    }

    #[test]
    fn resolves_single_and_multiple_refs() {
        let got = resolve_env_refs("tok=${ENV:TOK} and ${ENV:HOME}", |n| match n {
            "TOK" => Some("abc".into()),
            "HOME" => Some("/home/me".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(got, "tok=abc and /home/me");
    }

    #[test]
    fn missing_var_names_the_variable() {
        let err = resolve_env_refs("${ENV:NOTION_TOKEN}", |_| None).unwrap_err();
        assert_eq!(
            err,
            EnvRefError::MissingVar {
                name: "NOTION_TOKEN".into()
            }
        );
        assert!(err.to_string().contains("NOTION_TOKEN"));
    }

    #[test]
    fn does_not_recurse_into_resolved_values() {
        // A value containing ${ENV:} is not expanded again (no recursion).
        let got = resolve_env_refs("${ENV:A}", lookup(&[("A", "${ENV:B}"), ("B", "x")])).unwrap();
        assert_eq!(got, "${ENV:B}");
    }

    #[test]
    fn leaves_other_syntaxes_untouched() {
        let input = "${env} $FOO ${FOO} ${param} ${ENVX:Y} ${{ENV:X}";
        let got = resolve_env_refs(input, |_| Some("SHOULD_NOT_APPEAR".into())).unwrap();
        // `${ENV:` must not appear (isn't `${ENVX:Y}`'s prefix `${ENV:`? — it is
        // a substring!). `${ENVX:Y}` starts with `${ENV` but its 6th char is `X` not `:`
        assert_eq!(got, input);
    }

    #[test]
    fn malformed_refs_fail_closed() {
        assert_eq!(
            resolve_env_refs("pre${ENV:NO_CLOSE", |_| None).unwrap_err(),
            EnvRefError::Unclosed
        );
        assert_eq!(
            resolve_env_refs("${ENV:}", |_| None).unwrap_err(),
            EnvRefError::EmptyName
        );
        // The later malformed ref must also error, not be silently truncated
        assert_eq!(
            resolve_env_refs("ok${ENV:A}bad${ENV:", |_| Some("v".into())).unwrap_err(),
            EnvRefError::Unclosed
        );
    }

    #[test]
    fn resolves_map_values_only() {
        let map = BTreeMap::from([
            (
                "NOTION_TOKEN".to_string(),
                "${ENV:NOTION_TOKEN}".to_string(),
            ),
            ("plain".to_string(), "literal".to_string()),
        ]);
        let got = resolve_env_map(&map, lookup(&[("NOTION_TOKEN", "tok-1")])).unwrap();
        assert_eq!(got.get("NOTION_TOKEN").unwrap(), "tok-1");
        assert_eq!(got.get("plain").unwrap(), "literal");

        let err = resolve_env_map(&map, |_| None).unwrap_err();
        assert_eq!(
            err,
            EnvRefError::MissingVar {
                name: "NOTION_TOKEN".into()
            }
        );
    }

    #[test]
    fn empty_input_and_no_refs_pass_through() {
        assert_eq!(resolve_env_refs("", |_| None).unwrap(), "");
        assert_eq!(
            resolve_env_refs("plain text", |_| None).unwrap(),
            "plain text"
        );
    }
}
