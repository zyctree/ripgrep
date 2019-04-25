use std::io;
use std::path::{Path, PathBuf};
use std::result;

use snafu::{Backtrace, Context, Snafu};

pub type Result<T> = result::Result<T, Error>;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    #[snafu(display("I/O error: {}", source))]
    IO {
        backtrace: Backtrace,
        source: io::Error,
    },
    #[snafu(display(
        "failed to retrieve current working directory: {}", source,
    ))]
    CurrentDir {
        backtrace: Backtrace,
        source: io::Error,
    },
    #[snafu(display("sorting by modified time is unsupported: {}", source))]
    UnsupportedSortModified {
        backtrace: Backtrace,
        source: io::Error,
    },
    #[snafu(display("sorting by access time is unsupported: {}", source))]
    UnsupportedSortAccess {
        backtrace: Backtrace,
        source: io::Error,
    },
    #[snafu(display("sorting by creation time is unsupported: {}", source))]
    UnsupportedSortCreation {
        backtrace: Backtrace,
        source: io::Error,
    },
    #[snafu(display(
        "I/O error parsing config in {}: {}",
        path.display(),
        source,
    ))]
    ConfigIO {
        backtrace: Backtrace,
        source: io::Error,
        path: PathBuf,
    },
    #[snafu(display(
        "config parse error on line {}: {}",
        line_number,
        source,
    ))]
    ConfigInvalidUTF8 {
        backtrace: Backtrace,
        source: bstr::Utf8Error,
        line_number: u64,
    },
    #[snafu(display("failed to initialize logger: {}", source))]
    LoggerInit {
        backtrace: Backtrace,
        source: log::SetLoggerError,
    },
    #[snafu(display("{}", source))]
    Clap {
        backtrace: Backtrace,
        source: clap::Error,
    },
    #[snafu(display("{}", source))]
    ReadManyPatterns {
        backtrace: Backtrace,
        source: io::Error,
    },
    #[snafu(display("{}", source))]
    ReadOSPattern {
        backtrace: Backtrace,
        source: grep::cli::InvalidPatternError,
    },
    #[snafu(display(
        "A path separator must be exactly one byte, but \
         the given separator is {} bytes: {}\n\
         In some shells on Windows '/' is automatically \
         expanded. Use '//' instead.",
        separator.len(), grep::cli::escape(&separator),
    ))]
    InvalidPathSeparator {
        backtrace: Backtrace,
        separator: Vec<u8>,
    },
    #[snafu(display("error parsing -g/--glob: {}", source))]
    InvalidGlobFlag {
        backtrace: Backtrace,
        source: ignore::Error,
    },
    #[snafu(display("error parsing --iglob: {}", source))]
    InvalidIGlobFlag {
        backtrace: Backtrace,
        source: ignore::Error,
    },
    #[snafu(display("error building --glob matcher: {}", source))]
    GlobBuild {
        backtrace: Backtrace,
        source: ignore::Error,
    },
    #[snafu(display("error parsing --pre-glob: {}", source))]
    InvalidPreGlob {
        backtrace: Backtrace,
        source: ignore::Error,
    },
    #[snafu(display("error building --pre-glob matcher: {}", source))]
    PreGlobBuild {
        backtrace: Backtrace,
        source: ignore::Error,
    },
    #[snafu(display("error parsing --type-add value: {}", source))]
    InvalidTypeDefinition {
        backtrace: Backtrace,
        source: ignore::Error,
    },
    #[snafu(display("error building type matcher: {}", source))]
    TypeDefinitionBuild {
        backtrace: Backtrace,
        source: ignore::Error,
    },
    #[snafu(display("failed to parse {} value as a number: {}", flag, source))]
    InvalidNumber {
        backtrace: Backtrace,
        source: std::num::ParseIntError,
        flag: String,
    },
    #[snafu(display(
        "failed to parse {} value as a file size: {}", flag, source,
    ))]
    InvalidHumanSize {
        backtrace: Backtrace,
        source: grep::cli::ParseSizeError,
        flag: String,
    },
    #[snafu(display(
        "number given to {} is too large (limit is {})",
        flag, limit,
    ))]
    NumberTooBig {
        backtrace: Backtrace,
        flag: String,
        limit: u64,
    },
    #[snafu(display("{}", source))]
    SearchConfig {
        backtrace: Backtrace,
        source: grep::searcher::ConfigError,
    },
    #[snafu(display("invalid --colors spec: {}", source))]
    InvalidColorSpec {
        backtrace: Backtrace,
        source: grep::printer::ColorError,
    },
    #[snafu(display("{}", suggest(source)))]
    RustRegex {
        backtrace: Backtrace,
        source: grep::regex::Error,
    },
    #[snafu(display(
        "regex could not be compiled with either the default regex \
         engine or with PCRE2.\n\n\
         default regex engine error:\n{}\n{}\n{}\n\n\
         PCRE2 regex engine error:\n{}",
        "~".repeat(79),
        rust_err,
        "~".repeat(79),
        pcre_err,
    ))]
    Hybrid {
        backtrace: Backtrace,
        rust_err: Box<Error>,
        pcre_err: Box<Error>,
    },
    #[snafu(display("failed to write type definitions to stdout: {}", source))]
    WriteTypes {
        backtrace: Backtrace,
        source: io::Error,
    },
    #[cfg(feature = "pcre2")]
    #[snafu(display("{}", suggest(source)))]
    PCRE2Regex {
        backtrace: Backtrace,
        source: grep::pcre2::Error,
    },
    #[snafu(display("PCRE2 is not available in this build of ripgrep"))]
    PCRE2Unavailable {
        backtrace: Backtrace,
    },
    #[snafu(display("failed to write PCRE2 version to stdout: {}", source))]
    PCRE2Version {
        backtrace: Backtrace,
        source: io::Error,
    },
}

impl Error {
    /// Return true if and only if this corresponds to an I/O error generated
    /// by a broken pipe.
    pub fn is_broken_pipe(&self) -> bool {
        self.io_err().map_or(false, |e| e.kind() == io::ErrorKind::BrokenPipe)
    }

    /// Return a reference to this error's underlying I/O error, if one exists.
    pub fn io_err(&self) -> Option<&io::Error> {
        match *self {
            Error::IO { ref source, .. } => Some(source),
            Error::CurrentDir { ref source, .. } => Some(source),
            Error::UnsupportedSortModified { ref source, .. } => Some(source),
            Error::UnsupportedSortAccess { ref source, .. } => Some(source),
            Error::UnsupportedSortCreation { ref source, .. } => Some(source),
            Error::ConfigIO { ref source, .. } => Some(source),
            Error::ReadManyPatterns { ref source, .. } => Some(source),
            Error::WriteTypes { ref source, .. } => Some(source),
            Error::PCRE2Version { ref source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Inspect an error's display string and look for potentially suggestions
/// to give to an end user.
///
/// These include:
///
/// 1. If the error results from the use of a new line literal, then return a
/// new message suggesting the use of the -U/--multiline flag.
/// 2. If the error correspond to a syntax error that PCRE2 could handle, then
/// add a message to suggest the use of -P/--pcre2.
fn suggest<E: std::error::Error>(err: &E) -> String {
    let msg = err.to_string();
    if msg.contains("the literal") && msg.contains("not allowed") {
        format!("{}

Consider enabling multiline mode with the --multiline flag (or -U for short).
When multiline mode is enabled, new line characters can be matched.", msg)
    } else if cfg!(feature = "pcre2") &&
        (msg.contains("backreferences") || msg.contains("look-around"))
    {
        format!("{}

Consider enabling PCRE2 with the --pcre2 flag, which can handle backreferences
and look-around.", msg)
    } else {
        msg
    }
}
