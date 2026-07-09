use std::fmt;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use serde::Deserialize;
use tokio::sync::broadcast;

use crate::config::ClaudeUsageConfig;
use crate::data_type::DataType;

use super::_base::Provider;

const DEFAULT_POLL_INTERVAL_SECONDS: u64 = 60;
const RAW_HID_REPORT_SIZE: usize = 32;
const USAGE_ENDPOINT: &str = "https://api.anthropic.com/api/oauth/usage";
const OAUTH_BETA_HEADER: &str = "oauth-2025-04-20";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ClaudeUsage {
    session_remaining: u8,
    weekly_remaining: u8,
    reset_minutes: u16,
    flags: u8,
}

impl ClaudeUsage {
    fn raw_hid_report(self) -> [u8; RAW_HID_REPORT_SIZE] {
        let mut report = [0; RAW_HID_REPORT_SIZE];
        report[0] = DataType::ClaudeUsage as u8;
        report[1] = self.session_remaining;
        report[2] = self.weekly_remaining;
        report[3..5].copy_from_slice(&self.reset_minutes.to_le_bytes());
        report[5] = self.flags;
        report
    }
}

struct OAuthToken(String);

impl OAuthToken {
    fn new(token: String) -> Result<Self, ClaudeUsageError> {
        let token = token.trim().to_owned();
        if token.is_empty() || token.contains('\r') || token.contains('\n') {
            return Err(ClaudeUsageError::InvalidCredentials);
        }

        Ok(Self(token))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ClaudeUsageError {
    CredentialsUnavailable,
    InvalidCredentials,
    InvalidUsageResponse,
    RequestFailed,
}

impl fmt::Display for ClaudeUsageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::CredentialsUnavailable => "credentials unavailable",
            Self::InvalidCredentials => "credentials are invalid",
            Self::InvalidUsageResponse => "usage response is invalid",
            Self::RequestFailed => "usage request failed",
        };

        formatter.write_str(message)
    }
}

trait TokenSource: Send + Sync {
    fn load_token(&self) -> Result<OAuthToken, ClaudeUsageError>;
}

trait UsageClient: Send + Sync {
    fn fetch_usage(&self, request: UsageRequest<'_>) -> Result<ClaudeUsage, ClaudeUsageError>;
}

struct UsageRequest<'a> {
    endpoint: &'a str,
    oauth_beta_header: &'a str,
    token: &'a OAuthToken,
}

#[derive(Clone)]
struct ClaudeCodeTokenSource {
    #[cfg(target_os = "macos")]
    keychain_service: Option<String>,
    credentials_path: Option<PathBuf>,
}

impl ClaudeCodeTokenSource {
    fn new(config: &ClaudeUsageConfig) -> Self {
        Self {
            #[cfg(target_os = "macos")]
            keychain_service: config.keychain_service.clone(),
            credentials_path: config.credentials_path.clone(),
        }
    }

    fn load_file_token(&self) -> Result<OAuthToken, ClaudeUsageError> {
        let path = match &self.credentials_path {
            Some(path) => path.clone(),
            None => default_credentials_path()?,
        };
        let credentials = std::fs::read_to_string(path).map_err(|_| ClaudeUsageError::CredentialsUnavailable)?;
        token_from_credentials_json(&credentials)
    }
}

impl TokenSource for ClaudeCodeTokenSource {
    fn load_token(&self) -> Result<OAuthToken, ClaudeUsageError> {
        #[cfg(target_os = "macos")]
        if let Some(service) = self.keychain_service.as_deref() {
            if let Ok(token) = load_keychain_token(service) {
                return Ok(token);
            }
        }

        self.load_file_token()
    }
}

struct CurlUsageClient;

impl UsageClient for CurlUsageClient {
    fn fetch_usage(&self, request: UsageRequest<'_>) -> Result<ClaudeUsage, ClaudeUsageError> {
        let beta_header = format!("anthropic-beta: {}", request.oauth_beta_header);
        let mut child = Command::new(curl_binary())
            .args([
                "--disable",
                "--fail",
                "--silent",
                "--show-error",
                "--connect-timeout",
                "10",
                "--max-time",
                "15",
                "--request",
                "GET",
                "--header",
                &beta_header,
                "--header",
                "@-",
                request.endpoint,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|_| ClaudeUsageError::RequestFailed)?;

        let Some(mut stdin) = child.stdin.take() else {
            return Err(ClaudeUsageError::RequestFailed);
        };

        if writeln!(stdin, "Authorization: Bearer {}", request.token.as_str()).is_err() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(ClaudeUsageError::RequestFailed);
        }
        drop(stdin);

        let output = child.wait_with_output().map_err(|_| ClaudeUsageError::RequestFailed)?;
        if !output.status.success() {
            return Err(ClaudeUsageError::RequestFailed);
        }

        let response = std::str::from_utf8(&output.stdout).map_err(|_| ClaudeUsageError::InvalidUsageResponse)?;
        usage_from_response(response, Utc::now())
    }
}

pub struct ClaudeUsageProvider {
    host_to_device_sender: broadcast::Sender<Vec<u8>>,
    is_started: Arc<AtomicBool>,
    poll_interval: Duration,
    token_source: Arc<dyn TokenSource>,
    usage_client: Arc<dyn UsageClient>,
}

impl ClaudeUsageProvider {
    pub fn new(host_to_device_sender: broadcast::Sender<Vec<u8>>, config: ClaudeUsageConfig) -> Box<dyn Provider> {
        let poll_interval_seconds = config.poll_interval_seconds.unwrap_or(DEFAULT_POLL_INTERVAL_SECONDS).max(1);
        Box::new(Self::with_dependencies(
            host_to_device_sender,
            Duration::from_secs(poll_interval_seconds),
            Arc::new(ClaudeCodeTokenSource::new(&config)),
            Arc::new(CurlUsageClient),
        ))
    }

    fn with_dependencies(
        host_to_device_sender: broadcast::Sender<Vec<u8>>,
        poll_interval: Duration,
        token_source: Arc<dyn TokenSource>,
        usage_client: Arc<dyn UsageClient>,
    ) -> Self {
        Self {
            host_to_device_sender,
            is_started: Arc::new(AtomicBool::new(false)),
            poll_interval,
            token_source,
            usage_client,
        }
    }
}

impl Provider for ClaudeUsageProvider {
    fn start(&self) {
        if self.is_started.swap(true, Relaxed) {
            return;
        }

        tracing::info!("Claude Usage Provider started");
        let host_to_device_sender = self.host_to_device_sender.clone();
        let is_started = self.is_started.clone();
        let poll_interval = self.poll_interval;
        let token_source = self.token_source.clone();
        let usage_client = self.usage_client.clone();

        std::thread::spawn(move || {
            while is_started.load(Relaxed) {
                match poll_report(token_source.as_ref(), usage_client.as_ref()) {
                    Ok(report) => {
                        if let Err(error) = host_to_device_sender.send(report.to_vec()) {
                            tracing::warn!("Claude Usage Provider failed to send data: {error}");
                        }
                    }
                    Err(error) => tracing::warn!("Claude Usage Provider poll failed: {error}"),
                }

                wait_for_next_poll(&is_started, poll_interval);
            }

            tracing::info!("Claude Usage Provider stopped");
        });
    }

    fn stop(&self) {
        self.is_started.store(false, Relaxed);
    }
}

fn poll_report(token_source: &dyn TokenSource, usage_client: &dyn UsageClient) -> Result<[u8; RAW_HID_REPORT_SIZE], ClaudeUsageError> {
    let token = token_source.load_token()?;
    let usage = usage_client.fetch_usage(UsageRequest {
        endpoint: USAGE_ENDPOINT,
        oauth_beta_header: OAUTH_BETA_HEADER,
        token: &token,
    })?;
    Ok(usage.raw_hid_report())
}

fn wait_for_next_poll(is_started: &AtomicBool, poll_interval: Duration) {
    let deadline = Instant::now() + poll_interval;
    while is_started.load(Relaxed) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return;
        }

        std::thread::sleep(remaining.min(Duration::from_secs(1)));
    }
}

#[cfg(target_os = "macos")]
fn load_keychain_token(service: &str) -> Result<OAuthToken, ClaudeUsageError> {
    let output = Command::new("/usr/bin/security")
        .args(["find-generic-password", "-s", service, "-w"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .map_err(|_| ClaudeUsageError::CredentialsUnavailable)?;
    if !output.status.success() {
        return Err(ClaudeUsageError::CredentialsUnavailable);
    }

    let credentials = std::str::from_utf8(&output.stdout).map_err(|_| ClaudeUsageError::InvalidCredentials)?;
    token_from_credentials_json(credentials)
}

fn default_credentials_path() -> Result<PathBuf, ClaudeUsageError> {
    let home = std::env::var_os("HOME").ok_or(ClaudeUsageError::CredentialsUnavailable)?;
    Ok(PathBuf::from(home).join(".claude").join(".credentials.json"))
}

fn curl_binary() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "/usr/bin/curl"
    }

    #[cfg(not(target_os = "macos"))]
    {
        "curl"
    }
}

fn token_from_credentials_json(credentials: &str) -> Result<OAuthToken, ClaudeUsageError> {
    let credentials: ClaudeCodeCredentials = serde_json::from_str(credentials).map_err(|_| ClaudeUsageError::InvalidCredentials)?;
    OAuthToken::new(credentials.claude_ai_oauth.access_token)
}

fn usage_from_response(response: &str, now: DateTime<Utc>) -> Result<ClaudeUsage, ClaudeUsageError> {
    let response: UsageResponse = serde_json::from_str(response).map_err(|_| ClaudeUsageError::InvalidUsageResponse)?;
    let five_hour = response.five_hour.ok_or(ClaudeUsageError::InvalidUsageResponse)?;
    let seven_day = response.seven_day.ok_or(ClaudeUsageError::InvalidUsageResponse)?;
    let reset_at = DateTime::parse_from_rfc3339(&five_hour.resets_at)
        .map_err(|_| ClaudeUsageError::InvalidUsageResponse)?
        .with_timezone(&Utc);
    let reset_minutes = reset_at.signed_duration_since(now).num_minutes().clamp(0, i64::from(u16::MAX)) as u16;

    Ok(ClaudeUsage {
        session_remaining: remaining_percent(five_hour.utilization)?,
        weekly_remaining: remaining_percent(seven_day.utilization)?,
        reset_minutes,
        flags: 0,
    })
}

fn remaining_percent(utilization: f64) -> Result<u8, ClaudeUsageError> {
    if !utilization.is_finite() {
        return Err(ClaudeUsageError::InvalidUsageResponse);
    }

    let used_percent = if (0.0..=1.0).contains(&utilization) {
        utilization * 100.0
    } else {
        utilization
    };
    let used = used_percent.clamp(0.0, 100.0).round() as u8;
    Ok(100 - used)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClaudeCodeCredentials {
    claude_ai_oauth: ClaudeAiOauth,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClaudeAiOauth {
    access_token: String,
}

#[derive(Deserialize)]
struct UsageResponse {
    five_hour: Option<UsageWindow>,
    seven_day: Option<UsageWindow>,
}

#[derive(Deserialize)]
struct UsageWindow {
    utilization: f64,
    resets_at: String,
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use super::*;

    struct StaticTokenSource;

    impl TokenSource for StaticTokenSource {
        fn load_token(&self) -> Result<OAuthToken, ClaudeUsageError> {
            OAuthToken::new("test-token".to_owned())
        }
    }

    struct StaticUsageClient;

    impl UsageClient for StaticUsageClient {
        fn fetch_usage(&self, request: UsageRequest<'_>) -> Result<ClaudeUsage, ClaudeUsageError> {
            assert_eq!(request.endpoint, USAGE_ENDPOINT);
            assert_eq!(request.oauth_beta_header, OAUTH_BETA_HEADER);
            Ok(ClaudeUsage {
                session_remaining: 61,
                weekly_remaining: 73,
                reset_minutes: 75,
                flags: 0,
            })
        }
    }

    #[test]
    fn raw_hid_report_should_match_claude_usage_wire_format() {
        let report = ClaudeUsage {
            session_remaining: 61,
            weekly_remaining: 73,
            reset_minutes: 75,
            flags: 2,
        }
        .raw_hid_report();

        assert_eq!(
            report,
            [0xB2, 61, 73, 75, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn usage_response_should_convert_utilization_to_remaining_percentages() {
        let response = r#"{
            "five_hour": {"utilization": 0.39, "resets_at": "2026-07-09T14:45:00Z"},
            "seven_day": {"utilization": 0.27, "resets_at": "2026-07-14T14:45:00Z"}
        }"#;
        let now = Utc.with_ymd_and_hms(2026, 7, 9, 13, 30, 0).unwrap();

        let usage = usage_from_response(response, now).unwrap();

        assert_eq!(
            usage,
            ClaudeUsage {
                session_remaining: 61,
                weekly_remaining: 73,
                reset_minutes: 75,
                flags: 0,
            }
        );
    }

    #[test]
    fn usage_response_should_reject_missing_weekly_window() {
        let response = r#"{
            "five_hour": {"utilization": 0.39, "resets_at": "2026-07-09T14:45:00Z"},
            "seven_day": null
        }"#;
        let now = Utc.with_ymd_and_hms(2026, 7, 9, 13, 30, 0).unwrap();

        let error = usage_from_response(response, now).unwrap_err();

        assert_eq!(error, ClaudeUsageError::InvalidUsageResponse);
    }

    #[test]
    fn remaining_percent_should_accept_legacy_percentage_values() {
        let remaining = remaining_percent(39.0).unwrap();

        assert_eq!(remaining, 61);
    }

    #[test]
    fn credentials_should_extract_access_token_without_exposing_other_fields() {
        let credentials = r#"{
            "claudeAiOauth": {
                "accessToken": "test-token",
                "refreshToken": "test-refresh-token"
            }
        }"#;

        let token = token_from_credentials_json(credentials);

        assert!(token.is_ok());
    }

    #[test]
    fn poll_report_should_use_injected_token_and_usage_sources() {
        let report = poll_report(&StaticTokenSource, &StaticUsageClient).unwrap();

        assert_eq!(report[0], DataType::ClaudeUsage as u8);
    }
}
