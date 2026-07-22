use std::fmt;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::broadcast;

use crate::config::ClaudeUsageConfig;
use crate::data_type::DataType;

use super::_base::Provider;

const DEFAULT_POLL_INTERVAL_SECONDS: u64 = 60;
const DEFAULT_KEYCHAIN_SERVICE: &str = "Claude Code-credentials";
const RAW_HID_REPORT_SIZE: usize = 32;
const USAGE_ENDPOINT: &str = "https://api.anthropic.com/api/oauth/usage";
const TOKEN_ENDPOINT: &str = "https://platform.claude.com/v1/oauth/token";
const CLAUDE_CODE_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const OAUTH_BETA_HEADER: &str = "oauth-2025-04-20";
const REFRESH_EARLY_MILLIS: i64 = 5 * 60 * 1000;
const FLAG_SAMPLE_VALID: u8 = 1 << 1;
const FLAG_EXTRA_ENABLED: u8 = 1 << 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ClaudeUsage {
    session_remaining: u8,
    weekly_remaining: u8,
    reset_minutes: u16,
    extra_remaining: u8,
    extra_remaining_euros: u16,
    extra_enabled: bool,
}

impl ClaudeUsage {
    fn raw_hid_report(self) -> [u8; RAW_HID_REPORT_SIZE] {
        raw_hid_report(Some(self), ClaudeUsageErrorCode::None)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum ClaudeUsageErrorCode {
    None = 0,
    Auth = 1,
    Network = 2,
    Api = 3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ClaudeUsageError {
    CredentialsUnavailable,
    InvalidCredentials,
    AuthenticationFailed,
    NetworkRequestFailed,
    InvalidUsageResponse,
    ApiRequestFailed,
}

impl ClaudeUsageError {
    fn code(self) -> ClaudeUsageErrorCode {
        match self {
            Self::CredentialsUnavailable | Self::InvalidCredentials | Self::AuthenticationFailed => ClaudeUsageErrorCode::Auth,
            Self::NetworkRequestFailed => ClaudeUsageErrorCode::Network,
            Self::InvalidUsageResponse | Self::ApiRequestFailed => ClaudeUsageErrorCode::Api,
        }
    }
}

impl fmt::Display for ClaudeUsageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::CredentialsUnavailable => "credentials unavailable",
            Self::InvalidCredentials => "credentials are invalid",
            Self::AuthenticationFailed => "authentication failed",
            Self::NetworkRequestFailed => "network request failed",
            Self::InvalidUsageResponse => "usage response is invalid",
            Self::ApiRequestFailed => "usage API request failed",
        };

        formatter.write_str(message)
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

trait TokenSource: Send + Sync {
    fn load_token(&self, force_refresh: bool) -> Result<OAuthToken, ClaudeUsageError>;
}

trait UsageClient: Send + Sync {
    fn fetch_usage(&self, request: UsageRequest<'_>) -> Result<ClaudeUsage, ClaudeUsageError>;
}

trait CredentialsStore: Send + Sync {
    fn load(&self) -> Result<StoredCredentials, ClaudeUsageError>;
    fn save(&self, location: &CredentialsLocation, credentials: &str) -> Result<(), ClaudeUsageError>;
}

trait TokenRefresher: Send + Sync {
    fn refresh(&self, request: RefreshRequest<'_>) -> Result<RefreshResponse, ClaudeUsageError>;
}

struct UsageRequest<'a> {
    endpoint: &'a str,
    oauth_beta_header: &'a str,
    token: &'a OAuthToken,
}

struct RefreshRequest<'a> {
    refresh_token: &'a str,
    scopes: &'a [String],
}

#[derive(Debug, Deserialize)]
struct RefreshResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: i64,
    refresh_token_expires_in: Option<i64>,
    scope: Option<String>,
}

struct StoredCredentials {
    contents: String,
    location: CredentialsLocation,
}

enum CredentialsLocation {
    #[cfg(target_os = "macos")]
    Keychain {
        service: String,
    },
    File(PathBuf),
}

#[derive(Clone)]
struct SystemCredentialsStore {
    #[cfg(target_os = "macos")]
    keychain_service: Option<String>,
    credentials_path: Option<PathBuf>,
}

impl SystemCredentialsStore {
    fn new(config: &ClaudeUsageConfig) -> Self {
        Self {
            #[cfg(target_os = "macos")]
            keychain_service: config.keychain_service.clone(),
            credentials_path: config.credentials_path.clone(),
        }
    }

    fn load_file(&self) -> Result<StoredCredentials, ClaudeUsageError> {
        let path = match &self.credentials_path {
            Some(path) => path.clone(),
            None => default_credentials_path()?,
        };
        let contents = fs::read_to_string(&path).map_err(|_| ClaudeUsageError::CredentialsUnavailable)?;
        Ok(StoredCredentials {
            contents,
            location: CredentialsLocation::File(path),
        })
    }
}

impl CredentialsStore for SystemCredentialsStore {
    fn load(&self) -> Result<StoredCredentials, ClaudeUsageError> {
        #[cfg(target_os = "macos")]
        {
            let service = keychain_service(self.keychain_service.as_deref());
            if let Ok(contents) = load_keychain_credentials(service) {
                return Ok(StoredCredentials {
                    contents,
                    location: CredentialsLocation::Keychain {
                        service: service.to_owned(),
                    },
                });
            }
        }

        self.load_file()
    }

    fn save(&self, location: &CredentialsLocation, credentials: &str) -> Result<(), ClaudeUsageError> {
        match location {
            #[cfg(target_os = "macos")]
            CredentialsLocation::Keychain { service } => save_keychain_credentials(service, credentials),
            CredentialsLocation::File(path) => save_file_credentials(path, credentials),
        }
    }
}

struct ClaudeCodeTokenSource {
    credentials_store: Box<dyn CredentialsStore>,
    token_refresher: Box<dyn TokenRefresher>,
    refresh_guard: Mutex<()>,
}

impl ClaudeCodeTokenSource {
    fn new(config: &ClaudeUsageConfig) -> Self {
        Self::with_dependencies(Box::new(SystemCredentialsStore::new(config)), Box::new(CurlTokenRefresher))
    }

    fn with_dependencies(credentials_store: Box<dyn CredentialsStore>, token_refresher: Box<dyn TokenRefresher>) -> Self {
        Self {
            credentials_store,
            token_refresher,
            refresh_guard: Mutex::new(()),
        }
    }
}

impl TokenSource for ClaudeCodeTokenSource {
    fn load_token(&self, force_refresh: bool) -> Result<OAuthToken, ClaudeUsageError> {
        let _guard = self.refresh_guard.lock().map_err(|_| ClaudeUsageError::CredentialsUnavailable)?;
        let stored = self.credentials_store.load()?;
        let credentials = credentials_from_json(&stored.contents)?;
        let now_millis = Utc::now().timestamp_millis();

        if !force_refresh && !token_needs_refresh(credentials.expires_at, now_millis) {
            return OAuthToken::new(credentials.access_token);
        }

        if credentials.refresh_token_expires_at <= now_millis {
            return Err(ClaudeUsageError::InvalidCredentials);
        }

        let response = self.token_refresher.refresh(RefreshRequest {
            refresh_token: &credentials.refresh_token,
            scopes: &credentials.scopes,
        })?;
        let (updated, access_token) = updated_credentials_json(&stored.contents, response, now_millis)?;
        self.credentials_store.save(&stored.location, &updated)?;

        OAuthToken::new(access_token)
    }
}

struct CurlUsageClient;

impl UsageClient for CurlUsageClient {
    fn fetch_usage(&self, request: UsageRequest<'_>) -> Result<ClaudeUsage, ClaudeUsageError> {
        let beta_header = format!("anthropic-beta: {}", request.oauth_beta_header);
        let mut child = Command::new(curl_binary())
            .args([
                "--disable",
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
                "--write-out",
                "\n%{http_code}",
                request.endpoint,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|_| ClaudeUsageError::NetworkRequestFailed)?;

        let Some(mut stdin) = child.stdin.take() else {
            return Err(ClaudeUsageError::NetworkRequestFailed);
        };
        if writeln!(stdin, "Authorization: Bearer {}", request.token.as_str()).is_err() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(ClaudeUsageError::NetworkRequestFailed);
        }
        drop(stdin);

        let output = child.wait_with_output().map_err(|_| ClaudeUsageError::NetworkRequestFailed)?;
        if !output.status.success() {
            return Err(ClaudeUsageError::NetworkRequestFailed);
        }

        let response = parse_curl_response(&output.stdout)?;
        match response.status {
            200 => usage_from_response(&response.body, Utc::now()),
            401 => Err(ClaudeUsageError::AuthenticationFailed),
            _ => Err(ClaudeUsageError::ApiRequestFailed),
        }
    }
}

struct CurlTokenRefresher;

impl TokenRefresher for CurlTokenRefresher {
    fn refresh(&self, request: RefreshRequest<'_>) -> Result<RefreshResponse, ClaudeUsageError> {
        let body = json!({
            "grant_type": "refresh_token",
            "refresh_token": request.refresh_token,
            "client_id": CLAUDE_CODE_CLIENT_ID,
            "scope": request.scopes.join(" "),
        })
        .to_string();
        let mut child = Command::new(curl_binary())
            .args([
                "--disable",
                "--silent",
                "--show-error",
                "--connect-timeout",
                "10",
                "--max-time",
                "30",
                "--request",
                "POST",
                "--header",
                "Content-Type: application/json",
                "--data-binary",
                "@-",
                "--write-out",
                "\n%{http_code}",
                TOKEN_ENDPOINT,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|_| ClaudeUsageError::NetworkRequestFailed)?;

        let Some(mut stdin) = child.stdin.take() else {
            return Err(ClaudeUsageError::NetworkRequestFailed);
        };
        if stdin.write_all(body.as_bytes()).is_err() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(ClaudeUsageError::NetworkRequestFailed);
        }
        drop(stdin);

        let output = child.wait_with_output().map_err(|_| ClaudeUsageError::NetworkRequestFailed)?;
        if !output.status.success() {
            return Err(ClaudeUsageError::NetworkRequestFailed);
        }

        let response = parse_curl_response(&output.stdout)?;
        match response.status {
            200 => serde_json::from_str(&response.body).map_err(|_| ClaudeUsageError::ApiRequestFailed),
            400 | 401 | 403 => Err(ClaudeUsageError::AuthenticationFailed),
            _ => Err(ClaudeUsageError::ApiRequestFailed),
        }
    }
}

struct CurlResponse {
    body: String,
    status: u16,
}

fn parse_curl_response(output: &[u8]) -> Result<CurlResponse, ClaudeUsageError> {
    let output = std::str::from_utf8(output).map_err(|_| ClaudeUsageError::ApiRequestFailed)?;
    let (body, status) = output.rsplit_once('\n').ok_or(ClaudeUsageError::ApiRequestFailed)?;
    let status = status.parse::<u16>().map_err(|_| ClaudeUsageError::ApiRequestFailed)?;
    Ok(CurlResponse {
        body: body.to_owned(),
        status,
    })
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
            let mut cached_usage = None;
            while is_started.load(Relaxed) {
                let result = poll_usage(token_source.as_ref(), usage_client.as_ref());
                if let Err(error) = result {
                    tracing::warn!("Claude Usage Provider poll failed: {error}");
                }
                let report = report_for_result(result, &mut cached_usage);
                if let Err(error) = host_to_device_sender.send(report.to_vec()) {
                    tracing::warn!("Claude Usage Provider failed to send data: {error}");
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

fn poll_usage(token_source: &dyn TokenSource, usage_client: &dyn UsageClient) -> Result<ClaudeUsage, ClaudeUsageError> {
    let token = token_source.load_token(false)?;
    let request = UsageRequest {
        endpoint: USAGE_ENDPOINT,
        oauth_beta_header: OAUTH_BETA_HEADER,
        token: &token,
    };

    match usage_client.fetch_usage(request) {
        Err(ClaudeUsageError::AuthenticationFailed) => {
            let refreshed_token = token_source.load_token(true)?;
            usage_client.fetch_usage(UsageRequest {
                endpoint: USAGE_ENDPOINT,
                oauth_beta_header: OAUTH_BETA_HEADER,
                token: &refreshed_token,
            })
        }
        result => result,
    }
}

fn report_for_result(result: Result<ClaudeUsage, ClaudeUsageError>, cached_usage: &mut Option<ClaudeUsage>) -> [u8; RAW_HID_REPORT_SIZE] {
    match result {
        Ok(usage) => {
            *cached_usage = Some(usage);
            usage.raw_hid_report()
        }
        Err(error) => raw_hid_report(*cached_usage, error.code()),
    }
}

fn raw_hid_report(usage: Option<ClaudeUsage>, error: ClaudeUsageErrorCode) -> [u8; RAW_HID_REPORT_SIZE] {
    let mut report = [0; RAW_HID_REPORT_SIZE];
    report[0] = DataType::ClaudeUsage as u8;
    report[6] = error as u8;

    let Some(usage) = usage else {
        return report;
    };

    report[1] = usage.session_remaining;
    report[2] = usage.weekly_remaining;
    report[3..5].copy_from_slice(&usage.reset_minutes.to_le_bytes());
    report[5] = FLAG_SAMPLE_VALID;
    if usage.extra_enabled {
        report[5] |= FLAG_EXTRA_ENABLED;
    }
    report[7] = usage.extra_remaining;
    report[8..10].copy_from_slice(&usage.extra_remaining_euros.to_le_bytes());
    report
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

fn token_needs_refresh(expires_at: i64, now_millis: i64) -> bool {
    expires_at.saturating_sub(now_millis) <= REFRESH_EARLY_MILLIS
}

fn credentials_from_json(credentials: &str) -> Result<ClaudeAiOauth, ClaudeUsageError> {
    let credentials: ClaudeCodeCredentials = serde_json::from_str(credentials).map_err(|_| ClaudeUsageError::InvalidCredentials)?;
    let oauth = credentials.claude_ai_oauth;
    OAuthToken::new(oauth.access_token.clone())?;
    if oauth.refresh_token.trim().is_empty() || oauth.scopes.iter().any(|scope| scope.trim().is_empty()) {
        return Err(ClaudeUsageError::InvalidCredentials);
    }
    Ok(oauth)
}

fn updated_credentials_json(credentials: &str, response: RefreshResponse, now_millis: i64) -> Result<(String, String), ClaudeUsageError> {
    let access_token = OAuthToken::new(response.access_token)?.0;
    if response.expires_in <= 0 {
        return Err(ClaudeUsageError::InvalidCredentials);
    }
    let expires_at = now_millis
        .checked_add(response.expires_in.checked_mul(1000).ok_or(ClaudeUsageError::InvalidCredentials)?)
        .ok_or(ClaudeUsageError::InvalidCredentials)?;

    let mut document: Value = serde_json::from_str(credentials).map_err(|_| ClaudeUsageError::InvalidCredentials)?;
    let oauth = document
        .get_mut("claudeAiOauth")
        .and_then(Value::as_object_mut)
        .ok_or(ClaudeUsageError::InvalidCredentials)?;
    oauth.insert("accessToken".to_owned(), Value::String(access_token.clone()));
    oauth.insert("expiresAt".to_owned(), Value::from(expires_at));

    if let Some(refresh_token) = response.refresh_token {
        if refresh_token.trim().is_empty() {
            return Err(ClaudeUsageError::InvalidCredentials);
        }
        oauth.insert("refreshToken".to_owned(), Value::String(refresh_token));
    }

    if let Some(refresh_expires_in) = response.refresh_token_expires_in {
        if refresh_expires_in <= 0 {
            return Err(ClaudeUsageError::InvalidCredentials);
        }
        let refresh_expires_at = now_millis
            .checked_add(refresh_expires_in.checked_mul(1000).ok_or(ClaudeUsageError::InvalidCredentials)?)
            .ok_or(ClaudeUsageError::InvalidCredentials)?;
        oauth.insert("refreshTokenExpiresAt".to_owned(), Value::from(refresh_expires_at));
    }

    if let Some(scope) = response.scope {
        let scopes: Vec<Value> = scope.split_whitespace().map(|value| Value::String(value.to_owned())).collect();
        if scopes.is_empty() {
            return Err(ClaudeUsageError::InvalidCredentials);
        }
        oauth.insert("scopes".to_owned(), Value::Array(scopes));
    }

    let updated = serde_json::to_string(&document).map_err(|_| ClaudeUsageError::InvalidCredentials)?;
    Ok((updated, access_token))
}

fn usage_from_response(response: &str, now: DateTime<Utc>) -> Result<ClaudeUsage, ClaudeUsageError> {
    let response: UsageResponse = serde_json::from_str(response).map_err(|_| ClaudeUsageError::InvalidUsageResponse)?;
    let five_hour = response.five_hour.ok_or(ClaudeUsageError::InvalidUsageResponse)?;
    let seven_day = response.seven_day.ok_or(ClaudeUsageError::InvalidUsageResponse)?;
    let reset_at = DateTime::parse_from_rfc3339(&five_hour.resets_at)
        .map_err(|_| ClaudeUsageError::InvalidUsageResponse)?
        .with_timezone(&Utc);
    let reset_minutes = reset_at.signed_duration_since(now).num_minutes().clamp(0, i64::from(u16::MAX)) as u16;
    let extra = extra_usage_from_response(response.extra_usage)?;

    Ok(ClaudeUsage {
        session_remaining: remaining_percent(five_hour.utilization)?,
        weekly_remaining: remaining_percent(seven_day.utilization)?,
        reset_minutes,
        extra_remaining: extra.remaining_percent,
        extra_remaining_euros: extra.remaining_euros,
        extra_enabled: extra.enabled,
    })
}

struct ExtraUsageRemaining {
    remaining_percent: u8,
    remaining_euros: u16,
    enabled: bool,
}

fn extra_usage_from_response(extra_usage: Option<ExtraUsageResponse>) -> Result<ExtraUsageRemaining, ClaudeUsageError> {
    let Some(extra_usage) = extra_usage else {
        return Ok(ExtraUsageRemaining {
            remaining_percent: 0,
            remaining_euros: 0,
            enabled: false,
        });
    };
    if !extra_usage.is_enabled {
        return Ok(ExtraUsageRemaining {
            remaining_percent: 0,
            remaining_euros: 0,
            enabled: false,
        });
    }
    if extra_usage.currency.as_deref() != Some("EUR") {
        return Err(ClaudeUsageError::InvalidUsageResponse);
    }

    let monthly_limit = extra_usage
        .monthly_limit
        .filter(|value| value.is_finite() && *value >= 0.0)
        .ok_or(ClaudeUsageError::InvalidUsageResponse)?;
    let used_credits = extra_usage
        .used_credits
        .filter(|value| value.is_finite() && *value >= 0.0)
        .ok_or(ClaudeUsageError::InvalidUsageResponse)?;
    let decimal_places = extra_usage
        .decimal_places
        .filter(|value| *value <= 6)
        .ok_or(ClaudeUsageError::InvalidUsageResponse)?;
    let remaining_minor = (monthly_limit - used_credits).max(0.0);
    let remaining_percent = if monthly_limit == 0.0 {
        0
    } else {
        ((remaining_minor / monthly_limit) * 100.0).round().clamp(0.0, 100.0) as u8
    };
    let units_per_euro = 10_f64.powi(decimal_places as i32);
    let remaining_euros = (remaining_minor / units_per_euro).round().clamp(0.0, f64::from(u16::MAX)) as u16;

    Ok(ExtraUsageRemaining {
        remaining_percent,
        remaining_euros,
        enabled: true,
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

#[cfg(target_os = "macos")]
fn keychain_service(configured_service: Option<&str>) -> &str {
    configured_service.unwrap_or(DEFAULT_KEYCHAIN_SERVICE)
}

#[cfg(target_os = "macos")]
fn load_keychain_credentials(service: &str) -> Result<String, ClaudeUsageError> {
    let output = Command::new("/usr/bin/security")
        .args(["find-generic-password", "-s", service, "-w"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .map_err(|_| ClaudeUsageError::CredentialsUnavailable)?;
    if !output.status.success() {
        return Err(ClaudeUsageError::CredentialsUnavailable);
    }

    String::from_utf8(output.stdout).map_err(|_| ClaudeUsageError::InvalidCredentials)
}

#[cfg(target_os = "macos")]
fn save_keychain_credentials(service: &str, credentials: &str) -> Result<(), ClaudeUsageError> {
    let account = keychain_account()?;
    let mut child = Command::new("/usr/bin/security")
        .args(["add-generic-password", "-U", "-a", &account, "-s", service, "-w"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| ClaudeUsageError::CredentialsUnavailable)?;

    let Some(mut stdin) = child.stdin.take() else {
        return Err(ClaudeUsageError::CredentialsUnavailable);
    };
    if stdin.write_all(credentials.as_bytes()).is_err() {
        let _ = child.kill();
        let _ = child.wait();
        return Err(ClaudeUsageError::CredentialsUnavailable);
    }
    drop(stdin);

    let status = child.wait().map_err(|_| ClaudeUsageError::CredentialsUnavailable)?;
    if status.success() {
        Ok(())
    } else {
        Err(ClaudeUsageError::CredentialsUnavailable)
    }
}

#[cfg(target_os = "macos")]
fn keychain_account() -> Result<String, ClaudeUsageError> {
    if let Ok(account) = std::env::var("USER") {
        if !account.trim().is_empty() {
            return Ok(account);
        }
    }

    let output = Command::new("/usr/bin/id")
        .arg("-un")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .map_err(|_| ClaudeUsageError::CredentialsUnavailable)?;
    if !output.status.success() {
        return Err(ClaudeUsageError::CredentialsUnavailable);
    }
    let account = String::from_utf8(output.stdout).map_err(|_| ClaudeUsageError::CredentialsUnavailable)?;
    let account = account.trim();
    if account.is_empty() {
        Err(ClaudeUsageError::CredentialsUnavailable)
    } else {
        Ok(account.to_owned())
    }
}

fn default_credentials_path() -> Result<PathBuf, ClaudeUsageError> {
    let home = std::env::var_os("HOME").ok_or(ClaudeUsageError::CredentialsUnavailable)?;
    Ok(PathBuf::from(home).join(".claude").join(".credentials.json"))
}

fn save_file_credentials(path: &Path, credentials: &str) -> Result<(), ClaudeUsageError> {
    let temporary_path = path.with_extension("tmp");
    fs::write(&temporary_path, credentials).map_err(|_| ClaudeUsageError::CredentialsUnavailable)?;
    if let Ok(metadata) = fs::metadata(path) {
        fs::set_permissions(&temporary_path, metadata.permissions()).map_err(|_| ClaudeUsageError::CredentialsUnavailable)?;
    }
    fs::rename(&temporary_path, path).map_err(|_| ClaudeUsageError::CredentialsUnavailable)
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

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClaudeCodeCredentials {
    claude_ai_oauth: ClaudeAiOauth,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClaudeAiOauth {
    access_token: String,
    refresh_token: String,
    expires_at: i64,
    refresh_token_expires_at: i64,
    scopes: Vec<String>,
}

#[derive(Deserialize)]
struct UsageResponse {
    five_hour: Option<UsageWindow>,
    seven_day: Option<UsageWindow>,
    extra_usage: Option<ExtraUsageResponse>,
}

#[derive(Deserialize)]
struct UsageWindow {
    utilization: f64,
    resets_at: String,
}

#[derive(Deserialize)]
struct ExtraUsageResponse {
    is_enabled: bool,
    monthly_limit: Option<f64>,
    used_credits: Option<f64>,
    currency: Option<String>,
    decimal_places: Option<u32>,
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use chrono::{TimeZone, Utc};

    use super::*;

    const SAMPLE_USAGE: ClaudeUsage = ClaudeUsage {
        session_remaining: 61,
        weekly_remaining: 73,
        reset_minutes: 75,
        extra_remaining: 40,
        extra_remaining_euros: 400,
        extra_enabled: true,
    };

    struct StaticTokenSource {
        calls: AtomicUsize,
    }

    impl StaticTokenSource {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl TokenSource for StaticTokenSource {
        fn load_token(&self, _force_refresh: bool) -> Result<OAuthToken, ClaudeUsageError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            OAuthToken::new("test-token".to_owned())
        }
    }

    struct StaticUsageClient;

    impl UsageClient for StaticUsageClient {
        fn fetch_usage(&self, request: UsageRequest<'_>) -> Result<ClaudeUsage, ClaudeUsageError> {
            assert_eq!(request.endpoint, USAGE_ENDPOINT);
            assert_eq!(request.oauth_beta_header, OAUTH_BETA_HEADER);
            Ok(SAMPLE_USAGE)
        }
    }

    struct AuthThenSuccessUsageClient {
        calls: AtomicUsize,
    }

    impl AuthThenSuccessUsageClient {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl UsageClient for AuthThenSuccessUsageClient {
        fn fetch_usage(&self, _request: UsageRequest<'_>) -> Result<ClaudeUsage, ClaudeUsageError> {
            if self.calls.fetch_add(1, Ordering::Relaxed) == 0 {
                Err(ClaudeUsageError::AuthenticationFailed)
            } else {
                Ok(SAMPLE_USAGE)
            }
        }
    }

    #[test]
    fn raw_hid_report_should_match_extended_wire_format() {
        let report = SAMPLE_USAGE.raw_hid_report();

        assert_eq!(
            report,
            [0xB2, 61, 73, 75, 0, 6, 0, 40, 144, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
        );
    }

    #[test]
    fn error_report_should_preserve_cached_usage() {
        let mut cached_usage = Some(SAMPLE_USAGE);

        let report = report_for_result(Err(ClaudeUsageError::NetworkRequestFailed), &mut cached_usage);

        assert_eq!((report[1], report[6], report[8], report[9]), (61, 2, 144, 1));
    }

    #[test]
    fn cold_error_report_should_clear_sample_valid_flag() {
        let mut cached_usage = None;

        let report = report_for_result(Err(ClaudeUsageError::AuthenticationFailed), &mut cached_usage);

        assert_eq!((report[5], report[6]), (0, 1));
    }

    #[test]
    fn poll_should_refresh_once_after_authentication_failure() {
        let token_source = StaticTokenSource::new();
        let usage_client = AuthThenSuccessUsageClient::new();

        let usage = poll_usage(&token_source, &usage_client).unwrap();

        assert_eq!(
            (
                usage,
                token_source.calls.load(Ordering::Relaxed),
                usage_client.calls.load(Ordering::Relaxed),
            ),
            (SAMPLE_USAGE, 2, 2)
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
                extra_remaining: 0,
                extra_remaining_euros: 0,
                extra_enabled: false,
            }
        );
    }

    #[test]
    fn usage_response_should_include_remaining_extra_budget() {
        let response = r#"{
            "five_hour": {"utilization": 15, "resets_at": "2026-07-22T13:30:00Z"},
            "seven_day": {"utilization": 54, "resets_at": "2026-07-23T01:00:00Z"},
            "extra_usage": {
                "is_enabled": true,
                "monthly_limit": 100000,
                "used_credits": 60002,
                "currency": "EUR",
                "decimal_places": 2
            }
        }"#;
        let now = Utc.with_ymd_and_hms(2026, 7, 22, 9, 30, 0).unwrap();

        let usage = usage_from_response(response, now).unwrap();

        assert_eq!((usage.extra_remaining, usage.extra_remaining_euros), (40, 400));
    }

    #[test]
    fn disabled_extra_usage_should_render_off() {
        let response = r#"{
            "five_hour": {"utilization": 15, "resets_at": "2026-07-22T13:30:00Z"},
            "seven_day": {"utilization": 54, "resets_at": "2026-07-23T01:00:00Z"},
            "extra_usage": {"is_enabled": false}
        }"#;
        let now = Utc.with_ymd_and_hms(2026, 7, 22, 9, 30, 0).unwrap();

        let usage = usage_from_response(response, now).unwrap();

        assert!(!usage.extra_enabled);
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
    fn token_should_refresh_when_expiring_within_five_minutes() {
        assert!(token_needs_refresh(1_000_000, 700_001));
    }

    #[test]
    fn token_should_not_refresh_before_early_window() {
        assert!(!token_needs_refresh(1_000_000, 699_999));
    }

    #[test]
    fn refreshed_credentials_should_rotate_tokens_and_preserve_other_fields() {
        let credentials = r#"{
            "claudeAiOauth": {
                "accessToken": "old-access",
                "refreshToken": "old-refresh",
                "expiresAt": 100,
                "refreshTokenExpiresAt": 200,
                "scopes": ["scope-a"]
            },
            "oauthAccount": {"emailAddress": "user@example.com"}
        }"#;
        let response = RefreshResponse {
            access_token: "new-access".to_owned(),
            refresh_token: Some("new-refresh".to_owned()),
            expires_in: 3600,
            refresh_token_expires_in: Some(7200),
            scope: Some("scope-a scope-b".to_owned()),
        };

        let (updated, access_token) = updated_credentials_json(credentials, response, 1_000).unwrap();
        let updated: Value = serde_json::from_str(&updated).unwrap();

        assert_eq!(
            (
                access_token,
                updated["claudeAiOauth"]["refreshToken"].as_str(),
                updated["claudeAiOauth"]["expiresAt"].as_i64(),
                updated["oauthAccount"]["emailAddress"].as_str(),
            ),
            (
                "new-access".to_owned(),
                Some("new-refresh"),
                Some(3_601_000),
                Some("user@example.com"),
            )
        );
    }

    #[test]
    fn poll_should_use_injected_token_and_usage_sources() {
        let token_source = StaticTokenSource::new();

        let usage = poll_usage(&token_source, &StaticUsageClient).unwrap();

        assert_eq!(usage, SAMPLE_USAGE);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn keychain_service_should_default_to_claude_code_credentials() {
        assert_eq!(keychain_service(None), DEFAULT_KEYCHAIN_SERVICE);
    }

}
