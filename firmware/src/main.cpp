#include <Arduino.h>
#include <ESPmDNS.h>
#include <WiFi.h>
#include <esp_camera.h>
#include <esp_http_server.h>

#include "wifi_secrets.h"

namespace {
constexpr int kPwdnPin = 32;
constexpr int kResetPin = -1;
constexpr int kXclkPin = 0;
constexpr int kSiodPin = 26;
constexpr int kSiocPin = 27;
constexpr int kY9Pin = 35;
constexpr int kY8Pin = 34;
constexpr int kY7Pin = 39;
constexpr int kY6Pin = 36;
constexpr int kY5Pin = 21;
constexpr int kY4Pin = 19;
constexpr int kY3Pin = 18;
constexpr int kY2Pin = 5;
constexpr int kVsyncPin = 25;
constexpr int kHrefPin = 23;
constexpr int kPclkPin = 22;
constexpr uint32_t kWifiRetryIntervalMs = 5000;
constexpr uint32_t kCameraFailureRestartThreshold = 10;
constexpr char kStreamContentType[] = "multipart/x-mixed-replace;boundary=frame";
constexpr char kStreamBoundary[] = "\r\n--frame\r\n";
constexpr char kStreamPart[] = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

httpd_handle_t http_server = nullptr;
volatile uint64_t frames_sent = 0;
volatile uint32_t capture_failures = 0;
volatile bool restart_requested = false;
uint32_t last_wifi_attempt_ms = 0;
uint16_t sensor_pid = 0;

const char* sensorName(const uint16_t pid) {
  switch (pid) {
    case OV3660_PID:
      return "OV3660";
    case OV2640_PID:
      return "OV2640";
    default:
      return "UNKNOWN";
  }
}

esp_err_t statusHandler(httpd_req_t* request) {
  char body[384];
  const bool connected = WiFi.status() == WL_CONNECTED;
  const sensor_t* sensor = esp_camera_sensor_get();
  const int frame_size = sensor == nullptr ? -1 : sensor->status.framesize;
  const int quality = sensor == nullptr ? -1 : sensor->status.quality;

  const int written = snprintf(
      body,
      sizeof(body),
      "{\"sensor\":\"%s\",\"sensor_pid\":%u,\"ov3660\":%s,"
      "\"wifi_connected\":%s,\"ip\":\"%s\",\"rssi\":%d,"
      "\"uptime_ms\":%lu,\"frames_sent\":%llu,\"capture_failures\":%lu,"
      "\"frame_size\":%d,\"jpeg_quality\":%d}",
      sensorName(sensor_pid),
      sensor_pid,
      sensor_pid == OV3660_PID ? "true" : "false",
      connected ? "true" : "false",
      connected ? WiFi.localIP().toString().c_str() : "",
      connected ? WiFi.RSSI() : 0,
      millis(),
      frames_sent,
      capture_failures,
      frame_size,
      quality);

  if (written < 0 || static_cast<size_t>(written) >= sizeof(body)) {
    return httpd_resp_send_err(request, HTTPD_500_INTERNAL_SERVER_ERROR, "status overflow");
  }

  httpd_resp_set_type(request, "application/json");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  httpd_resp_set_hdr(request, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(request, body, written);
}

esp_err_t streamHandler(httpd_req_t* request) {
  esp_err_t result = httpd_resp_set_type(request, kStreamContentType);
  if (result != ESP_OK) {
    return result;
  }
  httpd_resp_set_hdr(request, "Cache-Control", "no-store, no-cache, must-revalidate");
  httpd_resp_set_hdr(request, "Access-Control-Allow-Origin", "*");

  uint32_t consecutive_failures = 0;
  char part_header[64];

  while (true) {
    camera_fb_t* frame = esp_camera_fb_get();
    if (frame == nullptr) {
      ++capture_failures;
      ++consecutive_failures;
      if (consecutive_failures >= kCameraFailureRestartThreshold) {
        restart_requested = true;
        return ESP_FAIL;
      }
      vTaskDelay(pdMS_TO_TICKS(20));
      continue;
    }

    consecutive_failures = 0;
    const int header_length = snprintf(
        part_header, sizeof(part_header), kStreamPart, static_cast<unsigned int>(frame->len));
    if (header_length <= 0 || static_cast<size_t>(header_length) >= sizeof(part_header)) {
      esp_camera_fb_return(frame);
      return ESP_FAIL;
    }

    result = httpd_resp_send_chunk(request, kStreamBoundary, strlen(kStreamBoundary));
    if (result == ESP_OK) {
      result = httpd_resp_send_chunk(request, part_header, header_length);
    }
    if (result == ESP_OK) {
      result = httpd_resp_send_chunk(
          request, reinterpret_cast<const char*>(frame->buf), frame->len);
    }
    esp_camera_fb_return(frame);

    if (result != ESP_OK) {
      return result;
    }
    ++frames_sent;
  }
}

bool initializeCamera() {
  camera_config_t config{};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = kY2Pin;
  config.pin_d1 = kY3Pin;
  config.pin_d2 = kY4Pin;
  config.pin_d3 = kY5Pin;
  config.pin_d4 = kY6Pin;
  config.pin_d5 = kY7Pin;
  config.pin_d6 = kY8Pin;
  config.pin_d7 = kY9Pin;
  config.pin_xclk = kXclkPin;
  config.pin_pclk = kPclkPin;
  config.pin_vsync = kVsyncPin;
  config.pin_href = kHrefPin;
  config.pin_sccb_sda = kSiodPin;
  config.pin_sccb_scl = kSiocPin;
  config.pin_pwdn = kPwdnPin;
  config.pin_reset = kResetPin;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 15;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  const esp_err_t result = esp_camera_init(&config);
  if (result != ESP_OK) {
    Serial.printf("Camera initialization failed: 0x%x\n", result);
    return false;
  }

  sensor_t* sensor = esp_camera_sensor_get();
  if (sensor == nullptr) {
    Serial.println("Camera sensor descriptor unavailable");
    esp_camera_deinit();
    return false;
  }

  sensor_pid = sensor->id.PID;
  Serial.printf("Detected camera sensor: %s (PID=0x%04x)\n", sensorName(sensor_pid), sensor_pid);
  if (sensor_pid != OV3660_PID) {
    Serial.println("Expected OV3660; refusing to start HTTP service");
    esp_camera_deinit();
    return false;
  }

  sensor->set_vflip(sensor, 1);
  sensor->set_brightness(sensor, 1);
  sensor->set_saturation(sensor, -2);
  sensor->set_whitebal(sensor, 1);
  sensor->set_exposure_ctrl(sensor, 1);
  return true;
}

bool startHttpServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.max_open_sockets = 4;
  config.lru_purge_enable = true;
  config.stack_size = 8192;

  if (httpd_start(&http_server, &config) != ESP_OK) {
    Serial.println("HTTP server failed to start");
    return false;
  }

  const httpd_uri_t status_uri{
      .uri = "/status",
      .method = HTTP_GET,
      .handler = statusHandler,
      .user_ctx = nullptr,
  };
  const httpd_uri_t stream_uri{
      .uri = "/stream",
      .method = HTTP_GET,
      .handler = streamHandler,
      .user_ctx = nullptr,
  };

  if (httpd_register_uri_handler(http_server, &status_uri) != ESP_OK ||
      httpd_register_uri_handler(http_server, &stream_uri) != ESP_OK) {
    httpd_stop(http_server);
    http_server = nullptr;
    Serial.println("HTTP route registration failed");
    return false;
  }
  return true;
}

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.printf("Connecting to Wi-Fi: %s", WIFI_SSID);
  const uint32_t started_at = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - started_at < 20000) {
    delay(500);
    Serial.print('.');
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("Wi-Fi connected: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("Wi-Fi connection timed out; background retries enabled");
  }
  last_wifi_attempt_ms = millis();
}
}  // namespace

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  delay(500);

  if (!initializeCamera()) {
    delay(3000);
    ESP.restart();
  }

  connectWifi();
  if (!startHttpServer()) {
    delay(3000);
    ESP.restart();
  }

  if (WiFi.status() == WL_CONNECTED && MDNS.begin("esp32cam")) {
    MDNS.addService("http", "tcp", 80);
    Serial.println("mDNS available at http://esp32cam.local");
  }
  Serial.println("OV3660 service ready: /status and /stream");
}

void loop() {
  if (restart_requested) {
    Serial.println("Repeated camera capture failures; restarting");
    delay(250);
    ESP.restart();
  }

  if (WiFi.status() != WL_CONNECTED && millis() - last_wifi_attempt_ms >= kWifiRetryIntervalMs) {
    last_wifi_attempt_ms = millis();
    Serial.println("Wi-Fi disconnected; reconnecting");
    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  }
  delay(100);
}

