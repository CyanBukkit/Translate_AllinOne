package com.cedarxuesong.translate_allinone.utils.llmapi.openclaw;

import com.cedarxuesong.translate_allinone.Translate_AllinOne;
import com.cedarxuesong.translate_allinone.utils.llmapi.LLMApiException;
import com.cedarxuesong.translate_allinone.utils.llmapi.ProviderSettings;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * OpenClaw 远程 API 客户端。
 * 通过 HTTP/WebSocket Gateway 连接到远程 OpenClaw 设备进行翻译请求。
 * OpenClaw Gateway 提供类似 OpenAI Chat Completions 的 API 接口。
 */
public class OpenClawClient {

    private static final Gson GSON = new Gson();
    private final HttpClient httpClient;
    private final ProviderSettings.OpenClawSettings settings;

    public OpenClawClient(ProviderSettings.OpenClawSettings settings) {
        this.httpClient = HttpClient.newHttpClient();
        this.settings = settings;
    }

    /**
     * 构建请求体，添加自定义参数。
     */
    private String buildRequestBody(Object request) {
        JsonObject jsonObject = GSON.toJsonTree(request).getAsJsonObject();

        if (settings.customParameters() != null && !settings.customParameters().isEmpty()) {
            settings.customParameters().forEach((key, value) ->
                    jsonObject.add(key, GSON.toJsonTree(value))
            );
        }
        return GSON.toJson(jsonObject);
    }

    /**
     * 发送非流式请求到 OpenClaw Gateway。
     * @param request 请求对象 (使用 OpenAI 兼容格式)
     * @return 包含完整响应的 Future
     */
    public CompletableFuture<String> getChatCompletion(OpenClawChatRequest request) {
        request.setStream(false);
        String requestBody = buildRequestBody(request);
        String requestSummary = summarizeRequest(request, requestBody);
        
        // OpenClaw Gateway URL 处理 - 支持 wss:// 和 https:// 格式
        String baseUrl = settings.gatewayUrl();
        if (baseUrl == null || baseUrl.isBlank()) {
            throw new IllegalArgumentException("OpenClaw Gateway URL 不能为空");
        }
        
        // 将 wss:// 转换为 https:// (HTTP API)
        if (baseUrl.startsWith("wss://")) {
            baseUrl = "https://" + baseUrl.substring(6);
        } else if (baseUrl.startsWith("ws://")) {
            baseUrl = "http://" + baseUrl.substring(5);
        }
        
        // 移除末尾的斜杠
        while (baseUrl.endsWith("/")) {
            baseUrl = baseUrl.substring(0, baseUrl.length() - 1);
        }
        String endpoint = baseUrl + "/v1/chat/completions";
        
        Translate_AllinOne.LOGGER.info("OpenClaw request: endpoint={}, body={}", endpoint, requestBody);

        HttpRequest httpRequest = HttpRequest.newBuilder()
                .uri(URI.create(endpoint))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + settings.apiKey())
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();

        CompletableFuture<String> future = httpClient.sendAsync(httpRequest, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() != 200) {
                        String message = resolveApiErrorMessage(response.body());
                        logApiError(response, endpoint, requestSummary, response.body());
                        throw new LLMApiException("OpenClaw API returned error: " + response.statusCode() + " - " + message);
                    }
                    return parseChatCompletionResponse(response.body());
                });

        return future.whenComplete((ignored, throwable) -> {
            if (throwable == null) {
                return;
            }
            Throwable root = unwrapThrowable(throwable);
            if (root instanceof LLMApiException) {
                return;
            }
            Translate_AllinOne.LOGGER.error("OpenClaw request failed before receiving valid API response. endpoint={} summary={}",
                    endpoint, requestSummary, root);
        });
    }

    /**
     * 发送流式请求到 OpenClaw Gateway。
     * @param request 请求对象
     * @return 响应文本块的 Stream
     */
    public Stream<String> getStreamingChatCompletion(OpenClawChatRequest request) {
        request.setStream(true);
        String requestBody = buildRequestBody(request);
        String requestSummary = summarizeRequest(request, requestBody);
        
        // OpenClaw Gateway URL 处理 - 支持 wss:// 和 https:// 格式
        String baseUrl = settings.gatewayUrl();
        if (baseUrl == null || baseUrl.isBlank()) {
            throw new IllegalArgumentException("OpenClaw Gateway URL 不能为空");
        }
        
        // 将 wss:// 转换为 https:// (HTTP API)
        if (baseUrl.startsWith("wss://")) {
            baseUrl = "https://" + baseUrl.substring(6);
        } else if (baseUrl.startsWith("ws://")) {
            baseUrl = "http://" + baseUrl.substring(5);
        }
        
        // 移除末尾的斜杠
        while (baseUrl.endsWith("/")) {
            baseUrl = baseUrl.substring(0, baseUrl.length() - 1);
        }
        String endpoint = baseUrl + "/v1/chat/completions";

        HttpRequest httpRequest = HttpRequest.newBuilder()
                .uri(URI.create(endpoint))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + settings.apiKey())
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();

        try {
            HttpResponse<Stream<String>> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofLines());

            if (response.statusCode() != 200) {
                String errorBody = response.body().collect(Collectors.joining("\n"));
                String message = resolveApiErrorMessage(errorBody);
                logApiError(response, endpoint, requestSummary, errorBody);
                throw new LLMApiException("OpenClaw API returned error: " + response.statusCode() + " - " + message);
            }

            return response.body()
                    .filter(line -> line.startsWith("data: "))
                    .map(line -> line.substring("data: ".length()))
                    .filter(data -> !data.equals("[DONE]"))
                    .map(data -> parseStreamingChunk(data))
                    .filter(chunk -> chunk != null && !chunk.isEmpty());

        } catch (Exception e) {
            if (e instanceof LLMApiException llmApiException) {
                throw llmApiException;
            }
            Translate_AllinOne.LOGGER.error("OpenClaw streaming request failed. endpoint={} summary={}", endpoint, requestSummary, e);
            throw new LLMApiException("请求 OpenClaw 流式 API 时出错", e);
        }
    }

    /**
     * 解析非流式响应的内容。
     */
    private String parseChatCompletionResponse(String responseBody) {
        try {
            JsonObject json = GSON.fromJson(responseBody, JsonObject.class);
            
            if (json.has("choices") && json.get("choices").isJsonArray()) {
                JsonArray choices = json.getAsJsonArray("choices");
                if (!choices.isEmpty()) {
                    JsonObject firstChoice = choices.get(0).getAsJsonObject();
                    if (firstChoice.has("message") && firstChoice.get("message").isJsonObject()) {
                        JsonObject message = firstChoice.getAsJsonObject("message");
                        if (message.has("content") && !message.get("content").isJsonNull()) {
                            return message.get("content").getAsString();
                        }
                    }
                }
            }
            
            // 备用：尝试从其他字段获取内容
            if (json.has("content") && !json.get("content").isJsonNull()) {
                return json.get("content").getAsString();
            }
            
            throw new LLMApiException("无法从 OpenClaw 响应中解析内容");
        } catch (LLMApiException e) {
            throw e;
        } catch (Exception e) {
            throw new LLMApiException("解析 OpenClaw 响应失败: " + e.getMessage(), e);
        }
    }

    /**
     * 解析流式响应的内容块。
     */
    private String parseStreamingChunk(String data) {
        try {
            JsonObject chunk = GSON.fromJson(data, JsonObject.class);
            
            if (chunk.has("choices") && chunk.get("choices").isJsonArray()) {
                JsonArray choices = chunk.getAsJsonArray("choices");
                if (!choices.isEmpty()) {
                    JsonObject firstChoice = choices.get(0).getAsJsonObject();
                    if (firstChoice.has("delta") && firstChoice.get("delta").isJsonObject()) {
                        JsonObject delta = firstChoice.getAsJsonObject("delta");
                        if (delta.has("content") && !delta.get("content").isJsonNull()) {
                            return delta.get("content").getAsString();
                        }
                    }
                }
            }
            
            return "";
        } catch (Exception e) {
            return "";
        }
    }

    /**
     * 解析 API 错误信息。
     */
    private String resolveApiErrorMessage(String responseBody) {
        try {
            JsonObject errorJson = GSON.fromJson(responseBody, JsonObject.class);
            
            // 尝试多种错误格式
            if (errorJson.has("error") && errorJson.get("error").isJsonObject()) {
                JsonObject error = errorJson.getAsJsonObject("error");
                if (error.has("message") && !error.get("message").isJsonNull()) {
                    return error.get("message").getAsString();
                }
            }
            
            if (errorJson.has("message") && !errorJson.get("message").isJsonNull()) {
                return errorJson.get("message").getAsString();
            }
            
            if (errorJson.has("detail") && !errorJson.get("detail").isJsonNull()) {
                return errorJson.get("detail").getAsString();
            }
            
            return "Unknown OpenClaw API error";
        } catch (Exception e) {
            return "Unknown OpenClaw API error";
        }
    }

    /**
     * 生成请求摘要用于日志记录。
     */
    private String summarizeRequest(OpenClawChatRequest request, String requestBody) {
        int bodyLength = requestBody == null ? 0 : requestBody.length();
        String bodyHash = requestBody == null ? "0" : Integer.toHexString(requestBody.hashCode());
        String customKeys = settings.customParameters() == null ? "[]" : settings.customParameters().keySet().toString();
        
        return "model=" + safeText(request.getModel())
                + ", stream=" + request.isStream()
                + ", temperature=" + request.getTemperature()
                + ", messages=" + (request.getMessages() != null ? request.getMessages().size() : 0)
                + ", custom_keys=" + customKeys
                + ", body_len=" + bodyLength
                + ", body_hash=" + bodyHash;
    }

    /**
     * 记录 API 错误。
     */
    private void logApiError(HttpResponse<?> response, String endpoint, String requestSummary, String responseBody) {
        String requestId = response.headers().firstValue("x-request-id").orElse("-");
        Translate_AllinOne.LOGGER.error(
                "OpenClaw API returned non-200. status={} requestId={} endpoint={} summary={} response_body={}"
                , response.statusCode()
                , requestId
                , endpoint
                , requestSummary
                , truncate(normalizeWhitespace(responseBody), 1200)
        );
    }

    /**
     * 规范化空白字符。
     */
    private String normalizeWhitespace(String value) {
        if (value == null) {
            return "";
        }
        return value.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').trim();
    }

    /**
     * 安全获取字符串值。
     */
    private String safeText(String value) {
        return value == null ? "" : value;
    }

    /**
     * 截断字符串到最大长度。
     */
    private String truncate(String value, int maxLength) {
        String safeValue = safeText(value);
        if (safeValue.length() <= maxLength) {
            return safeValue;
        }
        return safeValue.substring(0, Math.max(0, maxLength - 3)) + "...";
    }

    /**
     * 解包异常。
     */
    private Throwable unwrapThrowable(Throwable throwable) {
        Throwable current = throwable;
        while (current instanceof java.util.concurrent.CompletionException && current.getCause() != null) {
            current = current.getCause();
        }
        return current;
    }
}
