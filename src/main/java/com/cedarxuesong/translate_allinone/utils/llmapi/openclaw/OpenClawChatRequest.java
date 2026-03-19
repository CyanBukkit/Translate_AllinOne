package com.cedarxuesong.translate_allinone.utils.llmapi.openclaw;

import com.google.gson.annotations.SerializedName;

import java.util.List;

/**
 * OpenClaw Chat 请求对象。
 * 使用 OpenAI 兼容的请求格式。
 */
public class OpenClawChatRequest {

    private String model;
    private List<Message> messages;
    private Double temperature;

    @SerializedName("stream")
    private Boolean stream;

    @SerializedName("format")
    private String format;

    public OpenClawChatRequest() {
    }

    public OpenClawChatRequest(String model, List<Message> messages, Double temperature, Boolean stream) {
        this.model = model;
        this.messages = messages;
        this.temperature = temperature;
        this.stream = stream;
    }

    // Getter and Setter methods
    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public List<Message> getMessages() {
        return messages;
    }

    public void setMessages(List<Message> messages) {
        this.messages = messages;
    }

    public Double getTemperature() {
        return temperature;
    }

    public void setTemperature(Double temperature) {
        this.temperature = temperature;
    }

    public Boolean isStream() {
        return stream;
    }

    public void setStream(Boolean stream) {
        this.stream = stream;
    }

    public String getFormat() {
        return format;
    }

    public void setFormat(String format) {
        this.format = format;
    }

    /**
     * 消息对象。
     */
    public static class Message {
        private String role;
        private String content;

        public Message() {
        }

        public Message(String role, String content) {
            this.role = role;
            this.content = content;
        }

        public String getRole() {
            return role;
        }

        public void setRole(String role) {
            this.role = role;
        }

        public String getContent() {
            return content;
        }

        public void setContent(String content) {
            this.content = content;
        }
    }
}
