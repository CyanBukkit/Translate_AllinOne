package com.cedarxuesong.translate_allinone.utils.config.pojos;

public enum ApiProviderType {
    OPENAI_COMPAT,
    OPENAI_RESPONSE,
    OLLAMA,
    /**
     * OpenClaw 远程 API 供应商类型。
     * 通过 Gateway 连接远程 OpenClaw 实例进行翻译请求。
     */
    OPENCLAW
}
