import { execSync } from "node:child_process";
import type {
	MessageCreateParamsStreaming,
	MessageParam,
	RawMessageStreamEvent,
	Tool as AnthropicTool,
} from "@anthropic-ai/sdk/resources/messages.js";
import { AnthropicVertex } from "@anthropic-ai/vertex-sdk";
import { GoogleAuth } from "google-auth-library";
import type { AuthClient } from "google-auth-library";
import {
	type Api,
	type AssistantMessage,
	calculateCost,
	type CacheRetention,
	type Context,
	type ImageContent,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type StopReason,
	type StreamFunction,
	type StreamOptions,
	type TextContent,
	type ThinkingContent,
	type ThinkingBudgets,
	type ThinkingLevel,
	type Tool,
	type ToolCall,
	type ToolResultMessage,
	createAssistantMessageEventStream,
} from "@mariozechner/pi-ai";
import type { ExtensionAPI, ProviderModelConfig } from "@mariozechner/pi-coding-agent";

// Inlined from @mariozechner/pi-ai/dist/providers/transform-messages.js.
//
// Every Anthropic-shaped provider in pi-ai (anthropic, bedrock, google-*) runs this
// pre-pass before serializing messages. It does two things this provider needs:
//   1. Drop assistant messages with stopReason "error" / "aborted" entirely, so an
//      incomplete turn (e.g. a tool_use the user cancelled mid-call) never replays.
//   2. Insert synthetic "No result provided" tool_result messages for any tool_use
//      that lacks a matching tool_result before the next user/assistant message.
//
// Without this, Anthropic Vertex rejects the request with:
//   `tool_use` ids were found without `tool_result` blocks immediately after.
//
// We inline rather than deep-import because the pi extension loader resolves the
// import against the package's `main` file path, which mangles `dist/...` subpaths.
function transformMessages<TApi extends Api>(
	messages: Message[],
	model: Model<TApi>,
	normalizeToolCallId?: (id: string, model: Model<TApi>, source: AssistantMessage) => string,
): Message[] {
	const toolCallIdMap = new Map<string, string>();

	const transformed = messages.map((msg) => {
		if (msg.role === "user") return msg;

		if (msg.role === "toolResult") {
			const normalizedId = toolCallIdMap.get(msg.toolCallId);
			if (normalizedId && normalizedId !== msg.toolCallId) {
				return { ...msg, toolCallId: normalizedId };
			}
			return msg;
		}

		if (msg.role === "assistant") {
			const assistantMsg = msg as AssistantMessage;
			const isSameModel =
				assistantMsg.provider === model.provider &&
				assistantMsg.api === model.api &&
				assistantMsg.model === model.id;

			const transformedContent = assistantMsg.content.flatMap((block) => {
				if (block.type === "thinking") {
					if (isSameModel && block.thinkingSignature) return block;
					if (!block.thinking || block.thinking.trim() === "") return [];
					if (isSameModel) return block;
					return { type: "text" as const, text: block.thinking };
				}
				if (block.type === "text") {
					if (isSameModel) return block;
					return { type: "text" as const, text: block.text };
				}
				if (block.type === "toolCall") {
					const toolCall = block as ToolCall;
					let normalized: ToolCall = toolCall;
					if (!isSameModel && toolCall.thoughtSignature) {
						normalized = { ...toolCall };
						delete (normalized as { thoughtSignature?: string }).thoughtSignature;
					}
					if (!isSameModel && normalizeToolCallId) {
						const normalizedId = normalizeToolCallId(toolCall.id, model, assistantMsg);
						if (normalizedId !== toolCall.id) {
							toolCallIdMap.set(toolCall.id, normalizedId);
							normalized = { ...normalized, id: normalizedId };
						}
					}
					return normalized;
				}
				return block;
			});

			return { ...assistantMsg, content: transformedContent };
		}
		return msg;
	});

	const result: Message[] = [];
	let pendingToolCalls: ToolCall[] = [];
	let existingToolResultIds = new Set<string>();

	const flushOrphans = () => {
		for (const tc of pendingToolCalls) {
			if (!existingToolResultIds.has(tc.id)) {
				result.push({
					role: "toolResult",
					toolCallId: tc.id,
					toolName: tc.name,
					content: [{ type: "text", text: "No result provided" }],
					isError: true,
					timestamp: Date.now(),
				} as ToolResultMessage);
			}
		}
		pendingToolCalls = [];
		existingToolResultIds = new Set();
	};

	for (const msg of transformed) {
		if (msg.role === "assistant") {
			if (pendingToolCalls.length > 0) flushOrphans();

			const assistantMsg = msg as AssistantMessage;
			if (assistantMsg.stopReason === "error" || assistantMsg.stopReason === "aborted") {
				continue;
			}

			const toolCalls = assistantMsg.content.filter((b) => b.type === "toolCall") as ToolCall[];
			if (toolCalls.length > 0) {
				pendingToolCalls = toolCalls;
				existingToolResultIds = new Set();
			}
			result.push(msg);
		} else if (msg.role === "toolResult") {
			existingToolResultIds.add(msg.toolCallId);
			result.push(msg);
		} else if (msg.role === "user") {
			if (pendingToolCalls.length > 0) flushOrphans();
			result.push(msg);
		} else {
			result.push(msg);
		}
	}

	return result;
}

const DEFAULT_REGION = "us-east5";
const BASE_URL = "https://{region}-aiplatform.googleapis.com";

const MODELS: ProviderModelConfig[] = [
	{
		id: "claude-sonnet-4-5@20250929",
		name: "Claude Sonnet 4.5 (Vertex AI)",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
		contextWindow: 200000,
		maxTokens: 64000,
	},
	{
		id: "claude-opus-4-5@20251101",
		name: "Claude Opus 4.5 (Vertex AI)",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 15, output: 75, cacheRead: 0.5, cacheWrite: 6.25 },
		contextWindow: 200000,
		maxTokens: 32000,
	},
	{
		id: "claude-fable-5@default",
		name: "Claude Fable 5 (Vertex AI)",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 10, output: 50, cacheRead: 1, cacheWrite: 12.5 },
		contextWindow: 1000000,
		maxTokens: 128000,
	},
	{
		id: "claude-opus-5@default",
		name: "Claude Opus 5 (Vertex AI)",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
		contextWindow: 1000000,
		maxTokens: 128000,
	},
	{
		id: "claude-opus-4-8@default",
		name: "Claude Opus 4.8 (Vertex AI)",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
		contextWindow: 1000000,
		maxTokens: 128000,
	},
	{
		id: "claude-opus-4-7@default",
		name: "Claude Opus 4.7 (Vertex AI)",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
		contextWindow: 1000000,
		maxTokens: 128000,
	},
	{
		id: "claude-opus-4-6@default",
		name: "Claude Opus 4.6 (Vertex AI)",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
		contextWindow: 1000000,
		maxTokens: 128000,
	},
	{
		id: "claude-haiku-4-5@20251001",
		name: "Claude Haiku 4.5 (Vertex AI)",
		reasoning: false,
		input: ["text", "image"],
		cost: { input: 1, output: 5, cacheRead: 0.1, cacheWrite: 1.25 },
		contextWindow: 200000,
		maxTokens: 8192,
	},
	{
		id: "claude-sonnet-4-20250514",
		name: "Claude Sonnet 4 (Vertex AI)",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
		contextWindow: 200000,
		maxTokens: 64000,
	},
	{
		id: "claude-3-5-sonnet-v2@20241022",
		name: "Claude 3.5 Sonnet v2 (Vertex AI)",
		reasoning: false,
		input: ["text", "image"],
		cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
		contextWindow: 200000,
		maxTokens: 8192,
	},
	{
		id: "claude-3-5-haiku@20241022",
		name: "Claude 3.5 Haiku (Vertex AI)",
		reasoning: false,
		input: ["text", "image"],
		cost: { input: 1, output: 5, cacheRead: 0.1, cacheWrite: 1.25 },
		contextWindow: 200000,
		maxTokens: 8192,
	},
	{
		id: "claude-3-opus@20240229",
		name: "Claude 3 Opus (Vertex AI)",
		reasoning: false,
		input: ["text", "image"],
		cost: { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
		contextWindow: 200000,
		maxTokens: 4096,
	},
	{
		id: "claude-3-haiku@20240307",
		name: "Claude 3 Haiku (Vertex AI)",
		reasoning: false,
		input: ["text", "image"],
		cost: { input: 0.25, output: 1.25, cacheRead: 0.025, cacheWrite: 0.3125 },
		contextWindow: 200000,
		maxTokens: 4096,
	},
];

type AnthropicVertexEffort = "low" | "medium" | "high" | "max";

interface AnthropicVertexOptions extends StreamOptions {
	thinkingEnabled?: boolean;
	thinkingBudgetTokens?: number;
	effort?: AnthropicVertexEffort;
	interleavedThinking?: boolean;
	toolChoice?: "auto" | "any" | "none" | { type: "tool"; name: string };
	project?: string;
	region?: string;
}

interface StreamingBlockBase {
	index: number;
}

type ToolCallStreamingBlock = ToolCall & {
	partialJson: string;
	index: number;
};

type AnthropicStreamingBlock =
	| (TextContent & StreamingBlockBase)
	| (ThinkingContent & StreamingBlockBase)
	| ToolCallStreamingBlock;

function sanitizeSurrogates(text: string): string {
	return text.replace(/[\uD800-\uDFFF]/g, "\uFFFD");
}

function mergeHeaders(...sources: Array<Record<string, string> | undefined>): Record<string, string> {
	const merged: Record<string, string> = {};
	for (const source of sources) {
		if (source) {
			Object.assign(merged, source);
		}
	}
	return merged;
}

function supportsAdaptiveThinking(modelId: string): boolean {
	return ["opus-4-6", "opus-4-7", "opus-4-8", "opus-5"].some((id) => modelId.includes(id));
}

function mapThinkingLevelToEffort(level: SimpleStreamOptions["reasoning"]): AnthropicVertexEffort {
	switch (level) {
		case "minimal":
		case "low":
			return "low";
		case "medium":
			return "medium";
		case "high":
			return "high";
		case "xhigh":
			return "max";
		default:
			return "high";
	}
}

function mapStopReason(reason: string): StopReason {
	switch (reason) {
		case "end_turn":
		case "pause_turn":
		case "stop_sequence":
			return "stop";
		case "max_tokens":
			return "length";
		case "tool_use":
			return "toolUse";
		case "refusal":
		case "sensitive":
			return "error";
		default:
			return "error";
	}
}

function resolveCacheRetention(cacheRetention?: CacheRetention): CacheRetention {
	if (cacheRetention) {
		return cacheRetention;
	}
	if (typeof process !== "undefined" && process.env.PI_CACHE_RETENTION === "long") {
		return "long";
	}
	return "short";
}

function getCacheControl(
	baseUrl: string,
	cacheRetention?: CacheRetention,
): { retention: CacheRetention; cacheControl?: { type: "ephemeral"; ttl?: "1h" } } {
	const retention = resolveCacheRetention(cacheRetention);
	if (retention === "none") {
		return { retention };
	}
	const ttl = retention === "long" && baseUrl.includes("api.anthropic.com") ? "1h" : undefined;
	return {
		retention,
		cacheControl: { type: "ephemeral", ...(ttl ? { ttl } : {}) },
	};
}

function convertContentBlocks(content: Array<TextContent | ImageContent>) {
	const hasImages = content.some((c) => c.type === "image");
	if (!hasImages) {
		return sanitizeSurrogates(content.map((c) => c.type === "text" ? c.text : "").join("\n"));
	}

	const blocks = content.map((block) => {
		if (block.type === "text") {
			return {
				type: "text" as const,
				text: sanitizeSurrogates(block.text),
			};
		}
		return {
			type: "image" as const,
			source: {
				type: "base64" as const,
				media_type: block.mimeType as "image/jpeg" | "image/png" | "image/gif" | "image/webp",
				data: block.data,
			},
		};
	});

	if (!blocks.some((b) => b.type === "text")) {
		blocks.unshift({ type: "text", text: "(see attached image)" });
	}

	return blocks;
}

function convertMessages(
	messages: Message[],
	model: Model<Api>,
	cacheControl?: { type: "ephemeral"; ttl?: "1h" },
): MessageParam[] {
	const params: MessageParam[] = [];
	// Pre-pass: drop aborted/errored assistant turns and inject synthetic tool_result
	// blocks for any orphaned tool_use, so the API never sees a tool_use without a
	// matching tool_result in the next message.
	const transformed = transformMessages(messages, model, normalizeToolCallId);

	for (let i = 0; i < transformed.length; i++) {
		const msg = transformed[i];

		if (msg.role === "user") {
			if (typeof msg.content === "string") {
				if (msg.content.trim().length > 0) {
					params.push({ role: "user", content: sanitizeSurrogates(msg.content) });
				}
			} else {
				const blocks = msg.content
					.map((item) => {
						if (item.type === "text") {
							return { type: "text" as const, text: sanitizeSurrogates(item.text) };
						}
						return {
							type: "image" as const,
							source: {
								type: "base64" as const,
								media_type: item.mimeType as "image/jpeg" | "image/png" | "image/gif" | "image/webp",
								data: item.data,
							},
						};
					})
					.filter((block) => {
						if (block.type === "text") {
							return block.text.trim().length > 0;
						}
						return model.input.includes("image");
					});

				if (blocks.length > 0) {
					params.push({ role: "user", content: blocks });
				}
			}
		} else if (msg.role === "assistant") {
			const blocks: NonNullable<MessageParam["content"]> extends string ? never : NonNullable<MessageParam["content"]> = [];
			for (const block of msg.content) {
				if (block.type === "text") {
					if (block.text.trim().length === 0) {
						continue;
					}
					blocks.push({ type: "text", text: sanitizeSurrogates(block.text) });
				} else if (block.type === "thinking") {
					if (block.thinking.trim().length === 0) {
						continue;
					}
					if (!block.thinkingSignature || block.thinkingSignature.trim().length === 0) {
						blocks.push({ type: "text", text: sanitizeSurrogates(block.thinking) });
					} else {
						blocks.push({ type: "thinking", thinking: sanitizeSurrogates(block.thinking), signature: block.thinkingSignature });
					}
				} else if (block.type === "toolCall") {
					blocks.push({
						type: "tool_use",
						id: block.id,
						name: block.name,
						input: block.arguments ?? {},
					});
				}
			}

			if (blocks.length > 0) {
				params.push({ role: "assistant", content: blocks });
			}
		} else if (msg.role === "toolResult") {
			const toolResults: Array<{
				type: "tool_result";
				tool_use_id: string;
				content: ReturnType<typeof convertContentBlocks>;
				is_error: boolean;
			}> = [];

			toolResults.push({
				type: "tool_result",
				tool_use_id: msg.toolCallId,
				content: convertContentBlocks(msg.content),
				is_error: msg.isError,
			});

			let j = i + 1;
			while (j < transformed.length && transformed[j].role === "toolResult") {
				const next = transformed[j] as ToolResultMessage;
				toolResults.push({
					type: "tool_result",
					tool_use_id: next.toolCallId,
					content: convertContentBlocks(next.content),
					is_error: next.isError,
				});
				j++;
			}
			i = j - 1;
			params.push({ role: "user", content: toolResults });
		}
	}

	if (cacheControl && params.length > 0) {
		const last = params[params.length - 1];
		if (last.role === "user") {
			if (Array.isArray(last.content)) {
				const block = last.content[last.content.length - 1];
				if (block && (block.type === "text" || block.type === "image" || block.type === "tool_result")) {
					(block as { cache_control?: { type: "ephemeral"; ttl?: "1h" } }).cache_control = cacheControl;
				}
			} else {
				last.content = [{ type: "text", text: last.content, cache_control: cacheControl }];
			}
		}
	}

	return params;
}

// Anthropic tool_use IDs must match ^[a-zA-Z0-9_-]+$ and be <= 64 chars.
// Used by transformMessages to rewrite IDs from foreign providers (e.g. OpenAI Responses).
function normalizeToolCallId(id: string): string {
	const sanitized = id.replace(/[^a-zA-Z0-9_-]/g, "_");
	return sanitized.length > 64 ? sanitized.slice(0, 64) : sanitized;
}

function convertTools(tools: Tool[] | undefined): AnthropicTool[] {
	if (!tools) {
		return [];
	}

	return tools.map((tool) => {
		const schema = tool.parameters as { properties?: Record<string, unknown>; required?: string[] };
		return {
			name: tool.name,
			description: tool.description,
			input_schema: {
				type: "object",
				properties: schema.properties ?? {},
				required: schema.required ?? [],
			},
		};
	});
}

function parseStreamingJson(partial: string): Record<string, unknown> {
	if (partial.trim().length === 0) {
		return {};
	}
	try {
		return JSON.parse(partial) as Record<string, unknown>;
	} catch {
		return {};
	}
}

let cachedGcloudProject: string | undefined;
let gcloudProjectResolved = false;

function readProjectFromGcloud(): string | undefined {
	if (gcloudProjectResolved) {
		return cachedGcloudProject;
	}
	gcloudProjectResolved = true;
	try {
		const value = execSync("gcloud config get-value project", {
			encoding: "utf-8",
			stdio: ["ignore", "pipe", "ignore"],
			timeout: 3000,
		}).trim();
		cachedGcloudProject = value.length > 0 ? value : undefined;
	} catch {
		cachedGcloudProject = undefined;
	}
	return cachedGcloudProject;
}

function resolveProject(options?: AnthropicVertexOptions): string | undefined {
	return (
		options?.project ??
		process.env.GOOGLE_CLOUD_PROJECT ??
		process.env.GCLOUD_PROJECT ??
		process.env.ANTHROPIC_VERTEX_PROJECT_ID ??
		readProjectFromGcloud()
	);
}

function resolveRegion(model: Model<Api>, options?: AnthropicVertexOptions): string {
        // Opus 4.7, 4.8 and 5 are only available in "global" right now
	if (["opus-4-7", "opus-4-8", "opus-5"].some((id) => model.id.includes(id))) {
		return "global";
	}
	return (
		options?.region ??
		process.env.GOOGLE_CLOUD_LOCATION ??
		process.env.CLOUD_ML_REGION ??
		DEFAULT_REGION
	);
}

// The Vertex SDK authenticates each request by POSTing to
// https://oauth2.googleapis.com/token via google-auth-library. That fetch runs
// *before* the Anthropic SDK's own retry loop, so a transient socket error
// (ECONNRESET / ETIMEDOUT / EAI_AGAIN) on the token endpoint fails the whole
// stream with a bare "request to .../token failed, reason:" and is never retried.
//
// We share one GoogleAuth across requests (so the access token is cached instead
// of refetched per stream) and wrap getRequestHeaders/getAccessToken with a small
// exponential backoff so a single network blip no longer kills the turn.
const TOKEN_RETRY_ATTEMPTS = 3;
const TOKEN_RETRY_BASE_MS = 250;

function isTransientAuthError(error: unknown): boolean {
	const message = error instanceof Error ? error.message : String(error);
	const code = (error as { code?: string } | undefined)?.code;
	const causeCode = (error as { cause?: { code?: string } } | undefined)?.cause?.code;
	return (
		/oauth2\.googleapis\.com\/token/i.test(message) ||
		/ECONNRESET|ETIMEDOUT|EAI_AGAIN|ENOTFOUND|ECONNREFUSED|socket hang up|network|fetch failed|timed? ?out/i.test(message) ||
		[code, causeCode].some(
			(c) => c === "ECONNRESET" || c === "ETIMEDOUT" || c === "EAI_AGAIN" || c === "ENOTFOUND" || c === "ECONNREFUSED",
		)
	);
}

async function withTokenRetry<T>(fn: () => Promise<T>): Promise<T> {
	let lastError: unknown;
	for (let attempt = 0; attempt < TOKEN_RETRY_ATTEMPTS; attempt++) {
		try {
			return await fn();
		} catch (error) {
			lastError = error;
			if (attempt === TOKEN_RETRY_ATTEMPTS - 1 || !isTransientAuthError(error)) {
				throw error;
			}
			const delay = TOKEN_RETRY_BASE_MS * 2 ** attempt + Math.floor(Math.random() * 100);
			await new Promise((resolve) => setTimeout(resolve, delay));
		}
	}
	throw lastError;
}

// Wraps the resolved AuthClient so its token fetches retry on transient errors.
// We delegate via Proxy so every other method/property stays identical to the
// underlying client (the Vertex SDK only calls getRequestHeaders, but this keeps
// behavior intact if that changes).
function wrapAuthClientWithRetry(client: AuthClient): AuthClient {
	return new Proxy(client, {
		get(target, prop, receiver) {
			if (prop === "getRequestHeaders") {
				return (...args: unknown[]) =>
					withTokenRetry(() => (target.getRequestHeaders as (...a: unknown[]) => Promise<unknown>)(...args));
			}
			if (prop === "getAccessToken") {
				return (...args: unknown[]) =>
					withTokenRetry(() => (target.getAccessToken as (...a: unknown[]) => Promise<unknown>)(...args));
			}
			const value = Reflect.get(target, prop, receiver);
			return typeof value === "function" ? value.bind(target) : value;
		},
	});
}

// One GoogleAuth instance shared across all clients. getClient() caches the
// resolved AuthClient, which in turn caches the access token until it expires.
let sharedGoogleAuth: GoogleAuth | undefined;

function getSharedGoogleAuth(): GoogleAuth {
	if (!sharedGoogleAuth) {
		const base = new GoogleAuth({ scopes: "https://www.googleapis.com/auth/cloud-platform" });
		// Wrap getClient so the AuthClient it hands to the Vertex SDK has retrying
		// token fetches. getClient itself can also touch the network (ADC
		// discovery), so retry it too.
		const originalGetClient = base.getClient.bind(base);
		base.getClient = (() =>
			withTokenRetry(originalGetClient).then((client) =>
				wrapAuthClientWithRetry(client as AuthClient),
			)) as typeof base.getClient;
		sharedGoogleAuth = base;
	}
	return sharedGoogleAuth;
}

function createClient(model: Model<Api>, options?: AnthropicVertexOptions): AnthropicVertex {
	const betaFeatures = ["fine-grained-tool-streaming-2025-05-14"];
	if (options?.interleavedThinking ?? true) {
		betaFeatures.push("interleaved-thinking-2025-05-14");
	}

	const project = resolveProject(options);
	if (!project) {
		throw new Error(
			"Anthropic Vertex requires a project ID. Set ANTHROPIC_VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT.",
		);
	}
	return new AnthropicVertex({
		projectId: project,
		region: resolveRegion(model, options),
		googleAuth: getSharedGoogleAuth(),
		defaultHeaders: mergeHeaders(
			{
				accept: "application/json",
				"anthropic-beta": betaFeatures.join(","),
			},
			model.headers,
			options?.headers,
		),
	});
}

function buildParams(
	model: Model<Api>,
	context: Context,
	options?: AnthropicVertexOptions,
): MessageCreateParamsStreaming {
	const { cacheControl } = getCacheControl(model.baseUrl, options?.cacheRetention);
	const params: MessageCreateParamsStreaming = {
		model: model.id,
		messages: convertMessages(context.messages, model, cacheControl),
		max_tokens: options?.maxTokens ?? ((model.maxTokens / 3) | 0),
		stream: true,
	};

	if (context.systemPrompt) {
		params.system = [
			{
				type: "text",
				text: sanitizeSurrogates(context.systemPrompt),
				...(cacheControl ? { cache_control: cacheControl } : {}),
			},
		];
	}

	if (options?.temperature !== undefined) {
		params.temperature = options.temperature;
	}

	if (context.tools) {
		params.tools = convertTools(context.tools);
	}

	if (options?.thinkingEnabled && model.reasoning) {
		if (supportsAdaptiveThinking(model.id)) {
			params.thinking = { type: "adaptive" };
			if (options.effort) {
				params.output_config = { effort: options.effort };
			}
		} else {
			params.thinking = {
				type: "enabled",
				budget_tokens: options.thinkingBudgetTokens ?? 1024,
			};
		}
	}

	if (options?.metadata && typeof options.metadata.user_id === "string") {
		params.metadata = { user_id: options.metadata.user_id };
	}

	if (options?.toolChoice) {
		params.tool_choice = typeof options.toolChoice === "string" ? { type: options.toolChoice } : options.toolChoice;
	}

	return params;
}

const streamAnthropicVertexStandalone: StreamFunction<Api, AnthropicVertexOptions> = (
	model,
	context,
	options,
) => {
	const stream = createAssistantMessageEventStream();

	(async () => {
		const output: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now(),
		};

		try {
			const client = createClient(model, options);
			const params = buildParams(model, context, options);
			options?.onPayload?.(params);
			const events = client.messages.stream({ ...params, stream: true }, { signal: options?.signal });

			stream.push({ type: "start", partial: output });
			const blocks = output.content as AnthropicStreamingBlock[];

			for await (const event of events as AsyncIterable<RawMessageStreamEvent>) {
				if (event.type === "message_start") {
					output.usage.input = event.message.usage.input_tokens || 0;
					output.usage.output = event.message.usage.output_tokens || 0;
					output.usage.cacheRead = event.message.usage.cache_read_input_tokens || 0;
					output.usage.cacheWrite = event.message.usage.cache_creation_input_tokens || 0;
					output.usage.totalTokens =
						output.usage.input + output.usage.output + output.usage.cacheRead + output.usage.cacheWrite;
					calculateCost(model, output.usage);
				} else if (event.type === "content_block_start") {
					if (event.content_block.type === "text") {
						blocks.push({ type: "text", text: "", index: event.index });
						stream.push({ type: "text_start", contentIndex: blocks.length - 1, partial: output });
					} else if (event.content_block.type === "thinking") {
						blocks.push({ type: "thinking", thinking: "", thinkingSignature: "", index: event.index });
						stream.push({ type: "thinking_start", contentIndex: blocks.length - 1, partial: output });
					} else if (event.content_block.type === "tool_use") {
						blocks.push({
							type: "toolCall",
							id: event.content_block.id,
							name: event.content_block.name,
							arguments: (event.content_block.input as Record<string, unknown>) ?? {},
							partialJson: "",
							index: event.index,
						});
						stream.push({ type: "toolcall_start", contentIndex: blocks.length - 1, partial: output });
					}
				} else if (event.type === "content_block_delta") {
					const index = blocks.findIndex((b) => b.index === event.index);
					const block = blocks[index];
					if (!block) {
						continue;
					}

					if (event.delta.type === "text_delta" && block.type === "text") {
						block.text += event.delta.text;
						stream.push({ type: "text_delta", contentIndex: index, delta: event.delta.text, partial: output });
					} else if (event.delta.type === "thinking_delta" && block.type === "thinking") {
						block.thinking += event.delta.thinking;
						stream.push({ type: "thinking_delta", contentIndex: index, delta: event.delta.thinking, partial: output });
					} else if (event.delta.type === "input_json_delta" && block.type === "toolCall") {
						block.partialJson += event.delta.partial_json;
						const parsed = parseStreamingJson(block.partialJson);
						if (Object.keys(parsed).length > 0) {
							block.arguments = parsed;
						}
						stream.push({
							type: "toolcall_delta",
							contentIndex: index,
							delta: event.delta.partial_json,
							partial: output,
						});
					} else if (event.delta.type === "signature_delta" && block.type === "thinking") {
						block.thinkingSignature = (block.thinkingSignature ?? "") + event.delta.signature;
					}
				} else if (event.type === "content_block_stop") {
					const index = blocks.findIndex((b) => b.index === event.index);
					const block = blocks[index];
					if (!block) {
						continue;
					}
					delete (block as { index?: number }).index;
					if (block.type === "text") {
						stream.push({ type: "text_end", contentIndex: index, content: block.text, partial: output });
					} else if (block.type === "thinking") {
						stream.push({ type: "thinking_end", contentIndex: index, content: block.thinking, partial: output });
					} else if (block.type === "toolCall") {
						const parsed = parseStreamingJson(block.partialJson);
						if (Object.keys(parsed).length > 0) {
							block.arguments = parsed;
						}
						delete (block as { partialJson?: string }).partialJson;
						stream.push({ type: "toolcall_end", contentIndex: index, toolCall: block, partial: output });
					}
				} else if (event.type === "message_delta") {
					if (event.delta.stop_reason) {
						output.stopReason = mapStopReason(event.delta.stop_reason);
					}
					if (event.usage.input_tokens != null) {
						output.usage.input = event.usage.input_tokens;
					}
					if (event.usage.output_tokens != null) {
						output.usage.output = event.usage.output_tokens;
					}
					if (event.usage.cache_read_input_tokens != null) {
						output.usage.cacheRead = event.usage.cache_read_input_tokens;
					}
					if (event.usage.cache_creation_input_tokens != null) {
						output.usage.cacheWrite = event.usage.cache_creation_input_tokens;
					}
					output.usage.totalTokens =
						output.usage.input + output.usage.output + output.usage.cacheRead + output.usage.cacheWrite;
					calculateCost(model, output.usage);
				}
			}

			if (options?.signal?.aborted) {
				throw new Error("Request was aborted");
			}

			if (output.stopReason === "aborted" || output.stopReason === "error") {
				throw new Error("An unknown error occurred");
			}

			stream.push({ type: "done", reason: output.stopReason, message: output });
			stream.end();
		} catch (error) {
			for (const block of output.content) {
				delete (block as { index?: number }).index;
				delete (block as { partialJson?: string }).partialJson;
			}
			output.stopReason = options?.signal?.aborted ? "aborted" : "error";
			output.errorMessage = error instanceof Error ? error.message : JSON.stringify(error);
			stream.push({ type: "error", reason: output.stopReason, error: output });
			stream.end();
		}
	})();

	return stream;
};

function clampReasoning(level: ThinkingLevel | undefined): Exclude<ThinkingLevel, "xhigh"> | undefined {
	if (!level) {
		return undefined;
	}
	return level === "xhigh" ? "high" : level;
}

function adjustMaxTokensForThinking(
	baseMaxTokens: number,
	modelMaxTokens: number,
	reasoningLevel: ThinkingLevel,
	customBudgets?: ThinkingBudgets,
): { maxTokens: number; thinkingBudget: number } {
	const defaultBudgets: ThinkingBudgets = {
		minimal: 1024,
		low: 2048,
		medium: 8192,
		high: 16384,
	};
	const budgets = { ...defaultBudgets, ...customBudgets };

	const minOutputTokens = 1024;
	const level = clampReasoning(reasoningLevel)!;
	let thinkingBudget = budgets[level] ?? defaultBudgets.high!;
	const maxTokens = Math.min(baseMaxTokens + thinkingBudget, modelMaxTokens);

	if (maxTokens <= thinkingBudget) {
		thinkingBudget = Math.max(0, maxTokens - minOutputTokens);
	}

	return { maxTokens, thinkingBudget };
}

const streamSimpleAnthropicVertexStandalone: StreamFunction<Api, SimpleStreamOptions> = (
	model,
	context,
	options,
) => {
	const base: AnthropicVertexOptions = {
		temperature: options?.temperature,
		maxTokens: options?.maxTokens ?? Math.min(model.maxTokens, 32000),
		signal: options?.signal,
		apiKey: options?.apiKey,
		cacheRetention: options?.cacheRetention,
		sessionId: options?.sessionId,
		headers: options?.headers,
		onPayload: options?.onPayload,
		maxRetryDelayMs: options?.maxRetryDelayMs,
		metadata: options?.metadata,
	};

	if (!options?.reasoning) {
		return streamAnthropicVertexStandalone(model, context, {
			...base,
			thinkingEnabled: false,
		});
	}

	if (supportsAdaptiveThinking(model.id)) {
		const effort = mapThinkingLevelToEffort(options.reasoning);
		return streamAnthropicVertexStandalone(model, context, {
			...base,
			thinkingEnabled: true,
			effort,
		});
	}

	const adjusted = adjustMaxTokensForThinking(
		base.maxTokens ?? 0,
		model.maxTokens,
		options.reasoning,
		options.thinkingBudgets,
	);

	return streamAnthropicVertexStandalone(model, context, {
		...base,
		maxTokens: adjusted.maxTokens,
		thinkingEnabled: true,
		thinkingBudgetTokens: adjusted.thinkingBudget,
	});
};

export default function registerAnthropicVertex(pi: ExtensionAPI): void {
	// Resolve the project ID once at registration time. Previously this used a
	// `!sh -lc '...'` apiKey, which pi re-executes via execSync on every prompt
	// submission (resolveConfigValueUncached bypasses the cache), causing a
	// ~1-2s pause on Enter due to spawning sh + gcloud each time.
	//
	// The Vertex SDK authenticates via google-auth-library / ADC, not this
	// apiKey value. pi only needs *some* truthy string here for hasConfiguredAuth
	// bookkeeping, so we hand it the already-resolved project ID as a literal.
	const resolvedProject = resolveProject();
	pi.registerProvider("anthropic-vertex", {
		baseUrl: BASE_URL,
		api: "anthropic-vertex",
		...(resolvedProject ? { apiKey: resolvedProject } : {}),
		models: MODELS,
		streamSimple: streamSimpleAnthropicVertexStandalone,
	});
}
