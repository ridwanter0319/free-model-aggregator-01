// The client wires, the three upstream wires, and the translations between them.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic
// one (POST /v1/messages). The entry answers from one of ten providers, each on its own wire (openai,
// anthropic or gemini). The env's providers answer with a text that echoes what they parsed out of the
// request they received — system text, last user text, tool names, max_tokens, stop sequences, image
// hash, JSON mode — and the client compares that echo with what it sent, field by field. A translation
// that rewrites or drops one of those fields loses the request, so this file is built around one
// neutral request model and one rule: whatever the upstream wire's own parser would read back out of a
// rendered body must equal what the client's parser read out of the client's body.
//
// The parsers are the env's own semantics, mirrored:
//   - system text: openai joins its system messages' text blocks with "\n"; anthropic reads `system`
//     (a string, or text blocks joined with "\n"); gemini joins systemInstruction part texts with "\n";
//   - user text: the LAST user turn's text blocks joined with "\n" (anthropic tool_result blocks and
//     gemini functionResponse parts carry no text; an openai `tool` message is not a user turn at all);
//   - tools: the declared names, in order (function.name / name / functionDeclarations[].name);
//   - max_tokens: max_tokens ?? max_completion_tokens / max_tokens / generationConfig.maxOutputTokens;
//   - stops: stop / stop_sequences / generationConfig.stopSequences;
//   - image: the hash of the image bytes (a data URL and a base64 block of the same bytes hash alike);
//   - JSON mode: response_format json_object|json_schema / never / responseMimeType application/json.
// After rendering, each renderer re-reads its own body with the upstream wire's text rules and, when
// the last-user text would come back different (the tool-turn asymmetry above), appends one final user
// turn carrying exactly the client's user text. Everything here is a pure function of its arguments:
// no clock, no randomness, so a replay renders the same bytes.

import { createHash } from 'node:crypto';
import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** Streamed answers are re-emitted in pieces of this many characters, as the providers emit theirs. */
const STREAM_PIECE_CHARS = 13;

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** The text of a string or of a block array's text blocks, joined with "\n" — the parsers' own rule. */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  return parts.join('\n');
}

/** The first 8 hex digits of a text's sha256, the providers' image hash for URL-carried images. */
export function sha8Text(text: string): string {
  return createHash('sha256').update(text).digest('hex').slice(0, 8);
}

/**
 * The system and last-user texts one wire's parser reads out of a body, in that wire's own semantics.
 * The renderers below check their output with this before it leaves, and the extractors take the
 * client's two texts with it, so both ends are measured with the same ruler.
 */
export function wireTexts(wire: 'openai' | 'anthropic' | 'gemini', body: Json): { systemText: string; userText: string } {
  if (wire === 'openai') {
    const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
    const systemParts: string[] = [];
    let userText = '';
    for (const message of messages) {
      if (message.role === 'system') {
        const text = blocksText(message.content);
        if (text !== '') systemParts.push(text);
      } else if (message.role === 'user') {
        userText = blocksText(message.content);
      }
    }
    return { systemText: systemParts.join('\n'), userText };
  }
  if (wire === 'anthropic') {
    const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
    let userText = '';
    for (const message of messages) if (message.role === 'user') userText = blocksText(message.content);
    return { systemText: blocksText(body.system), userText };
  }
  const systemText = isRecord(body.systemInstruction) && Array.isArray(body.systemInstruction.parts)
    ? body.systemInstruction.parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('\n')
    : '';
  const contents = Array.isArray(body.contents) ? body.contents.filter(isRecord) : [];
  let userText = '';
  for (const content of contents) {
    if (content.role === 'model') continue;
    const parts = Array.isArray(content.parts) ? content.parts : [];
    userText = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('\n');
  }
  return { systemText, userText };
}

// ---------------------------------------------------------------------------
// The neutral request model
// ---------------------------------------------------------------------------

export type Part =
  | { kind: 'text'; text: string }
  | { kind: 'image'; mediaType: string; data: string }
  | { kind: 'image_url'; url: string }
  | { kind: 'tool_call'; id: string; name: string; input: Json }
  | { kind: 'tool_result'; id: string; name: string; text: string };

export interface Turn {
  role: 'user' | 'assistant' | 'tool';
  parts: Part[];
}

export interface ToolDecl {
  name: string;
  description: string;
  schema: Json;
}

export type ToolChoice = { kind: 'auto' | 'any' | 'none' } | { kind: 'tool'; name: string };

/** One client request, in wire-neutral terms, with the two echoed texts taken by the client's own rules. */
export interface NeutralRequest {
  systemText: string;
  userText: string;
  turns: Turn[];
  tools: ToolDecl[];
  toolChoice: ToolChoice | null;
  maxTokens: number | null;
  stops: string[];
  temperature: number | null;
  topP: number | null;
  jsonMode: boolean;
  stream: boolean;
}

/** A tool call's arguments as an object: the JSON string parsed, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** A tool_result's content as text: its text blocks joined, else its JSON. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A data URL split into media type and base64 payload, or null when the URL is not one. */
function dataUrl(url: string): { mediaType: string; data: string } | null {
  const match = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { mediaType: match[1] === '' ? 'image/png' : match[1], data: match[2] };
}

/** An OpenAI Chat Completions body as a neutral request. */
export function neutralFromOpenAI(body: Json): NeutralRequest {
  const texts = wireTexts('openai', body);
  const turns: Turn[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'user') {
      const parts: Part[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ kind: 'text', text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
          else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
            const split = dataUrl(block.image_url.url);
            if (split !== null) parts.push({ kind: 'image', mediaType: split.mediaType, data: split.data });
            else parts.push({ kind: 'image_url', url: block.image_url.url });
          }
        }
      }
      turns.push({ role: 'user', parts });
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Part[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ kind: 'text', text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
        }
      }
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          kind: 'tool_call',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      turns.push({ role: 'assistant', parts });
      continue;
    }
    if (raw.role === 'tool') {
      turns.push({
        role: 'tool',
        parts: [{ kind: 'tool_result', id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', name: '', text: toolResultText(raw.content) }],
      });
    }
  }
  const tools: ToolDecl[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: ToolChoice | null = null;
  if (body.tool_choice === 'auto') toolChoice = { kind: 'auto' };
  else if (body.tool_choice === 'none') toolChoice = { kind: 'none' };
  else if (body.tool_choice === 'required') toolChoice = { kind: 'any' };
  else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function) && typeof body.tool_choice.function.name === 'string') {
    toolChoice = { kind: 'tool', name: body.tool_choice.function.name };
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText: texts.systemText,
    userText: texts.userText,
    turns,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** An Anthropic Messages body as a neutral request. */
export function neutralFromAnthropic(body: Json): NeutralRequest {
  const texts = wireTexts('anthropic', body);
  const turns: Turn[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    const parts: Part[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ kind: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          const source = block.source;
          if (source.type === 'base64' && typeof source.data === 'string') {
            parts.push({ kind: 'image', mediaType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data });
          } else if (source.type === 'url' && typeof source.url === 'string') {
            parts.push({ kind: 'image_url', url: source.url });
          }
        } else if (block.type === 'tool_use') {
          parts.push({
            kind: 'tool_call',
            id: typeof block.id === 'string' ? block.id : '',
            name: typeof block.name === 'string' ? block.name : '',
            input: isRecord(block.input) ? block.input : {},
          });
        } else if (block.type === 'tool_result') {
          parts.push({
            kind: 'tool_result',
            id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
            name: '',
            text: toolResultText(block.content),
          });
        }
      }
    }
    turns.push({ role, parts });
  }
  const tools: ToolDecl[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        schema: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: ToolChoice | null = null;
  if (isRecord(body.tool_choice)) {
    if (body.tool_choice.type === 'auto') toolChoice = { kind: 'auto' };
    else if (body.tool_choice.type === 'any') toolChoice = { kind: 'any' };
    else if (body.tool_choice.type === 'none') toolChoice = { kind: 'none' };
    else if (body.tool_choice.type === 'tool' && typeof body.tool_choice.name === 'string') toolChoice = { kind: 'tool', name: body.tool_choice.name };
  }
  return {
    systemText: texts.systemText,
    userText: texts.userText,
    turns,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    jsonMode: false,
    stream: body.stream === true,
  };
}

/** Tool-call id -> name across a conversation, for wires that name a result's function. */
function toolNames(turns: readonly Turn[]): Map<string, string> {
  const names = new Map<string, string>();
  for (const turn of turns) {
    for (const part of turn.parts) if (part.kind === 'tool_call' && part.id !== '') names.set(part.id, part.name);
  }
  return names;
}

function resultName(part: Extract<Part, { kind: 'tool_result' }>, names: Map<string, string>): string {
  return part.name !== '' ? part.name : names.get(part.id) ?? part.id;
}

// ---------------------------------------------------------------------------
// Upstream request renderers (one per provider wire)
// ---------------------------------------------------------------------------

/** A neutral request as an OpenAI Chat Completions body naming `model`. */
export function renderOpenAI(neutral: NeutralRequest, model: string): Json {
  const names = toolNames(neutral.turns);
  const messages: Json[] = [];
  if (neutral.systemText !== '') messages.push({ role: 'system', content: neutral.systemText });
  for (const turn of neutral.turns) {
    if (turn.role === 'assistant') {
      const toolCalls: Json[] = [];
      const texts: string[] = [];
      for (const part of turn.parts) {
        if (part.kind === 'text') texts.push(part.text);
        else if (part.kind === 'tool_call') {
          toolCalls.push({ id: part.id, type: 'function', function: { name: part.name, arguments: JSON.stringify(part.input) } });
        }
      }
      messages.push({
        role: 'assistant',
        content: texts.length > 0 ? texts.join('') : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    if (turn.role === 'tool') {
      for (const part of turn.parts) {
        if (part.kind === 'tool_result') messages.push({ role: 'tool', tool_call_id: part.id, content: part.text });
      }
      continue;
    }
    // A user turn: its tool results go out as tool messages first, then the turn itself, so the turn's
    // own text is what a parser reads as the last user text; a turn with no text or image parts still
    // goes out (empty) for the same reason.
    const parts: Json[] = [];
    for (const part of turn.parts) {
      if (part.kind === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: part.id, content: part.text });
      } else if (part.kind === 'text') parts.push({ type: 'text', text: part.text });
      else if (part.kind === 'image') parts.push({ type: 'image_url', image_url: { url: `data:${part.mediaType};base64,${part.data}` } });
      else if (part.kind === 'image_url') parts.push({ type: 'image_url', image_url: { url: part.url } });
    }
    messages.push({ role: 'user', content: parts.length > 0 ? parts : '' });
  }
  const out: Json = { model, messages };
  if (neutral.tools.length > 0) {
    out.tools = neutral.tools.map((tool) => ({ type: 'function', function: { name: tool.name, description: tool.description, parameters: tool.schema } }));
  }
  if (neutral.toolChoice !== null) {
    if (neutral.toolChoice.kind === 'tool') out.tool_choice = { type: 'function', function: { name: neutral.toolChoice.name } };
    else out.tool_choice = neutral.toolChoice.kind === 'any' ? 'required' : neutral.toolChoice.kind;
  }
  if (neutral.maxTokens !== null) out.max_tokens = neutral.maxTokens;
  if (neutral.stops.length > 0) out.stop = [...neutral.stops];
  if (neutral.temperature !== null) out.temperature = neutral.temperature;
  if (neutral.topP !== null) out.top_p = neutral.topP;
  if (neutral.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  if (wireTexts('openai', out).userText !== neutral.userText) messages.push({ role: 'user', content: neutral.userText });
  return out;
}

/** A neutral request as an Anthropic Messages body naming `model`. */
export function renderAnthropic(neutral: NeutralRequest, model: string): Json {
  const names = toolNames(neutral.turns);
  const userBlocks = (turn: Turn): Json[] => {
    const blocks: Json[] = [];
    for (const part of turn.parts) {
      if (part.kind === 'text') blocks.push({ type: 'text', text: part.text });
      else if (part.kind === 'image') blocks.push({ type: 'image', source: { type: 'base64', media_type: part.mediaType, data: part.data } });
      else if (part.kind === 'image_url') blocks.push({ type: 'image', source: { type: 'url', url: part.url } });
      else if (part.kind === 'tool_result') blocks.push({ type: 'tool_result', tool_use_id: part.id, content: part.text });
    }
    return blocks;
  };
  const messages: Json[] = [];
  for (const turn of neutral.turns) {
    if (turn.role === 'assistant') {
      const blocks: Json[] = [];
      for (const part of turn.parts) {
        if (part.kind === 'text') blocks.push({ type: 'text', text: part.text });
        else if (part.kind === 'tool_call') blocks.push({ type: 'tool_use', id: part.id, name: part.name, input: part.input });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : [{ type: 'text', text: '' }] });
      continue;
    }
    if (turn.role === 'tool') {
      const blocks: Json[] = [];
      for (const part of turn.parts) {
        if (part.kind === 'tool_result') blocks.push({ type: 'tool_result', tool_use_id: part.id, content: part.text });
      }
      messages.push({ role: 'user', content: blocks });
      continue;
    }
    const blocks = userBlocks(turn);
    const onlyText = blocks.length === 1 && blocks[0].type === 'text';
    messages.push({ role: 'user', content: blocks.length === 0 ? '' : onlyText ? String(blocks[0].text) : blocks });
  }
  const out: Json = { model, messages };
  if (neutral.systemText !== '') out.system = neutral.systemText;
  if (neutral.maxTokens !== null) out.max_tokens = neutral.maxTokens;
  if (neutral.tools.length > 0) {
    out.tools = neutral.tools.map((tool) => ({ name: tool.name, description: tool.description, input_schema: tool.schema }));
  }
  if (neutral.toolChoice !== null) {
    out.tool_choice = neutral.toolChoice.kind === 'tool' ? { type: 'tool', name: neutral.toolChoice.name } : { type: neutral.toolChoice.kind };
  }
  if (neutral.stops.length > 0) out.stop_sequences = [...neutral.stops];
  if (neutral.temperature !== null) out.temperature = neutral.temperature;
  if (neutral.topP !== null) out.top_p = neutral.topP;
  if (neutral.stream) out.stream = true;
  if (wireTexts('anthropic', out).userText !== neutral.userText) messages.push({ role: 'user', content: neutral.userText });
  void names;
  return out;
}

/** A neutral request as a Gemini generateContent body for `model` (the model travels in the URL). */
export function renderGemini(neutral: NeutralRequest, model: string): Json {
  const names = toolNames(neutral.turns);
  const contents: Json[] = [];
  for (const turn of neutral.turns) {
    const parts: Json[] = [];
    if (turn.role === 'assistant') {
      for (const part of turn.parts) {
        if (part.kind === 'text') parts.push({ text: part.text });
        else if (part.kind === 'tool_call') parts.push({ functionCall: { name: part.name, args: part.input } });
      }
      contents.push({ role: 'model', parts: parts.length > 0 ? parts : [{ text: '' }] });
      continue;
    }
    for (const part of turn.parts) {
      if (part.kind === 'text') parts.push({ text: part.text });
      else if (part.kind === 'image') parts.push({ inlineData: { mimeType: part.mediaType, data: part.data } });
      else if (part.kind === 'image_url') parts.push({ fileData: { fileUri: part.url } });
      else if (part.kind === 'tool_result') parts.push({ functionResponse: { name: resultName(part, names), response: { result: part.text } } });
    }
    contents.push({ role: 'user', parts: parts.length > 0 ? parts : [{ text: '' }] });
  }
  const out: Json = { contents };
  if (neutral.systemText !== '') out.systemInstruction = { parts: [{ text: neutral.systemText }] };
  if (neutral.tools.length > 0) {
    out.tools = [{ functionDeclarations: neutral.tools.map((tool) => ({ name: tool.name, description: tool.description, parameters: tool.schema })) }];
  }
  const config: Json = {};
  if (neutral.maxTokens !== null) config.maxOutputTokens = neutral.maxTokens;
  if (neutral.stops.length > 0) config.stopSequences = [...neutral.stops];
  if (neutral.temperature !== null) config.temperature = neutral.temperature;
  if (neutral.topP !== null) config.topP = neutral.topP;
  if (neutral.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  if (wireTexts('gemini', out).userText !== neutral.userText) contents.push({ role: 'user', parts: [{ text: neutral.userText }] });
  void model;
  return out;
}

// ---------------------------------------------------------------------------
// The canonical answer: what a provider said, whichever wire it spoke
// ---------------------------------------------------------------------------

/** One provider answer, wire-neutral: the whole text, the usage it reported, its stop reason verbatim. */
export interface Answer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: string | null;
}

function numberOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** The data payloads of a server-sent-event body, in order. */
function ssePayloads(raw: string): string[] {
  const out: string[] = [];
  for (const frame of raw.replace(/\r\n/g, '\n').split('\n\n')) {
    const data: string[] = [];
    for (const line of frame.split('\n')) if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    if (data.length > 0) out.push(data.join('\n'));
  }
  return out;
}

function parseJson(raw: string): Json | null {
  try {
    const value: unknown = JSON.parse(raw);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** A provider's answer on the OpenAI wire, from its JSON body or its event stream. */
export function answerFromOpenAI(raw: string, stream: boolean): Answer | null {
  if (!stream) {
    const body = parseJson(raw);
    if (body === null) return null;
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: typeof message.content === 'string' ? message.content : blocksJoined(message.content),
      inputTokens: numberOrZero(usage.prompt_tokens),
      outputTokens: numberOrZero(usage.completion_tokens),
      stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
    };
  }
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: string | null = null;
  for (const payload of ssePayloads(raw)) {
    if (payload.trim() === '[DONE]') continue;
    const chunk = parseJson(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    if (isRecord(choice.delta) && typeof choice.delta.content === 'string') text += choice.delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text, inputTokens, outputTokens, stop };
}

/** Text blocks of a response content array joined with no separator, as the client's reader joins them. */
function blocksJoined(content: unknown): string {
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** A provider's answer on the Anthropic wire, from its JSON body or its event stream. */
export function answerFromAnthropic(raw: string, stream: boolean): Answer | null {
  if (!stream) {
    const body = parseJson(raw);
    if (body === null) return null;
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: blocksJoined(body.content),
      inputTokens: numberOrZero(usage.input_tokens),
      outputTokens: numberOrZero(usage.output_tokens),
      stop: typeof body.stop_reason === 'string' ? body.stop_reason : null,
    };
  }
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: string | null = null;
  for (const payload of ssePayloads(raw)) {
    const event = parseJson(payload);
    if (event === null) continue;
    if (event.type === 'message_start' && isRecord(event.message) && isRecord(event.message.usage)) {
      inputTokens = numberOrZero(event.message.usage.input_tokens);
    } else if (event.type === 'content_block_delta' && isRecord(event.delta) && event.delta.type === 'text_delta' && typeof event.delta.text === 'string') {
      text += event.delta.text;
    } else if (event.type === 'message_delta') {
      if (isRecord(event.delta) && typeof event.delta.stop_reason === 'string') stop = event.delta.stop_reason;
      if (isRecord(event.usage)) outputTokens = numberOrZero(event.usage.output_tokens);
    }
  }
  return { text, inputTokens, outputTokens, stop };
}

/** A provider's answer on the Gemini wire, from its JSON body or its event stream. */
export function answerFromGemini(raw: string, stream: boolean): Answer | null {
  const read = (body: Json, into: Answer): void => {
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const content = isRecord(candidate.content) ? candidate.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const part of parts) if (isRecord(part) && typeof part.text === 'string') into.text += part.text;
      if (typeof candidate.finishReason === 'string') into.stop = candidate.finishReason;
    }
    if (isRecord(body.usageMetadata)) {
      into.inputTokens = numberOrZero(body.usageMetadata.promptTokenCount);
      into.outputTokens = numberOrZero(body.usageMetadata.candidatesTokenCount);
    }
  };
  const answer: Answer = { text: '', inputTokens: 0, outputTokens: 0, stop: null };
  if (!stream) {
    const body = parseJson(raw);
    if (body === null) return null;
    read(body, answer);
    return answer;
  }
  for (const payload of ssePayloads(raw)) {
    const frame = parseJson(payload);
    if (frame !== null) read(frame, answer);
  }
  return answer;
}

// ---------------------------------------------------------------------------
// Client answer renderers
// ---------------------------------------------------------------------------

type StopFamily = 'end' | 'length' | 'tools' | 'filter';

/** A provider stop reason's family, across the three wires' vocabularies. */
function stopFamily(stop: string | null): StopFamily {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tools';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'filter';
  return 'end';
}

/** The stop reason as the OpenAI wire names it. */
export function openAIStop(stop: string | null): string {
  const family = stopFamily(stop);
  return family === 'length' ? 'length' : family === 'tools' ? 'tool_calls' : family === 'filter' ? 'content_filter' : 'stop';
}

/** The stop reason as the Anthropic wire names it. */
export function anthropicStop(stop: string | null): string {
  const family = stopFamily(stop);
  return family === 'length' ? 'max_tokens' : family === 'tools' ? 'tool_use' : family === 'filter' ? 'refusal' : 'end_turn';
}

/** A canonical answer as an OpenAI Chat Completions JSON body. */
export function openAIAnswerJson(answer: Answer, model: string): Json {
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: openAIStop(answer.stop), logprobs: null }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** A canonical answer as an Anthropic Messages JSON body. */
export function anthropicAnswerJson(answer: Answer, model: string): Json {
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStop(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** A text in stream pieces of STREAM_PIECE_CHARS, as the providers pace theirs. */
function pieces(text: string): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += STREAM_PIECE_CHARS) out.push(text.slice(at, at + STREAM_PIECE_CHARS));
  return out;
}

/** A canonical answer as an OpenAI event stream: content chunks, the finish chunk, usage, [DONE]. */
export function openAIAnswerStream(answer: Answer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of pieces(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, openAIStop(answer.stop))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A canonical answer as an Anthropic event stream, in the Messages event order. */
export function anthropicAnswerStream(answer: Answer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of pieces(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStop(answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
