// The translation seams, tested at the level the trial scores them: for every client body, the
// upstream body a renderer produces must read back — under the upstream wire's own text rules — with
// the same system and last-user texts the client's wire read out of the client's body, and a
// provider's answer must survive extraction and re-rendering on the other wire with its text, usage
// and stop family intact. Run with: node --test test/wire.test.ts

import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { Json } from '../lib/providers.ts';
import {
  answerFromAnthropic,
  answerFromGemini,
  answerFromOpenAI,
  anthropicAnswerStream,
  anthropicStop,
  neutralFromAnthropic,
  neutralFromOpenAI,
  openAIAnswerJson,
  openAIAnswerStream,
  openAIStop,
  renderAnthropic,
  renderGemini,
  renderOpenAI,
  wireTexts,
} from '../lib/wire.ts';

const IMAGE_BYTES = Buffer.from('fake-png-bytes').toString('base64');

/** An OpenAI client body with a tool history: the last user text precedes a tool turn. */
const openAIToolHistory: Json = {
  model: 'auto',
  messages: [
    { role: 'system', content: 'You are careful.' },
    { role: 'user', content: 'What is the weather in Paris?' },
    { role: 'assistant', content: null, tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Paris"}' } }] },
    { role: 'tool', tool_call_id: 'call_1', content: 'sunny, 24C' },
  ],
  tools: [{ type: 'function', function: { name: 'get_weather', description: 'Weather', parameters: { type: 'object', properties: { city: { type: 'string' } } } } }],
  max_tokens: 300,
  stop: ['END', 'STOP'],
};

test('openai tool history: the last-user text survives rendering to anthropic and gemini', () => {
  const neutral = neutralFromOpenAI(openAIToolHistory);
  assert.equal(neutral.userText, 'What is the weather in Paris?');
  assert.equal(neutral.systemText, 'You are careful.');
  assert.deepEqual(neutral.tools.map((t) => t.name), ['get_weather']);
  assert.equal(neutral.maxTokens, 300);
  assert.deepEqual(neutral.stops, ['END', 'STOP']);
  // On the anthropic and gemini wires the tool turn becomes a user turn with no text, which would
  // read back as the last user text "" — the renderers pin the client's text back on.
  assert.equal(wireTexts('anthropic', renderAnthropic(neutral, 'm')).userText, neutral.userText);
  assert.equal(wireTexts('gemini', renderGemini(neutral, 'm')).userText, neutral.userText);
  assert.equal(wireTexts('openai', renderOpenAI(neutral, 'm')).userText, neutral.userText);
  assert.equal(wireTexts('gemini', renderGemini(neutral, 'm')).systemText, neutral.systemText);
});

/** An Anthropic client body whose last user turn is a bare tool_result. */
const anthropicToolResult: Json = {
  model: 'auto',
  system: [{ type: 'text', text: 'Line one.' }, { type: 'text', text: 'Line two.' }],
  messages: [
    { role: 'user', content: 'Read the file.' },
    { role: 'assistant', content: [{ type: 'tool_use', id: 'tu_1', name: 'read_file', input: { path: '/x' } }] },
    { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'tu_1', content: 'file body' }] },
  ],
  tools: [{ name: 'read_file', description: 'Reads', input_schema: { type: 'object', properties: {} } }],
  max_tokens: 128,
  stop_sequences: ['HALT'],
};

test('anthropic tool_result tail: system joins with a newline and the empty user text stays empty', () => {
  const neutral = neutralFromAnthropic(anthropicToolResult);
  assert.equal(neutral.systemText, 'Line one.\nLine two.');
  assert.equal(neutral.userText, '');
  // The openai rendering emits the tool message and an empty user message for the turn, so the
  // provider reads "" too; without the empty message it would read "Read the file." and fail.
  const rendered = renderOpenAI(neutral, 'm');
  assert.equal(wireTexts('openai', rendered).userText, '');
  assert.equal(wireTexts('openai', rendered).systemText, 'Line one.\nLine two.');
  assert.equal(wireTexts('gemini', renderGemini(neutral, 'm')).userText, '');
  const messages = rendered.messages as Json[];
  const toolMessage = messages.find((m) => m.role === 'tool');
  assert.equal(toolMessage?.content, 'file body');
});

test('an image crosses wires as the same bytes', () => {
  const body: Json = {
    model: 'auto',
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'What is this?' },
          { type: 'image', source: { type: 'base64', media_type: 'image/png', data: IMAGE_BYTES } },
        ],
      },
    ],
    max_tokens: 64,
  };
  const neutral = neutralFromAnthropic(body);
  const openAI = renderOpenAI(neutral, 'm');
  const messages = openAI.messages as Json[];
  const content = (messages[0].content ?? []) as Json[];
  const image = content.find((b) => b.type === 'image_url');
  assert.equal((image?.image_url as Json).url, `data:image/png;base64,${IMAGE_BYTES}`);
  const gemini = renderGemini(neutral, 'm');
  const contents = gemini.contents as Json[];
  const parts = contents[0].parts as Json[];
  const inline = parts.find((p) => p.inlineData !== undefined);
  assert.equal((inline?.inlineData as Json).data, IMAGE_BYTES);
  assert.equal(wireTexts('openai', openAI).userText, 'What is this?');
});

test('json mode reaches gemini as a response mime type and openai limits fall through', () => {
  const body: Json = {
    model: 'auto',
    messages: [{ role: 'user', content: 'Give me JSON.' }],
    response_format: { type: 'json_object' },
    max_completion_tokens: 77,
    stop: 'FIN',
  };
  const neutral = neutralFromOpenAI(body);
  assert.equal(neutral.jsonMode, true);
  assert.equal(neutral.maxTokens, 77);
  assert.deepEqual(neutral.stops, ['FIN']);
  const gemini = renderGemini(neutral, 'm');
  assert.equal((gemini.generationConfig as Json).responseMimeType, 'application/json');
  assert.equal((gemini.generationConfig as Json).maxOutputTokens, 77);
  assert.deepEqual((gemini.generationConfig as Json).stopSequences, ['FIN']);
  // No invented values: an anthropic request without limits renders no max_tokens anywhere.
  const bare = neutralFromAnthropic({ model: 'auto', messages: [{ role: 'user', content: 'hi' }] });
  assert.equal(bare.maxTokens, null);
  assert.equal('max_tokens' in renderOpenAI(bare, 'm'), false);
  assert.equal('generationConfig' in renderGemini(bare, 'm'), false);
});

test('an answer crosses from an openai stream to an anthropic stream intact', () => {
  const providerStream =
    'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n' +
    'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"ANSWER ECHO{}"},"finish_reason":null}]}\n\n' +
    'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n' +
    'data: {"id":"x","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":41,"completion_tokens":9,"total_tokens":50}}\n\n' +
    'data: [DONE]\n\n';
  const answer = answerFromOpenAI(providerStream, true);
  assert.ok(answer !== null);
  assert.equal(answer.text, 'ANSWER ECHO{}');
  assert.equal(answer.inputTokens, 41);
  assert.equal(answer.outputTokens, 9);
  const back = answerFromAnthropic(anthropicAnswerStream(answer, 'm'), true);
  assert.ok(back !== null);
  assert.equal(back.text, answer.text);
  assert.equal(back.inputTokens, 41);
  assert.equal(back.outputTokens, 9);
  assert.equal(anthropicStop(back.stop), 'end_turn');
});

test('an answer crosses from a gemini body to an openai body intact', () => {
  const geminiBody = JSON.stringify({
    candidates: [{ content: { role: 'model', parts: [{ text: 'ANSWER ' }, { text: 'ECHO{}' }] }, finishReason: 'STOP', index: 0 }],
    usageMetadata: { promptTokenCount: 12, candidatesTokenCount: 4, totalTokenCount: 16 },
  });
  const answer = answerFromGemini(geminiBody, false);
  assert.ok(answer !== null);
  assert.equal(answer.text, 'ANSWER ECHO{}');
  const rendered = openAIAnswerJson(answer, 'served-model');
  assert.equal(rendered.model, 'served-model');
  const choices = rendered.choices as Json[];
  assert.equal((choices[0].message as Json).content, 'ANSWER ECHO{}');
  assert.equal(choices[0].finish_reason, 'stop');
  assert.equal((rendered.usage as Json).prompt_tokens, 12);
  assert.equal(openAIStop('MAX_TOKENS'), 'length');
  assert.equal(anthropicStop('length'), 'max_tokens');
  // The openai stream rendering parses back to the same answer on the openai wire.
  const roundTrip = answerFromOpenAI(openAIAnswerStream(answer, 'm'), true);
  assert.ok(roundTrip !== null);
  assert.equal(roundTrip.text, answer.text);
  assert.equal(roundTrip.outputTokens, 4);
});
