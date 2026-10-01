#!/usr/bin/env node
// Check the browser illustration without a browser or native library dependency.
// These checks do not certify the Rust library, neural training, or performance.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

async function main() {
  const source = fs.readFileSync(path.join(__dirname, '..', 'docs', 'site', 'home.js'), 'utf8');
  const demo = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
  assert.equal(demo.HUE_MODULUS, 391680);

  // Independent expected bytes at the RGB wheel's corners and byte boundaries.
  const vectors = [
    [0, [255, 0, 0, 0]],
    [255, [255, 0, 0, 255]],
    [256, [255, 1, 0, 0]],
    [65279, [255, 254, 0, 255]],
    [65280, [255, 255, 0, 0]],
    [130560, [0, 255, 0, 0]],
    [195840, [0, 255, 255, 0]],
    [261120, [0, 0, 255, 0]],
    [326400, [255, 0, 255, 0]],
    [391679, [255, 0, 1, 255]],
  ];
  for (const [phase, token] of vectors) {
    assert.deepEqual(demo.encodePhase(phase), token, `Phase ${phase} must encode to the expected bytes.`);
    assert.equal(demo.decodeToken(token), phase, `Token ${token} must decode to the expected phase.`);
  }
  for (let phase = 0; phase < demo.HUE_MODULUS; phase += 1) {
    assert.equal(demo.decodeToken(demo.encodePhase(phase)), phase, `Round trip failed at phase ${phase}.`);
  }

  const invalidPhases = [-1, 391680, 1.5, NaN, Infinity];
  const invalidTokens = [[0, 0, 0, 0], [255, 255, 255, 0], [255, 0, 0], [256, 0, 0, 0], [255, 0, 0, -1]];
  for (const phase of invalidPhases) assert.throws(() => demo.encodePhase(phase), RangeError);
  for (const token of invalidTokens) assert.throws(() => demo.decodeToken(token), RangeError);

  // Hand-calculated output pairs for the documented fixed network.
  const networks = [[0, [0.028, 0.392]], [0.6, [0.574, 0.464]], [1, [0.938, 0.512]]];
  for (const [input, expected] of networks) {
    const actual = demo.evaluateNetwork(input);
    actual.outputs.forEach((value, index) => {
      assert.ok(Math.abs(value - expected[index]) < 1e-12, `Output ${index} differs for input ${input}.`);
    });
    assert.equal(actual.hidden[2], 0, 'ReLU must zero the negative third hidden sum.');
  }
  for (const input of [-0.01, 1.01, NaN, Infinity]) {
    assert.throws(() => demo.evaluateNetwork(input), RangeError);
  }
  assert.deepEqual(demo.encodePhase((391679 + 1) % demo.HUE_MODULUS), [255, 0, 0, 0]);
  assert.deepEqual(demo.encodePhase((391679 + 391679) % demo.HUE_MODULUS), [255, 0, 1, 254]);

  console.log(JSON.stringify({
    scope: 'Browser illustration math only',
    status: 'passed',
    phaseRoundTrips: demo.HUE_MODULUS,
    boundaryVectors: vectors.length,
    invalidPhases: invalidPhases.length,
    invalidTokens: invalidTokens.length,
    networkCases: networks.length,
    wrapCases: 2,
  }, null, 2));
}

main().catch(error => {
  console.error(error.message);
  process.exitCode = 1;
});
