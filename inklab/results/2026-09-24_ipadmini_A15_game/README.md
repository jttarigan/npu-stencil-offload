# iPad mini 6 (A15), in-game runs, 2026-09-24 — the fair RQ4 test

The A15 keeps the WHOLE ink graph on the neural engine (placement logged the
same evening, ../2026-09-19_ipad_mini_A15), unlike the A14 (iPhone 12), which
sends resampling to the CPU.

## Protocol
- Scripted play: a build-time autoplay option of the game (not part of this
  archive) steers every frame to the nearest pellet with the same input a
  tap produces; Sparks forced to x1; random seed 0x1A4B5EED; idle timer
  disabled. Checked in the simulator first (won 30/30, finale, CSV).
- Two rounds, each off -> gpu -> npu -> npu4 -> npua, one full game per backend,
  installed and launched over Wi-Fi, unattended (author away). Round 1
  19:23-19:33, round 2 19:33-19:43. iPadOS 26.7.
- The game is iPhone-only (TARGETED_DEVICE_FAMILY = 1): the iPad runs it in
  compatibility mode at a phone-sized drawable, hence GPU ~2.7-3.7 ms/frame
  (iPhone 12: 10.7-12.4).
- Whether each scripted game ended by winning (30) or by the countdown is not
  in the CSV; frame counts (4300-5550 play frames) match the simulator check,
  which won.

## Result (play phase; game_auto.md = round 1, game_auto_r2.md = round 2)
backend   GPU ms r1 / r2    returned vs gpu r1 / r2   fluid on render thread   long frames
off       2.703 / 2.698     -                          -                        0.1 / 0.1%
gpu       3.715 / 3.666     -                          -                        0.0 / 0.1%
npu       3.145 / 3.131     0.570 / 0.535              1.45 / 1.45 (max 8.6)    0.0 / 0.0%
npu4      3.186 / 3.180     0.529 / 0.486              0.84 / 0.84 (max 7.8)    0.0 / 0.0%
npua      3.175 / 3.140     0.540 / 0.526              0.14 / 0.14 (max 1.2)    0.1 / 0.0%
All p50 16.67, p99 <= 16.89 ms.

## Reading
- ON THE A15 THE NE RETURNS GPU TIME: about half of the prepass's ~1.0 ms
  (0.49-0.57 ms/frame), replicated in both rounds (GPU within 0.05 ms).
  The rest is plausibly drawing/sampling the ink, which every ink backend pays
  and the off build skips (not timed separately).
- Sync call on the render thread 1.45 ms (A14: 4.7); batching lowers it to 0.84
  (A14: rises to 5.4) -> the resampling fallback was the A14's problem.
- No backend drops frames here (plenty of headroom in compat mode).
- The script's "does not pay" labels for npu/npu4 come from game_frames.py's
  thresholds on render-thread time; read the numbers, not the label.
- Caveats: scripted, not human, play; compat-mode drawable; A19 Pro not yet run
  in game (could be run with the same script, no player needed).

## Files
game_auto/ (round 1 CSVs), game_auto_r2/ (round 2), game_auto.md, game_auto_r2.md.
