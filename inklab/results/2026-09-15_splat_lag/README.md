# Perceptual cost of batching splats for the unrolled graph (2026-09-15)

trace/lag_check.py on the recorded game (gameplay_script.json, seed 12345,
2,074 steps, finale from step 1,749), float32 NumPy reference throughout so
the temporal artefact is isolated from half-precision effects.

The unrolled program needs K frames of splat fields per submission and cannot
know future splats, so the displayed field is stale. Two presentations:
  DELAY  emit all K intermediate dye fields, show one per frame, K frames
         late. Smooth motion, constant lag.
  HOLD   emit only the final field and show it for K frames. Same average
         staleness, but the background updates at 30/K Hz.

## Staleness (dye PSNR of the displayed field against the true field, dB)
        play median / 5th pct      finale median / 5th pct
K=2  delay  43.2 / 35.6                51.4 / 29.0
K=2  hold   49.2 / 41.0                57.5 / 30.3
K=4  delay  37.6 / 31.5                45.4 / 25.8
K=4  hold   43.7 / 33.6                52.8 / 27.8
K=8  delay  32.8 / 28.9                39.5 / 22.5
K=8  hold   38.2 / 30.9                47.8 / 25.2

HOLD scores better here only because it is stale by K/2 on average rather
than K. This metric cannot see stutter, so it must not be read alone.

## Smoothness (PSNR between consecutive DISPLAYED frames, play phase)
shipped and delay: 49.2 dB every frame, no jumps.
hold K=2: unchanged 1 frame in 2, then a 43.4 dB jump 15.0 times a second.
hold K=4: unchanged 3 frames in 4, then a 37.9 dB jump 7.5 times a second.
hold K=8: unchanged 7 frames in 8, then a 33.0 dB jump 3.75 times a second.
A 33 dB jump 3.75 times a second in a background that otherwise changes by
49 dB per frame is visible stutter. DELAY is the only usable presentation,
and it costs one extra output tensor of K x 3 x H x W (1.5 MB at K=8).

## Head lag of the delay scheme (iPhone 12 geometry, as recorded)
K=2:  5.03 sim texels =  46 screen px = 0.51 grid cells = 0.067 s
K=4: 10.07 sim texels =  92 screen px = 1.02 grid cells = 0.133 s
K=8: 19.18 sim texels = 175 screen px = 1.95 grid cells = 0.267 s

## Reading
Against the same float32 reference the half-precision neural engine path
already sits at 29 to 33 dB. The K=8 delay scheme sits at 32.8 dB in play,
i.e. the lag changes the displayed field about as much as the precision
change the project has already accepted, and K=4 (37.6 dB) is clearly
milder. The ink field itself looks unchanged: lag_ab.png shows current vs
8-frames-stale side by side with the difference amplified 4x, and the
difference is thin edge outlines, not structure.

## Limitation
This measures the ink field alone. In the game the ink is composited under
the board, and the artefact a player would actually notice is the ink's
position relative to the head, which at K=8 is about two grid cells behind.
Whether two cells of trail displacement reads as wrong on a lava-lamp
background is a judgement this data cannot settle; it needs the phone build
and eyes on it. K=4, at one cell, is the conservative choice.
