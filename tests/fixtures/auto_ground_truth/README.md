Auto Mode smoke benchmark fixtures.

These WAV files are deterministic SAPI-generated interview prompts used by
`benchmarks/auto_mode_benchmark.py`. They are committed so Auto Mode regressions
can be reproduced without regenerating local speech assets.

Primary smoke set:

- frontend_css_grid_flex.wav
- frontend_react_hooks.wav
- fullstack_api_design.wav
- fullstack_auth_jwt.wav
- fullstack_db_scaling.wav

Regenerate with:

```powershell
.\scripts\generate_test_audio.ps1
```
