Add-Type -AssemblyName System.Speech

$queries = @{
    "frontend_react_hooks" = "So, moving on to the next topic... I was looking at your resume, and I see you've used React quite a bit. Could you walk me through how you would decide between using a custom hook versus a higher order component for sharing stateful logic?";
    "fullstack_db_scaling" = "Alright, let's talk about scaling. Imagine we have a monolithic application backed by a single relational database that's starting to slow down under heavy read traffic. What are some strategies you would consider to alleviate that bottleneck?";
    "technical_rust_ownership" = "Okay, I noticed you have some experience with systems programming. In Rust, one of the core concepts is ownership and borrowing. Can you explain to me what the borrow checker does and why it's important for memory safety?";
    "general_agile_conflict" = "In a typical sprint, disagreements can happen. Tell me about a time when you disagreed with a senior engineer or product manager about a technical decision. How did you approach the situation and what was the outcome?";
    "frontend_css_grid_flex" = "Let's pivot to some CSS basics. A lot of developers get confused between CSS Grid and Flexbox. Can you explain the primary differences between the two, and give an example of a layout where you would definitely choose Grid over Flexbox?";
    "fullstack_auth_jwt" = "When building a secure REST API, authentication is critical. Could you explain how JSON Web Tokens work, and what the potential security risks are if you store a JWT in the browser's local storage instead of an HTTP-only cookie?";
    "technical_algo_complexity" = "I'd like to ask a quick algorithmic question. If you are traversing a binary search tree to find a specific node, what is the expected time complexity in Big-O notation, and what is the worst-case scenario if the tree is unbalanced?";
    "general_code_review" = "Code reviews are a big part of our engineering culture here. What do you consider to be the most important aspects to look for when reviewing a colleague's pull request, besides just checking for syntax errors?";
    "fullstack_api_design" = "Imagine we are designing a new public-facing API for our mobile app. What are some of the key principles you would follow to ensure the API is robust, versionable, and provides a good developer experience for the frontend team?";
    "technical_sysdesign_cache" = "Let's do a quick system design thought experiment. We are building a high-traffic news website. Explain how you would implement a caching layer, and what cache eviction policies you might use to keep the content fresh."
}

$outDir = "tests\fixtures\auto_ground_truth"
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
# We need 16kHz PCM for tests
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)

foreach ($key in $queries.Keys) {
    $outFile = Join-Path $outDir "$key.wav"
    Write-Host "Generating $outFile ..."
    $synth.SetOutputToWaveFile((Resolve-Path -Path $outDir).Path + "\$key.wav", $format)
    # Slow down the speech rate slightly to simulate real talking and make it longer (default is 0, range -10 to 10)
    $synth.Rate = -1
    $synth.Speak($queries[$key])
    $synth.SetOutputToNull()
}

Write-Host "Done generating 10 interview test fixtures!"
