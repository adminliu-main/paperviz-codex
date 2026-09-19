# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for interacting with Gemini and Claude APIs, image processing, and PDF handling.
"""

import json
import asyncio
import base64
from datetime import datetime, timezone
from io import BytesIO
from functools import partial
from ast import literal_eval
from typing import List, Dict, Any
import tempfile
import uuid
import shutil
import subprocess

import aiofiles
from PIL import Image
from google import genai
from google.genai import types
from anthropic import AsyncAnthropicVertex
from openai import AsyncOpenAI

import os

import yaml
from pathlib import Path

# Load config
config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
model_config = {}
if config_path.exists():
    with open(config_path, "r") as f:
        model_config = yaml.safe_load(f) or {}

def get_config_val(section, key, env_var, default=""):
    val = os.getenv(env_var)
    if not val and section in model_config:
        val = model_config[section].get(key)
    return val or default


def write_openai_image_debug_log(event: str, **details: Any) -> None:
    """Persist opt-in image API diagnostics without secrets or image payloads."""
    if os.getenv("PAPERVIZ_DEBUG_LOG", "").lower() not in {"1", "true", "yes"}:
        return

    debug_path = Path(__file__).parent.parent / "logs" / "openai_image_debug.jsonl"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **details,
    }
    try:
        with open(debug_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except OSError as e:
        print(f"Warning: Could not write image debug log: {e}")


def use_codex_backend() -> bool:
    """Whether agent reasoning should use the user's logged-in Codex CLI."""
    backend = get_config_val("defaults", "backend", "PAPERVIZ_BACKEND", "api")
    return str(backend).lower() == "codex"


def _codex_config(key: str, env_var: str, default: Any) -> Any:
    return get_config_val("codex", key, env_var, default)


def _content_to_codex_prompt(contents: List[Dict[str, Any]], config: Any) -> tuple[str, List[tuple[bytes, str]]]:
    """Translate the agents' provider-neutral content objects for `codex exec`."""
    system_prompt = getattr(config, "system_instruction", None)
    if not system_prompt and isinstance(config, dict):
        system_prompt = config.get("system_prompt", "")

    text_blocks = [
        "You are running inside an automated pipeline. Follow the requested output format exactly; do not explain your process or modify project files.",
        f"SYSTEM INSTRUCTIONS:\n{system_prompt or ''}",
        "USER CONTENT:",
    ]
    images: List[tuple[bytes, str]] = []
    for item in contents:
        if item.get("type") == "text":
            text_blocks.append(item.get("text", ""))
        elif item.get("type") == "image":
            source = item.get("source", {})
            image_data = source.get("data") or item.get("image_base64")
            if image_data:
                try:
                    images.append((base64.b64decode(image_data), source.get("media_type", "image/jpeg")))
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not decode Codex image attachment: {e}")
    return "\n\n".join(text_blocks), images


async def _run_codex_exec(
    prompt: str,
    image_attachments: List[tuple[bytes, str]] | None = None,
    writable_dir: Path | None = None,
) -> tuple[str, str, int]:
    """Run the locally authenticated Codex CLI and return stdout, stderr, status."""
    command = str(_codex_config("command", "CODEX_COMMAND", "codex"))
    timeout_seconds = int(_codex_config("timeout_seconds", "CODEX_TIMEOUT_SECONDS", 300))
    codex_model = str(_codex_config("model", "CODEX_MODEL", "")).strip()
    attachments = image_attachments or []

    with tempfile.TemporaryDirectory(prefix="paperviz-codex-") as temp_dir_str:
        runtime_dir = Path(temp_dir_str)
        image_paths = []
        for index, (image_bytes, media_type) in enumerate(attachments):
            suffix = ".png" if "png" in media_type else ".jpg"
            image_path = runtime_dir / f"input-{index}{suffix}"
            image_path.write_bytes(image_bytes)
            image_paths.append(image_path)

        work_dir = writable_dir or runtime_dir
        command_args = [
            "exec", "--ephemeral", "--skip-git-repo-check",
            "--sandbox", "workspace-write" if writable_dir else "read-only",
            "--cd", str(work_dir),
        ]
        if codex_model:
            command_args.extend(["--model", codex_model])
        for image_path in image_paths:
            command_args.extend(["--image", str(image_path)])

        resolved_command = shutil.which(command) or command
        if os.name == "nt" and resolved_command.lower().endswith((".cmd", ".bat")):
            # npm installs Codex as codex.cmd on Windows. cmd.exe is required
            # to invoke that wrapper from Python's subprocess APIs reliably.
            comspec = os.environ.get("COMSPEC", "cmd.exe")
            process_args = [
                comspec, "/d", "/s", "/c",
                subprocess.list2cmdline([resolved_command, *command_args]),
            ]
        else:
            process_args = [resolved_command, *command_args]

        try:
            process = await asyncio.create_subprocess_exec(
                *process_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return "", f"Codex CLI command not found: {command}. Install Codex and run `codex login` first.", 127

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(prompt.encode("utf-8")), timeout=timeout_seconds
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            return "", f"Codex CLI timed out after {timeout_seconds} seconds.", 124

        return stdout.decode("utf-8", errors="replace").strip(), stderr.decode("utf-8", errors="replace").strip(), process.returncode


async def call_codex_with_retry_async(
    contents: List[Dict[str, Any]], config: Any, max_attempts=3, retry_delay=5, error_context=""
) -> List[str]:
    """Use `codex exec` for the text and visual-reasoning agents without an API key."""
    prompt, attachments = _content_to_codex_prompt(contents, config)
    for attempt in range(max_attempts):
        output, stderr, return_code = await _run_codex_exec(prompt, attachments)
        if return_code == 0 and output:
            return [output]

        context_msg = f" for {error_context}" if error_context else ""
        details = stderr[-1000:] if stderr else "no final response"
        print(f"Codex attempt {attempt + 1} failed{context_msg} (exit {return_code}): {details}")
        if attempt < max_attempts - 1:
            await asyncio.sleep(retry_delay)
    return ["Error"]


async def call_codex_svg_generation_async(
    prompt: str, system_prompt: str, aspect_ratio: str, output_root: Path
) -> List[str]:
    """Ask Codex to author an SVG, then convert it to PNG for the existing pipeline."""
    output_dir = output_root / uuid.uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "diagram.svg"
    codex_prompt = f"""{system_prompt}

You are the image-generation stage of an automated scientific-figure pipeline.
Create one polished, editable vector diagram as SVG and save it exactly to:
{svg_path}

Requirements:
- Use only SVG elements, embedded CSS, and standard fonts; do not use external assets, network access, scripts, or raster images.
- Use a {aspect_ratio} canvas and a white background.
- Render the requested scientific content accurately with concise labels.
- Do not include a figure title unless the description explicitly requires one.
- Do not modify any file outside the current working directory.
- Your final response must be exactly: DONE

Detailed description to render:
{prompt}
"""
    output, stderr, return_code = await _run_codex_exec(codex_prompt, writable_dir=output_dir)
    if return_code != 0 or not svg_path.exists() or svg_path.stat().st_size == 0:
        details = stderr[-1000:] if stderr else output[-1000:]
        print(f"Codex SVG generation failed (exit {return_code}): {details}")
        return ["Error"]

    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(url=str(svg_path))
        return [base64.b64encode(png_bytes).decode("utf-8")]
    except ImportError:
        print("Codex created diagram.svg, but SVG conversion needs cairosvg. Run `uv pip install cairosvg`.")
    except Exception as e:
        print(f"Could not convert Codex SVG to PNG: {e}")
    return ["Error"]


async def call_codex_image_generation_async(
    prompt: str, system_prompt: str, aspect_ratio: str, output_root: Path
) -> List[str]:
    """Use Codex's built-in image_generation tool and return its PNG as base64."""
    output_dir = output_root / uuid.uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "diagram.png"
    codex_prompt = f"""{system_prompt}

You are the image-generation stage of an automated scientific-figure pipeline.
Use the built-in image_generation tool to create one raster scientific diagram.
Do NOT create SVG, HTML, Canvas, Mermaid, Matplotlib, or any code-rendered substitute.
Do NOT call an external API or ask for an API key.

Image requirements:
- Use case: scientific-educational.
- Aspect ratio: {aspect_ratio}; white background; publication-quality academic methodology figure.
- Render requested scientific content accurately with concise, legible labels.
- Do not include a figure title unless the description explicitly requires one.
- No watermarks, decorative icons, photorealistic elements, or unrelated text.

After generation, copy the selected generated image from Codex's generated-images
location into this exact project path:
{image_path}
Verify that this file exists and is non-empty. Do not modify any other project file.
Your final response must be exactly: DONE

Detailed description to render:
{prompt}
"""
    output, stderr, return_code = await _run_codex_exec(codex_prompt, writable_dir=output_dir)
    if return_code != 0 or not image_path.exists() or image_path.stat().st_size == 0:
        details = stderr[-1000:] if stderr else output[-1000:]
        print(f"Codex native image generation failed (exit {return_code}): {details}")
        return ["Error"]

    try:
        return [base64.b64encode(image_path.read_bytes()).decode("utf-8")]
    except OSError as e:
        print(f"Could not read Codex generated image: {e}")
        return ["Error"]

# Endpoint configuration is intentionally separate from credentials: a proxy can
# use the same provider key while routing requests to its own gateway.
gemini_base_url = get_config_val("endpoints", "gemini_base_url", "GEMINI_BASE_URL", "")
openai_base_url = get_config_val("endpoints", "openai_base_url", "OPENAI_BASE_URL", "")
google_api_key = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or model_config.get("api_keys", {}).get("google_api_key", "")
)
project_id = get_config_val("google_cloud", "project_id", "GOOGLE_CLOUD_PROJECT", "")
location = get_config_val("google_cloud", "location", "GOOGLE_CLOUD_LOCATION", "global")

# Initialize clients lazily or with robust defaults.
# A configured Gemini endpoint takes precedence over Vertex ADC because a
# third-party gateway normally exposes the Gemini Developer API, not Vertex.
if use_codex_backend():
    print("Initialized local Codex CLI backend; cloud model credentials are not required.")
    gemini_client = None
elif gemini_base_url:
    if not google_api_key:
        print("Warning: GEMINI_BASE_URL is configured but GOOGLE_API_KEY/GEMINI_API_KEY is missing.")
        gemini_client = None
    else:
        try:
            gemini_client = genai.Client(
                api_key=google_api_key,
                http_options=types.HttpOptions(base_url=gemini_base_url),
            )
            print(f"Initialized Gemini Client with custom endpoint: {gemini_base_url}")
        except Exception as e:
            print(f"Warning: Could not initialize Gemini Client with custom endpoint: {e}")
            gemini_client = None
else:
    try:
        import google.auth
        creds, _ = google.auth.default()
        if not hasattr(creds, "service_account_email"):
            print(f"DEBUG: Running with credentials: {type(creds)}")
        print(f"DEBUG: Initialized Gemini Client with Project: {project_id}, Location: {location}")

        # Try Vertex AI first (preferred for Cloud Run)
        gemini_client = genai.Client(vertexai=True, project=project_id, location=location)
    except ValueError:
        # Fallback to API Key if Vertex fails (e.g. local dev without ADC)
        if google_api_key:
            gemini_client = genai.Client(api_key=google_api_key)
            print("Initialized Gemini Client with API Key")
        else:
            print("Warning: Could not initialize Gemini Client. Missing credentials.")
            gemini_client = None

anthropic_project_id = get_config_val("anthropic", "project_id", "ANTHROPIC_PROJECT_ID", project_id)
anthropic_region = get_config_val("anthropic", "region", "ANTHROPIC_REGION", "us-central1")
if use_codex_backend():
    anthropic_client = None
else:
    try:
        anthropic_client = AsyncAnthropicVertex(region=anthropic_region, project_id=anthropic_project_id)
    except Exception as e:
        print(f"Warning: Could not initialize Anthropic Vertex Client: {e}")
        anthropic_client = None

if use_codex_backend():
    openai_client = None
else:
    try:
        openai_api_key = get_config_val("api_keys", "openai_api_key", "OPENAI_API_KEY", "")
        openai_client_args = {"api_key": openai_api_key} if openai_api_key else {}
        if openai_base_url:
            openai_client_args["base_url"] = openai_base_url
        openai_client = AsyncOpenAI(**openai_client_args) # Falls back to OPENAI_API_KEY when omitted.
        if openai_base_url:
            print(f"Initialized OpenAI Client with custom endpoint: {openai_base_url}")
    except Exception as e:
        print(f"Warning: Could not initialize OpenAI Client: {e}")
        openai_client = None



def _convert_to_gemini_parts(contents: List[Dict[str, Any]]) -> List[types.Part]:
    """
    Convert a generic content list to a list of Gemini's genai.types.Part objects.
    """
    gemini_parts = []
    for item in contents:
        if item.get("type") == "text":
            gemini_parts.append(types.Part.from_text(text=item["text"]))
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(source["data"]),
                        mime_type=source["media_type"],
                    )
                )
    return gemini_parts


async def call_gemini_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context=""
):
    """
    ASYNC: Call Gemini API with asynchronous retry logic.
    """
    if use_codex_backend():
        return await call_codex_with_retry_async(
            contents=contents,
            config=config,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    result_list = []
    target_candidate_count = config.candidate_count
    # Gemini API max candidate count is 8. We will call multiple times if needed.
    if config.candidate_count > 8:
        config.candidate_count = 8

    current_contents = contents
    for attempt in range(max_attempts):
        try:
            # Use global client
            client = gemini_client
            
            # Convert generic content list to Gemini's format right before the API call
            gemini_contents = _convert_to_gemini_parts(current_contents)
            response = await client.aio.models.generate_content(
                model=model_name, contents=gemini_contents, config=config
            )

            # If we are using Image Generation models to generate images
            if (
                "nanoviz" in model_name
                or "image" in model_name
            ):
                raw_response_list = []
                if not response.candidates or not response.candidates[0].content.parts:
                    print(
                        f"[Warning]: Failed to generate image, retrying in {retry_delay} seconds..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue

                # In this mode, we can only have one candidate
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        # Append base64 encoded image data to raw_response_list
                        raw_response_list.append(
                            base64.b64encode(part.inline_data.data).decode("utf-8")
                        )
                        break

            # Otherwise, for text generation models
            else:
                raw_response_list = [
                    part.text
                    for candidate in response.candidates
                    for part in candidate.content.parts
                ]
            result_list.extend([r for r in raw_response_list if r.strip() != ""])
            if len(result_list) >= target_candidate_count:
                result_list = result_list[:target_candidate_count]
                break

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            
            # Exponential backoff (capped at 30s)
            current_delay = min(retry_delay * (2 ** attempt), 30)
            
            print(
                f"Attempt {attempt + 1} for model {model_name} failed{context_msg}: {e}. Retrying in {current_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                result_list = ["Error"] * target_candidate_count

    if len(result_list) < target_candidate_count:
        result_list.extend(["Error"] * (target_candidate_count - len(result_list)))
    return result_list

def _convert_to_claude_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list to Claude's API format.
    Currently, the formats are identical, so this acts as a pass-through
    for architectural consistency and future-proofing.

    Claude API's format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]
    """
    return contents


def _convert_to_openai_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list (Claude format) to OpenAI's API format.
    
    Claude format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]
    
    OpenAI format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        ...
    ]
    """
    openai_contents = []
    for item in contents:
        if item.get("type") == "text":
            openai_contents.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                # OpenAI expects data URL format
                data_url = f"data:{media_type};base64,{data}"
                openai_contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
    return openai_contents


async def call_claude_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call Claude API with asynchronous retry logic.
    This version efficiently handles input size errors by validating and modifying
    the content list once before generating all candidates.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_output_tokens = config["max_output_tokens"]
    response_text_list = []

    # --- Preparation Phase ---
    # Convert to the Claude-specific format and perform an initial optimistic resize.
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    # Note that this check is required because Claude only has 128k / 256k context windows.
    # For Gemini series that support 1M, we do not need this step.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            claude_contents = _convert_to_claude_format(current_contents)
            # Attempt to generate the very first candidate.
            first_response = await anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": claude_contents}],
                system=system_prompt,
            )
            response_text_list.append(first_response.content[0].text)
            is_input_valid = True
            break

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_claude_contents = _convert_to_claude_format(current_contents)
        tasks = [
            anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": valid_claude_contents}
                ],
                system=system_prompt,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.content[0].text)

    return response_text_list

async def call_openai_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI API with asynchronous retry logic.
    This follows the same pattern as Claude's implementation.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_completion_tokens = config["max_completion_tokens"]
    response_text_list = []

    # --- Preparation Phase ---
    # Convert to the OpenAI-specific format
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            openai_contents = _convert_to_openai_format(current_contents)
            # Attempt to generate the very first candidate.
            first_response = await openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            # If we reach here, the input is valid.
            response_text_list.append(first_response.choices[0].message.content)
            is_input_valid = True
            break  # Exit the validation loop

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_openai_contents = _convert_to_openai_format(current_contents)
        tasks = [
            openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": valid_openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.choices[0].message.content)

    return response_text_list


async def call_openai_image_generation_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI Image Generation API (GPT-Image) with asynchronous retry logic.
    """
    size = config.get("size", "1536x1024")
    quality = config.get("quality", "high")
    background = config.get("background", "opaque")
    output_format = config.get("output_format", "png")
    
    # Base parameters for all models
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "n": 1,
        "size": size,
    }
    
    # Add GPT-Image specific parameters
    gen_params.update({
        "quality": quality,
        "background": background,
        "output_format": output_format,
    })

    for attempt in range(max_attempts):
        try:
            response = await openai_client.images.generate(**gen_params)
            
            # OpenAI images.generate returns a list of images in response.data
            if response.data and response.data[0].b64_json:
                return [response.data[0].b64_json]
            else:
                response_data = []
                for item in response.data or []:
                    # Deliberately omit b64_json: it can be very large and may
                    # contain the generated image itself.
                    if hasattr(item, "model_dump"):
                        item_data = item.model_dump(exclude={"b64_json"}, exclude_none=True)
                    else:
                        item_data = {"type": type(item).__name__}
                    response_data.append(item_data)
                debug_details = {
                    "model": model_name,
                    "data_count": len(response.data or []),
                    "response_data": response_data,
                }
                print(f"[Warning]: Failed to generate image via OpenAI, no b64_json returned: {debug_details}")
                write_openai_image_debug_log("empty_image_response", **debug_details)
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            write_openai_image_debug_log(
                "image_request_error",
                model=model_name,
                attempt=attempt + 1,
                error=str(e),
            )
            print(
                f"Attempt {attempt + 1} for OpenAI image generation model {model_name} failed{context_msg}: {e}. Retrying in {retry_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]
